import copy
import json
import sys
import numpy as np
import pandas as pd
import pytest
import torch
from uvarfs.cosine_cache import CachedCosineObjective
from uvarfs.cosine_uvarfs import CosineObjective
from uvarfs.performance_uvarfs import fit_performance_uvarfs
from scripts.analyze_results import completed_return_scope,audit,audit_sparse_path,source_hashes
from test_v18_refitted_exchange import normal_case,new_fixture
from test_v10_representation import CountingExtractor,CPUIndex


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('chunk',[1,7,512])
@pytest.mark.parametrize('zero_variability',[False,True])
def test_cached_loss_and_both_gradients_match_original_exactly(dtype,chunk,zero_variability):
    gen=torch.Generator().manual_seed(610);x=torch.randn(19,13,generator=gen,dtype=dtype)
    v=torch.zeros(3,13,dtype=dtype) if zero_variability else torch.rand(3,13,generator=gen,dtype=dtype)
    a=torch.tensor([0,2,4,5,8,12]);p=torch.rand(6,generator=gen,dtype=dtype);p/=p.sum()
    plain=CosineObjective(x,v,chunk=chunk);cached=CachedCosineObjective(x,v,chunk=chunk)
    for gradient,full in [(False,False),(True,False),(True,True),(True,True),(True,False),(False,False)]:
        expected=plain.evaluate(a,p,gradient,full);actual=cached.evaluate(a,p,gradient,full)
        torch.testing.assert_close(actual[0],expected[0],atol=0,rtol=0)
        if gradient:torch.testing.assert_close(actual[1],expected[1],atol=0,rtol=0)
        assert actual[2]==expected[2]
    d=cached.cache_diagnostics()
    assert (d['objective_requests'],d['geometry_evaluations'],d['geometry_reuses'])==(6,1,5)
    assert 0<d['peak_cache_bytes']<=d['cache_limit_bytes']
    p[0]=torch.nextafter(p[0],p.new_tensor(float('inf')))
    cached.evaluate(a,p);assert cached.geometry_evaluations==2
    cached.evaluate(a.flip(0),p.flip(0));assert cached.geometry_evaluations==3


def test_caller_mutations_and_invalid_weights_do_not_poison_state():
    x,v,_=normal_case();engine=CachedCosineObjective(x,v);a=torch.arange(8);p=torch.ones(8,dtype=x.dtype)/8
    value,g,terms=engine.evaluate(a,p,True);original=value.clone();gradient=g.clone()
    value.zero_();g.zero_();terms['geometry_error']=-1
    new=engine.evaluate(a,p,True)
    torch.testing.assert_close(new[0],original,rtol=0,atol=0)
    torch.testing.assert_close(new[1],gradient,rtol=0,atol=0)
    assert new[2]['geometry_error']>=0
    p[0]+=1e-3;engine.evaluate(a,p);assert engine.geometry_evaluations==2
    with pytest.raises(ValueError,match='invalid candidate'):engine.evaluate(a,-p)
    plain=CosineObjective(x,v)
    torch.testing.assert_close(engine.evaluate(a,p,True)[0],plain.evaluate(a,p,True)[0],atol=0,rtol=0)


@pytest.mark.parametrize('limit',[0,1e-6])
def test_disabled_or_oversized_cache_falls_back_to_original(limit):
    x,v,_=normal_case();engine=CachedCosineObjective(x,v,cache_mib=limit);plain=CosineObjective(x,v)
    a=torch.arange(8);p=torch.ones(8,dtype=x.dtype)/8
    for _ in range(3):
        actual=engine.evaluate(a,p,True);expected=plain.evaluate(a,p,True)
        torch.testing.assert_close(actual[0],expected[0],atol=0,rtol=0)
        torch.testing.assert_close(actual[1],expected[1],atol=0,rtol=0)
    assert (engine.geometry_evaluations,engine.geometry_reuses,engine.peak_cache_bytes)==(3,0,0)


@pytest.mark.parametrize('limit',[-1,float('inf'),float('nan')])
def test_invalid_memory_limits(limit):
    x,v,_=normal_case()
    with pytest.raises(ValueError,match='cache limit'):CachedCosineObjective(x,v,cache_mib=limit)


def test_complete_solve_keeps_same_pool_support_weights_and_solver_decisions():
    x,v,cfg=normal_case();gram={'budgeted_weights':np.ones(14),'active':np.arange(8)}
    plain=fit_performance_uvarfs(x,v,{**cfg,'performance_objective_cache_mib':0},gram)
    cached=fit_performance_uvarfs(x,v,{**cfg,'performance_objective_cache_mib':128},gram)
    for key in ['candidate_id','objective','geometry_error','solver_iterations','projected_residual']:
        assert cached[key]==plain[key]
    assert len(cached['candidate_pool'])==len(plain['candidate_pool'])
    for left,right in zip(plain['candidate_pool'],cached['candidate_pool']):
        np.testing.assert_array_equal(left['active'],right['active']);np.testing.assert_array_equal(left['weights'],right['weights'])
        assert left['base_objective']==right['base_objective']
    for left,right in zip(plain['exchange_search']['trajectories'],cached['exchange_search']['trajectories']):
        assert left['steps']==right['steps'] and left['termination']==right['termination']
        assert left['objective_evaluations']==right['objective_evaluations']
    audit_sparse_path(cached,cfg)
    d=cached['exchange_search']['objective_cache'];assert d['geometry_reuses']>0
    for change in [lambda c:c.__setitem__('peak_cache_bytes',c['cache_limit_bytes']+1),lambda c:c.__setitem__('geometry_reuses',0)]:
        bad=copy.deepcopy(cached);change(bad['exchange_search']['objective_cache'])
        with pytest.raises(ValueError,match='cache memory/cost'):audit_sparse_path(bad,cfg)


def completed_folder(root,name='brain',stage='complete',metrics=True):
    folder=root/name;folder.mkdir(parents=True)
    (folder/'eval_progress.json').write_text(json.dumps({'stage':stage}))
    if metrics:(folder/'metrics.csv').write_text('dataset,method\n'+name+',main\n')
    return folder


def test_partial_scope_requires_complete_state_and_exact_coverage(tmp_path):
    completed_folder(tmp_path);(tmp_path/'camelyon16').mkdir()
    frame=pd.DataFrame({'dataset':['brain'],'method':['main']})
    names,states=completed_return_scope(tmp_path,frame)
    assert names==['brain'] and len(states)==6
    assert {s['dataset']:s['status'] for s in states}['camelyon16']=='incomplete'
    for bad in [pd.concat([frame,frame]),frame.assign(dataset='liver'),frame.iloc[:0]]:
        with pytest.raises(ValueError,match='partial aggregate'):completed_return_scope(tmp_path,bad)
    completed_folder(tmp_path,'liver',metrics=False)
    with pytest.raises(ValueError,match='state disagree'):completed_return_scope(tmp_path,frame)


def test_partial_audit_preserves_sources_and_rejects_mixed_incomplete_metadata(tmp_path,monkeypatch):
    import scripts.run_all as runner
    import scripts.analyze_results as analyzer
    cfg,_,_=new_fixture(tmp_path);cfg['paths']['processed_data']=str(tmp_path);cfg['results_dir']=str(tmp_path/'output')
    cfg['uvarfs']['performance_objective_cache_mib']=128
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    monkeypatch.setattr(runner,'load_config',lambda _:cfg);monkeypatch.setattr(runner,'DINOv3Extractor',lambda *args:CountingExtractor())
    monkeypatch.setattr(runner,'discover_bmad_roots',lambda _: {'liver':tmp_path/'liver'})
    monkeypatch.setattr('uvarfs.pipeline_eval.make_index',lambda m,c:CPUIndex(m))
    monkeypatch.setattr(sys,'argv',['run_all','--dataset','liver']);runner.main()
    source=tmp_path/'output'
    # A partial server return retains its partial aggregate, not just metrics.
    (source/'all_metrics.csv').replace(source/'all_metrics.partial.csv')
    meta=json.loads((source/'liver'/'run_metadata.json').read_text())
    partial=source/'camelyon16';partial.mkdir();path=partial/'run_metadata.json';path.write_text(json.dumps({**meta,'dataset':'camelyon16'}))
    # Fixture code has no committed source snapshot: mark verification unavailable.
    original_check=analyzer.subprocess.check_output
    def unavailable(command,*args,**kwargs):
        if command[:2]==['git','ls-files']:return original_check(command,*args,**kwargs)
        raise OSError('constructed fixture')
    monkeypatch.setattr(analyzer.subprocess,'check_output',unavailable)
    before=source_hashes(source)
    with pytest.raises(FileNotFoundError):audit(source,tmp_path/'strict',100)
    out=tmp_path/'report';data=audit(source,out,100,completed_only=True)
    assert source_hashes(source)==before and data['completed_datasets']==['liver'] and data['all_six_complete'] is False
    assert data['analysis_scope']=='completed_datasets_only' and data['source_aggregate']=='all_metrics.partial.csv'
    assert data['rows']==164 and not (out/'all_metrics.csv').exists() and (out/'completed_metrics.csv').is_file()
    text=(out/'analysis.md').read_text(encoding='utf-8')
    assert '部分回传分析' in text and '1/6' in text and '不构成全六性能结论' in text and 'Main Macro Image AUROC' not in text
    path.write_text(json.dumps({**meta,'code_fingerprint':'different-code'}))
    with pytest.raises(ValueError,match='partial run provenance'):audit(source,tmp_path/'mixed',100,completed_only=True)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA runtime unavailable locally')
def test_cuda_cache_exact_same_device_loss_and_gradients():
    x,v,_=normal_case();x=x.cuda().float();v=v.cuda().float();plain=CosineObjective(x,v);cached=CachedCosineObjective(x,v)
    a=torch.arange(8,device=x.device);p=torch.ones(8,device=x.device)/8;cached.evaluate(a,p)
    expected=plain.evaluate(a,p,True,True);actual=cached.evaluate(a,p,True,True)
    torch.testing.assert_close(actual[0],expected[0],rtol=0,atol=0);torch.testing.assert_close(actual[1],expected[1],rtol=0,atol=0)
