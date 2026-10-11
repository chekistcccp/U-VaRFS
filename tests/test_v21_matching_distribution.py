import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F

from uvarfs.matching_uvarfs import (MatchingObjective, normal_temperature,
    generate_fit_only_pool, fit_matching_uvarfs)
from uvarfs.objectives import MATCHING, LOCAL, QUADRATIC, experiment_branches
from uvarfs.protocol import expected_method_names, EXPERIMENT_VERSION
from uvarfs.utils import load_config
from test_v20_local_selection import case, local_fixture
from test_v10_representation import CountingExtractor, CPUIndex


def independent_objective(x, g, active, p, tau, paired, beta, score_weight=1., distribution_weight=1.):
    """Independent autograd oracle: rebuild dense teacher and weighted detector."""
    x=x.double();mask=g[:,None]!=g[None,:]
    dense=F.normalize(x,dim=1)@F.normalize(x,dim=1).T
    teacher_cos=dense.masked_fill(~mask,-torch.inf)
    tlog=(teacher_cos/tau).log_softmax(1).masked_fill(~mask,0)
    t=(teacher_cos/tau).softmax(1)
    nearest=teacher_cos.max(1).values
    distance=(1-nearest).clamp_min(0)
    w=(1+distance/distance.mean().clamp_min(1e-4)).clamp_max(4);w=w/w.mean()
    score_scale=(w*distance.square()).mean().clamp_min(1e-4)
    scale=(w*(mask.sum(1).double().log()+(t*tlog).sum(1))).mean().clamp_min(1e-6)
    student=F.normalize(x[:,active]*p.sqrt(),dim=1)
    cosine=(student@student.T).masked_fill(~mask,-torch.inf)
    slogs=(cosine/tau).log_softmax(1).masked_fill(~mask,0)
    neighbor=(w*(t*(tlog-slogs)).sum(1)).mean()/scale
    score=(w*(cosine.max(1).values-nearest).square()).mean()/score_scale
    va,vb=paired[0].double().repeat(len(paired[1]),1),paired[1].double().reshape(-1,x.shape[1])
    vscale=(1-F.cosine_similarity(va,vb)).mean().clamp_min(1e-4)
    var=(1-F.cosine_similarity(va[:,active]*p.sqrt(),vb[:,active]*p.sqrt())).mean()/vscale
    return score_weight*score+distribution_weight*neighbor+beta*var


@pytest.mark.parametrize('score_weight,distribution_weight',[(1.,1.),(0.,1.),(1.,0.)])
def test_analytic_full_reference_gradient_matches_independent_autograd(score_weight,distribution_weight):
    x,_,g,_,paired=case();tau=normal_temperature(x,g)
    engine=MatchingObjective(x,x,g,g,tau,paired,beta=.13,chunk=7,
        score_weight=score_weight,distribution_weight=distribution_weight)
    active=torch.tensor([0,2,3,7,10]);p=torch.tensor([.1,.2,.3,.15,.25],dtype=torch.float64,requires_grad=True)
    value,grad,_=engine.evaluate(active,p,True)
    oracle=independent_objective(x,g,active,p,tau,paired,.13,score_weight,distribution_weight)
    expected=torch.autograd.grad(oracle,p)[0]
    assert value==pytest.approx(float(oracle),abs=1e-10)
    torch.testing.assert_close(grad,expected,rtol=1e-9,atol=1e-9)
    vv,full,_=engine.evaluate(active,p,True,True)
    assert vv==pytest.approx(value,abs=1e-10)
    torch.testing.assert_close(full[active],grad,rtol=1e-9,atol=1e-9)
    all_ids=torch.arange(x.shape[1]);wp=torch.zeros(x.shape[1],dtype=torch.float64);wp[active]=p.detach()
    delta=torch.zeros_like(wp);delta[1]=1;delta[0]=-1
    eps=1e-7
    assert (engine.evaluate(all_ids,wp+eps*delta)[0]-value)/eps==pytest.approx(float(full@delta),abs=1e-4)


def test_dense_identity_actual_student_nearest_and_chunk_equivalence():
    x,_,g,_,_=case();tau=normal_temperature(x,g)
    engine=MatchingObjective(x,x,g,g,tau,beta=0.,chunk=7)
    value,_,terms=engine.evaluate(torch.arange(x.shape[1]),torch.full((x.shape[1],),1/x.shape[1],dtype=torch.float64))
    assert value==pytest.approx(0.,abs=1e-12)
    assert terms['normal_match_score_relative_mse']<1e-12
    active=torch.tensor([0,2,4,6]);p=torch.full((4,),.25,dtype=torch.float64)
    a,ag,_=engine.evaluate(active,p,True)
    other=MatchingObjective(x,x,g,g,tau,beta=0.,chunk=24)
    b,bg,_=other.evaluate(active,p,True)
    assert a==pytest.approx(b,abs=1e-12)
    torch.testing.assert_close(ag,bg,atol=1e-10,rtol=1e-10)
    allowed=g[:,None]!=g[None,:]
    dense=(F.normalize(x,dim=1)@F.normalize(x,dim=1).T).masked_fill(~allowed,-torch.inf)
    z=F.normalize(x[:,active].double()*p.sqrt(),dim=1)
    student=(z@z.T).masked_fill(~allowed,-torch.inf)
    assert (dense.argmax(1)!=student.argmax(1)).any()  # Competitors really change.


def settings():
    return {'max_features':4,'matching_refit_max_iter':8,'matching_exchange_max_steps':1,
            'matching_exchange_candidates':2,'matching_chunk':7,'beta':.002}


def serialized(result):
    return json.loads(json.dumps(result,default=lambda x:x.tolist()))


def test_fit_only_pool_beta0_independence_constraints_and_audit_tampering():
    from scripts.analyze_local_results import audit_matching_solution
    x,val,g,vg,paired=case();cfg=settings()
    result=fit_matching_uvarfs(x,val,g,vg,paired,cfg)
    r=serialized(result);assert audit_matching_solution(r,cfg)
    assert len(r['candidate_pool'])<=18 and not r['support_generation_uses_holdout']
    # Every support/weight/fit loss is unchanged when the validator changes.
    changed=serialized(fit_matching_uvarfs(x,val*torch.arange(1,14),g,vg,paired,cfg))
    assert r['candidate_pool_fit_only_sha256']==changed['candidate_pool_fit_only_sha256']
    c0={**cfg,'beta':0.,'matching_exchange_max_steps':0}
    _,a,_=generate_fit_only_pool(x,g,paired,c0)
    _,b,_=generate_fit_only_pool(x,g,(torch.full_like(x,float('nan')),torch.empty(0)),c0)
    assert a==b
    assert all(c['training_terms']['variability_term']==0 for c in a)
    for key,value,match in [('support_generation_uses_holdout',True,'protocol'),
                           ('candidate_pool_fit_only_sha256','wrong','fingerprint'),
                           ('global_optimality_claimed',True,'certificate')]:
        bad=copy.deepcopy(r);bad[key]=value
        with pytest.raises(ValueError,match=match):audit_matching_solution(bad,cfg)
    bad=copy.deepcopy(r);bad['candidate_id']=(r['candidate_id']+1)%len(r['candidate_pool'])
    with pytest.raises(ValueError,match='winner'):audit_matching_solution(bad,cfg)


@pytest.mark.parametrize('kwargs',[{'temperature':0.},{'temperature':float('nan')},
    {'beta':-1.},{'chunk':0},{'score_weight':0.,'distribution_weight':0.}])
def test_invalid_parameters_fail_explicitly(kwargs):
    x,_,g,_,_=case();params={'temperature':.01,**kwargs}
    with pytest.raises(ValueError):MatchingObjective(x,x,g,g,**params)


def test_invalid_dense_groups_holdout_and_zero_sparse_rows():
    x,val,g,vg,paired=case();broken=x.clone();broken[0]=0
    with pytest.raises(ValueError,match='zero dense'):normal_temperature(broken,g)
    with pytest.raises(ValueError,match='other-image'):normal_temperature(x,torch.zeros_like(g))
    with pytest.raises(ValueError,match='disjoint'):fit_matching_uvarfs(x,val,g,g[:len(val)],paired,settings())
    engine=MatchingObjective(x,x,g,g,.01,beta=0.)
    with pytest.raises(ValueError,match='zero sparse'):engine.evaluate([0],torch.zeros(1))


def matching_fixture(tmp_path,full=False):
    cfg,train,test=local_fixture(tmp_path)
    new=load_config(Path(__file__).resolve().parents[1]/'configs/default.yaml')
    cfg['uvarfs'].update(objective=MATCHING,control_objectives=[LOCAL],
        matching_refit_max_iter=3,matching_exchange_max_steps=0,matching_chunk=7)
    if full:cfg['methods']=new['methods']
    else:cfg['methods']+=['asls_no_score_uvarfs','asls_no_distribution_uvarfs']
    return cfg,train,test


def test_default_approved_protocol_preserves_all_v20_controls_and_budget():
    cfg=load_config(Path(__file__).resolve().parents[1]/'configs/default.yaml')
    assert cfg['uvarfs']['objective']==MATCHING and cfg['uvarfs']['max_features']==256
    assert cfg['asls']['discrete_selection']=='normal_local' and EXPERIMENT_VERSION in cfg['results_dir']
    old=load_config(Path(__file__).resolve().parents[1]/'configs/v20_control.yaml')
    expected=expected_method_names(cfg)
    assert len(expected)==262
    # Old Gram names retain identity; prior Main/input controls become local_*.
    for name in expected_method_names(old):
        renamed=name if name.startswith(('gram_','raw_input_gram_')) else (
            'raw_input_local_'+name.removeprefix('raw_input_') if name.startswith('raw_input_') else 'local_'+name)
        assert renamed in expected


def test_full_fit_inference_memory_pixel_controls_and_runner_worker(tmp_path,monkeypatch):
    import scripts.run_all as runner
    import scripts.analyze_local_results as analyzer
    cfg,train,test=matching_fixture(tmp_path,True)
    cfg['paths']['processed_data']=str(tmp_path);cfg['results_dir']=str(tmp_path/'output')
    extractor=CountingExtractor()
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    monkeypatch.setattr(runner,'load_config',lambda _:copy.deepcopy(cfg))
    monkeypatch.setattr(runner,'DINOv3Extractor',lambda *args:extractor)
    monkeypatch.setattr(runner,'discover_bmad_roots',lambda _: {'liver':tmp_path/'liver'})
    monkeypatch.setattr('uvarfs.pipeline_eval.make_index',lambda memory,cfg:CPUIndex(memory))
    monkeypatch.setattr(sys,'argv',['run_all','--dataset','liver','--dataset-worker'])
    runner.main();root=tmp_path/'output';folder=root/'liver'
    assert not (root/'all_metrics.csv').exists() and not (root/'summary_metrics.csv').exists()
    frame=pd.read_csv(folder/'metrics.csv');assert len(frame)==262
    main=frame.set_index('method').loc['main']
    assert main.uvarfs_objective_definition==MATCHING and pd.isna(main.uvarfs_geometry_feasible)
    assert main.uvarfs_objective_dtype=='float64'
    before=len(extractor.outputs);runner.main();assert len(extractor.outputs)==before
    monkeypatch.setattr(analyzer,'verify_recorded_code',lambda _:False)
    audit=analyzer.audit(root,tmp_path/'audit',True,50)
    assert audit['completed_datasets']==['liver']
    (folder/'uvarfs_asls_no_distribution.json').unlink()
    runner.main();assert len(extractor.outputs)>before  # Incomplete causal result cannot resume.


def test_all_six_worker_merge_refuses_partial_mixed_code_and_seed(tmp_path):
    from scripts.merge_dataset_workers import merge_workers
    from uvarfs.protocol import BMAD_DATASETS,config_fingerprint
    cfg={'seed':42,'methods':['main'],'results_dir':str(tmp_path),
         'uvarfs':{'objective':MATCHING},'representation':{'layer_normalization':'l2'}}
    for name in sorted(BMAD_DATASETS):
        folder=tmp_path/name;folder.mkdir()
        meta={'dataset':name,'experiment_version':EXPERIMENT_VERSION,'config':cfg,
              'config_fingerprint':config_fingerprint(cfg),'code_fingerprint':'source',
              'data_fingerprint':'data','git_head':'revision','backbone':{'num_layers':2}}
        (folder/'run_metadata.json').write_text(json.dumps(meta))
        (folder/'eval_progress.json').write_text(json.dumps({'stage':'complete','methods':1}))
        pd.DataFrame([{'dataset':name,'method':'main','experiment_version':EXPERIMENT_VERSION,
            **{key:.8 for key in ['image_auroc','image_auprc','pixel_auroc','pixel_auprc','aupro']}}]).to_csv(folder/'metrics.csv',index=False)
        (folder/'asls.json').write_text(json.dumps({'selected_layers':[1]}))
        for filename in ['fit_manifest.json','memory_manifest.json','image_predictions.csv',
                         'representation_manifest.json','uvarfs_main.json','normal_geometry_audit.json']:
            (folder/filename).write_text('{}')
    path=tmp_path/'xray'/'eval_progress.json'
    path.write_text(json.dumps({'stage':'running','methods':1}))
    with pytest.raises(ValueError,match='incomplete'):merge_workers(tmp_path,cfg,'source')
    assert not (tmp_path/'all_metrics.csv').exists()
    path.write_text(json.dumps({'stage':'complete','methods':1}))
    with pytest.raises(ValueError,match='inconsistent'):merge_workers(tmp_path,{**cfg,'seed':123},'source')
    with pytest.raises(ValueError,match='inconsistent'):merge_workers(tmp_path,cfg,'different')
    assert not (tmp_path/'all_metrics.csv').exists()
    frame=merge_workers(tmp_path,cfg,'source')
    assert len(frame)==6 and (tmp_path/'workers_complete.json').is_file()
