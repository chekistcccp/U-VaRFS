import copy
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch

from uvarfs.cosine_uvarfs import CosineObjective,refit_simplex,exchange_simplex
from uvarfs.performance_uvarfs import project_bounded_simplex,fit_performance_uvarfs,fit_fixed_budget_reference
from uvarfs.performance_exchange import exchange_after_refit
from uvarfs.pipeline_fit import fit_all_method_specs
from uvarfs.pipeline_eval import build_memories,evaluate
from uvarfs.protocol import expected_method_names
from uvarfs.objectives import PERFORMANCE,experiment_branches
from uvarfs.utils import load_config
from scripts.analyze_results import audit_sparse_path,audit_representation_protocol
from test_v17_performance_first import numerical_cfg,performance_fixture
from test_v10_representation import CountingExtractor,CPUIndex


def normal_case():
    generator=torch.Generator().manual_seed(2)
    x=torch.randn(25,14,generator=generator,dtype=torch.float64)
    v=torch.rand(3,14,generator=generator,dtype=torch.float64)
    cfg=numerical_cfg(tol=1e-7,cosine_refit_max_iter=150,cosine_polish_max_iter=100,
        performance_exchange_strategy='refit_before_accept',performance_exchange_max_steps=1,
        performance_exchange_candidates=8,geometry_tolerance=.8)
    return x,v,cfg


def test_uphill_finite_swap_can_improve_after_same_objective_capped_refit():
    x,v,cfg=normal_case();engine=CosineObjective(x,v);active=torch.arange(8)
    project=lambda p,f:project_bounded_simplex(p,f,4.)
    p,_,_=refit_simplex(engine,active,torch.ones(8,dtype=x.dtype)/8,cfg,'parent',project,4.)
    before,_,_=engine.evaluate(active,p)
    candidate=active.clone();candidate[5]=9;order=candidate.argsort();candidate=candidate[order]
    initial=p[order];trial,_,_=engine.evaluate(candidate,initial)
    fitted,_,_=refit_simplex(engine,candidate,initial,cfg,'counterexample',project,4.)
    after,_,_=engine.evaluate(candidate,fitted)
    assert float(trial)>float(before)+.01 and float(after)<float(before)-.03
    old_active,old_p,_,_,_=exchange_simplex(engine,active,p,{**cfg,'cosine_exchange_max_steps':1},'old',project,4.)
    new_active,new_p,trace=exchange_after_refit(engine,active,p,cfg,'new',project,4.)
    assert trace['attempts'] and any(a['eligible'] and a['before_refit']>=a['before'] for a in trace['attempts'])
    assert float(engine.evaluate(new_active,new_p)[0])<float(engine.evaluate(old_active,old_p)[0])
    assert len(new_active)==8 and new_p.max()<=4/8+1e-12


def test_new_pool_preserves_complete_v17_references_and_audits_all_attempt_costs():
    x,v,cfg=normal_case();gram={'budgeted_weights':np.ones(14),'active':np.arange(8)}
    reference=fit_fixed_budget_reference(x,v,cfg,gram)
    original=copy.deepcopy(reference)
    result=fit_performance_uvarfs(x,v,cfg,gram,reference=reference)
    assert result['objective']<=reference['objective'] and result['feasible']==reference['feasible']
    assert len(result['active'])==len(reference['active'])==8
    for before,after in zip(original['candidate_pool'],result['candidate_pool']):
        np.testing.assert_array_equal(before['active'],after['active'])
        np.testing.assert_array_equal(before['weights'],after['weights'])
        assert before['base_objective']==after['base_objective']
    np.testing.assert_array_equal(reference['weights'],original['weights'])
    audit=audit_sparse_path(result,cfg)
    assert audit['support_refits_attempted']==24 and audit['uphill_initial_proposals_refitted']>0
    assert audit['support_exchanges_accepted']==3 and audit['refitted_exchange_guards_verified']
    attempts=[a for t in result['exchange_search']['trajectories'] for a in t['attempts']]
    assert any(not a['accepted'] and a['refit']['solver_iterations']>0 for a in attempts)
    assert result['solver_iterations']==reference['solver_iterations']+sum(a['refit']['solver_iterations'] for a in attempts)
    for mutate in [
        lambda r:r['exchange_search'].__setitem__('solver_iterations',0),
        lambda r:r['exchange_search']['trajectories'][0].__setitem__('parent_candidate_id',0),
        lambda r:r['exchange_search']['trajectories'][0]['steps'][0].__setitem__('geometry_after',1.),
        lambda r:r['candidate_pool'][-1].__setitem__('parent_candidate_id',0),
        lambda r:r['polish_summary'].__setitem__('iterations',r['polish_summary']['iterations']+1)]:
        bad=copy.deepcopy(result);mutate(bad)
        with pytest.raises(ValueError):audit_sparse_path(bad,cfg)


def test_refitted_exchange_preserves_parent_normal_geometry_feasibility(monkeypatch):
    class Engine:
        x=torch.ones(2,3,dtype=torch.float64)
        evaluations=0
        def evaluate(self,a,p,gradient=False,full_gradient=False):
            self.evaluations+=1
            if list(a)==[0,1]:loss,geometry=.1,.04
            elif p[0]<.4:loss,geometry=.05,.06
            else:loss,geometry=.2,.03
            g=torch.tensor([0.,0.,-1.],dtype=p.dtype) if full_gradient else torch.zeros_like(p)
            return torch.tensor(loss,dtype=p.dtype),g if gradient else None,{'geometry_error':geometry}
    def refit(engine,active,p,cfg,label,projection,cap):
        return p.new_tensor([.25,.75]),{}, {'solver_iterations':7}
    monkeypatch.setattr('uvarfs.performance_exchange.refit_simplex',refit)
    cfg=numerical_cfg(performance_exchange_max_steps=1,performance_exchange_candidates=1,geometry_tolerance=.05)
    active=torch.arange(2);p=torch.ones(2,dtype=torch.float64)/2
    out,q,trace=exchange_after_refit(Engine(),active,p,cfg,'guard',project_bounded_simplex,4.)
    torch.testing.assert_close(out,active);torch.testing.assert_close(q,p)
    assert trace['solver_iterations']==7 and not trace['steps']
    assert not trace['attempts'][0]['eligible'] and trace['attempts'][0]['after_refit']<trace['attempts'][0]['before']


@pytest.mark.parametrize('settings',[{'performance_exchange_max_steps':0},{'performance_exchange_candidates':0},{'performance_exchange_strategy':'unknown'}])
def test_disabled_search_or_invalid_limits(settings):
    x,v,cfg=normal_case();cfg.update(settings)
    gram={'budgeted_weights':np.ones(14),'active':np.arange(8)}
    if settings.get('performance_exchange_max_steps')==0:
        r=fit_performance_uvarfs(x,v,cfg,gram)
        assert len(r['candidate_pool'])==6 and not any(t['attempts'] for t in r['exchange_search']['trajectories'])
        assert audit_sparse_path(r,cfg)['support_refits_attempted']==0
    else:
        with pytest.raises(ValueError,match='performance exchange'):fit_performance_uvarfs(x,v,cfg,gram)


def new_fixture(tmp_path,primary='none'):
    cfg,train,test=performance_fixture(tmp_path,primary)
    cfg['methods'].append('asls_fixed_budget_uvarfs')
    cfg['uvarfs'].update(performance_exchange_strategy='refit_before_accept',performance_exchange_max_steps=1,performance_exchange_candidates=2)
    return cfg,train,test


@pytest.mark.parametrize('primary',['none','l2'])
def test_164_methods_share_fit_memory_and_full_v17_compression_gram_controls(tmp_path,monkeypatch,primary):
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    monkeypatch.setattr('uvarfs.pipeline_eval.make_index',lambda m,c:CPUIndex(m))
    cfg,train,test=new_fixture(tmp_path,primary);extractor=CountingExtractor();out=tmp_path/'fit'
    specs=fit_all_method_specs(extractor,train,cfg,out)
    assert len(specs)==164 and sorted(specs)==expected_method_names(cfg) and len(extractor.outputs)==10
    for branch in experiment_branches(cfg):
        prefix=branch['prefix'];control=specs[prefix+'asls_fixed_budget_uvarfs']['obj'];r=specs[prefix+'main']['obj']
        assert control['objective_definition']==PERFORMANCE and 'exchange_search' not in control
        assert audit_sparse_path(control,cfg['uvarfs'])['selection_and_guards_verified']
        if branch['objective']==PERFORMANCE:
            assert audit_sparse_path(r,cfg['uvarfs'])['refitted_exchange_guards_verified']
            assert len(r['active'])==len(control['active'])==cfg['uvarfs']['max_features']
            if control['feasible']:assert r['feasible'] and r['objective']<=control['objective']
    memories=build_memories(extractor,train,specs,cfg,progress_path=out/'memory_progress.json')
    np.testing.assert_array_equal(memories['asls_fixed_budget_uvarfs'],memories['gram_asls_fixed_budget_uvarfs'])
    frame=pd.DataFrame(evaluate(extractor,test,specs,memories,cfg,progress_path=out/'eval_progress.json'))
    frame['feature_dim']=[memories[n].shape[1] for n in frame.method]
    frame['layer_normalization']=[specs[n]['layer_normalization'] for n in frame.method]
    frame['objective_branch']=[specs[n]['objective_branch'] for n in frame.method]
    assert frame[['image_auroc','image_auprc','pixel_auroc','pixel_auprc','aupro']].notna().all().all()
    assert len(audit_representation_protocol(out,cfg,frame,json.loads((out/'fit_manifest.json').read_text())))==4
    bad=json.loads((out/'uvarfs_asls_fixed_budget.json').read_text());bad['sparsity_strategy']='wrong'
    (out/'uvarfs_asls_fixed_budget.json').write_text(json.dumps(bad))
    with pytest.raises(ValueError,match='v17 fixed-budget control'):audit_representation_protocol(out,cfg,frame,json.loads((out/'fit_manifest.json').read_text()))


def test_default_configuration_and_version_use_fixed_original_budget():
    cfg=load_config(Path(__file__).resolve().parents[1]/'configs/default.yaml')
    assert len(expected_method_names(cfg))==164 and cfg['uvarfs']['max_features']==256
    assert cfg['uvarfs']['performance_weight_cap_factor']==4. and cfg['representation']['layer_normalization']=='none'
    assert cfg['uvarfs']['performance_exchange_strategy']=='refit_before_accept'


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA runtime unavailable locally')
def test_exchange_cpu_cuda_endpoint_and_support_agree():
    x,v,cfg=normal_case();cfg.update(performance_exchange_candidates=2,cosine_polish_max_iter=0)
    gram={'budgeted_weights':np.ones(14),'active':np.arange(8)}
    cpu=fit_performance_uvarfs(x,v,cfg,gram);cuda=fit_performance_uvarfs(x.cuda(),v.cuda(),cfg,gram)
    np.testing.assert_array_equal(cpu['active'],cuda['active'])
    np.testing.assert_allclose(cpu['weights'],cuda['weights'],atol=1e-6,rtol=1e-6)


def test_full_support_needs_no_exchanges_and_preserves_uniform_feasibility():
    g=torch.Generator().manual_seed(811)
    x=torch.randn(15,8,generator=g,dtype=torch.float64);v=torch.zeros(2,8,dtype=torch.float64)
    cfg=numerical_cfg(max_features=8,performance_exchange_strategy='refit_before_accept',performance_exchange_max_steps=1)
    r=fit_performance_uvarfs(x,v,cfg,{'budgeted_weights':np.ones(8),'active':np.arange(8)})
    assert r['feasible'] and r['geometry_error']<1e-10
    assert all(t['termination']=='full_support' and not t['attempts'] for t in r['exchange_search']['trajectories'])
    assert audit_sparse_path(r,cfg)['support_refits_attempted']==0


def test_zero_row_support_proposal_is_rejected_without_refitting(monkeypatch):
    class Engine:
        x=torch.tensor([[1.,0.,0.],[0.,1.,1.]],dtype=torch.float64)
        evaluations=0
        def evaluate(self,a,p,gradient=False,full_gradient=False):
            self.evaluations+=1
            if 0 not in a:raise ValueError('zero row')
            g=p.new_tensor([0.,0.,-1.]) if full_gradient else torch.zeros_like(p)
            return p.new_tensor(.1),g if gradient else None,{'geometry_error':.02}
    def forbidden(*args):raise AssertionError('invalid support was refitted')
    monkeypatch.setattr('uvarfs.performance_exchange.refit_simplex',forbidden)
    cfg=numerical_cfg(performance_exchange_max_steps=1,performance_exchange_candidates=1)
    active=torch.arange(2);p=torch.ones(2,dtype=torch.float64)/2
    out,q,trace=exchange_after_refit(Engine(),active,p,cfg,'zero-row',project_bounded_simplex,4.)
    assert trace['solver_iterations']==0 and not trace['attempts'][0]['initial_valid']
    torch.testing.assert_close(out,active);torch.testing.assert_close(q,p)


def test_runner_resume_requires_complete_v17_control_files(tmp_path,monkeypatch):
    import scripts.run_all as runner
    cfg,_,_=new_fixture(tmp_path)
    cfg['paths']['processed_data']=str(tmp_path);cfg['results_dir']=str(tmp_path/'output')
    cfg['methods']=['main','asls_raw','asls_selected_raw','asls_fixed_budget_uvarfs','asls_compression_uvarfs','asls_pca','asls_random','all_raw']
    extractor=CountingExtractor()
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    monkeypatch.setattr(runner,'load_config',lambda _:cfg)
    monkeypatch.setattr(runner,'DINOv3Extractor',lambda *args:extractor)
    monkeypatch.setattr(runner,'discover_bmad_roots',lambda _: {'liver':tmp_path/'liver'})
    monkeypatch.setattr('uvarfs.pipeline_eval.make_index',lambda m,c:CPUIndex(m))
    monkeypatch.setattr(sys,'argv',['run_all','--dataset','liver']);runner.main()
    frame=pd.read_csv(tmp_path/'output'/'all_metrics.csv').set_index('method')
    assert frame.loc['main'].uvarfs_sparsity_strategy=='fixed_budget_refitted_support_exchange'
    assert frame.loc['asls_fixed_budget_uvarfs'].uvarfs_sparsity_strategy=='fixed_budget_capped_simplex_support_search'
    assert frame.loc['main'].feature_dim==frame.loc['asls_fixed_budget_uvarfs'].feature_dim
    before=len(extractor.outputs);runner.main();assert len(extractor.outputs)==before
    (tmp_path/'output'/'liver'/'uvarfs_asls_fixed_budget.json').unlink()
    runner.main();assert len(extractor.outputs)>before


@pytest.mark.parametrize('tolerance',[0.,.8])
def test_multistep_search_and_infeasible_fallback_keep_original_fixed_budget(tolerance):
    x,v,cfg=normal_case();cfg.update(geometry_tolerance=tolerance,performance_exchange_max_steps=3,performance_exchange_candidates=3)
    r=fit_performance_uvarfs(x,v,cfg,{'budgeted_weights':np.ones(14),'active':np.arange(8)})
    audit=audit_sparse_path(r,cfg)
    assert audit['refitted_exchange_guards_verified'] and len(r['active'])==8
    assert any(len(t['steps'])>1 for t in r['exchange_search']['trajectories'])
    if tolerance==0:
        assert not r['feasible'] and r['selection_reason']=='minimum_geometry_error_fallback'
        assert r['geometry_error']==min(c['geometry_error'] for c in r['candidate_pool'])
    else:
        refs=r['candidate_pool'][:r['exchange_search']['reference_candidate_count']]
        old=min([c for c in refs if c['geometry_error']<=tolerance],key=lambda c:c['base_objective'])
        assert r['feasible'] and r['objective']<=old['base_objective']
