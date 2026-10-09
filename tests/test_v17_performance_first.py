import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.optimize import minimize

from uvarfs.performance_uvarfs import project_bounded_simplex, fit_performance_uvarfs
from uvarfs.cosine_uvarfs import CosineObjective, refit_simplex, project_simplex
from uvarfs.objectives import COSINE, QUADRATIC, PERFORMANCE, experiment_branches
from uvarfs.pipeline_fit import fit_all_method_specs
from uvarfs.pipeline_eval import build_memories, evaluate
from uvarfs.protocol import expected_method_names, EXPERIMENT_VERSION
from uvarfs.utils import load_config
from scripts.analyze_results import audit_sparse_path, audit_representation_protocol, write_cosine_report, DATASETS
from test_v10_representation import CountingExtractor, CPUIndex
from test_v12_cosine_uvarfs import fixture


@pytest.mark.parametrize('size,cap',[(1,4),(2,1),(7,1.01),(13,2),(24,4),(256,4)])
def test_bounded_projection_matches_independent_threshold_bisection(size,cap):
    rng=np.random.default_rng(97+size); values=rng.normal(0,5,size)
    for vector in [values,np.zeros(size),np.arange(size,dtype=float)]:
        lower=1e-4/size;upper=min(1,cap/size)
        left=float(np.min(vector-upper));right=float(np.max(vector-lower))
        for _ in range(120):
            middle=(left+right)/2
            if np.clip(vector-middle,lower,upper).sum()>1: left=middle
            else: right=middle
        expected=np.clip(vector-(left+right)/2,lower,upper)
        actual=project_bounded_simplex(torch.tensor(vector,dtype=torch.float64),cap_factor=cap)
        np.testing.assert_allclose(actual.numpy(),expected,atol=1e-11,rtol=0)
        assert float(actual.sum())==pytest.approx(1,abs=1e-11)
        assert float(1/actual.square().sum())>=size/min(cap,size)-1e-9


def test_projection_has_independent_convex_optimum_and_translation_invariance():
    v=np.array([.9,-.2,1.4,.3,-1.,.2,.8]);size=len(v);cap=2.
    result=project_bounded_simplex(torch.tensor(v,dtype=torch.float64),cap_factor=cap)
    solved=minimize(lambda p:(.5*np.square(p-v).sum(),p-v),np.full(size,1/size),jac=True,
        method='SLSQP',bounds=[(1e-4/size,cap/size)]*size,
        constraints={'type':'eq','fun':lambda p:p.sum()-1.,'jac':lambda p:np.ones(size)},
        options={'ftol':1e-13,'maxiter':500})
    assert solved.success
    np.testing.assert_allclose(result.numpy(),solved.x,atol=1e-10,rtol=0)
    torch.testing.assert_close(project_bounded_simplex(torch.tensor(v+1000,dtype=torch.float64),cap_factor=cap),result,atol=1e-12,rtol=0)


@pytest.mark.parametrize('cap',[0,.99,float('nan'),float('inf')])
def test_invalid_caps_fail(cap):
    with pytest.raises(ValueError,match='capped simplex'):
        project_bounded_simplex(torch.ones(5),cap_factor=cap)


def numerical_cfg(**changes):
    cfg=load_config(Path(__file__).resolve().parents[1]/'configs/default.yaml')['uvarfs']
    cfg.update(max_features=8,min_features=2,max_iter=100,cosine_refit_max_iter=100,
               cosine_polish_max_iter=80,cosine_exchange_max_steps=1,cosine_exchange_candidates=3,
               cosine_geometry_chunk=7,log_every=10000,geometry_tolerance=.2)
    cfg.update(changes);return cfg


def test_cap_prevents_stability_only_weight_collapse_without_reducing_k():
    gen=torch.Generator().manual_seed(901)
    x=torch.randn(31,1,generator=gen,dtype=torch.float64).repeat(1,16)
    variability=torch.ones(2,16,dtype=torch.float64);variability[:,0]=0
    engine=CosineObjective(x,variability)
    active=torch.arange(16);initial=torch.full((16,),1/16,dtype=torch.float64)
    cfg=numerical_cfg(max_features=16,cosine_refit_max_iter=2000)
    unbounded,_,_=refit_simplex(engine,active,initial,cfg,'unbounded')
    cap=4.;project=lambda w,f:project_bounded_simplex(w,f,cap)
    bounded,_,diag=refit_simplex(engine,active,initial,cfg,'bounded',project,cap)
    assert unbounded.max()>.99 and bounded.max()<=cap/16+1e-12
    assert float(1/bounded.square().sum())>=16/cap-1e-10
    assert len(bounded)==16 and diag['polish']['after']<=diag['polish']['before']
    _,gradient,_=engine.evaluate(active,bounded,True)
    residual=float((bounded-project(bounded-gradient,1e-4)).norm())
    assert residual==pytest.approx(diag['fixed_support_first_order_residual'])


@pytest.mark.parametrize('tolerance',[0.,.2])
def test_fixed_budget_selection_and_fallback_are_auditable(tolerance):
    gen=torch.Generator().manual_seed(807)
    x=torch.randn(29,15,generator=gen,dtype=torch.float64)
    v=torch.rand(3,15,generator=gen,dtype=torch.float64)
    gram={'budgeted_weights':np.linspace(.01,1,15),'active':np.array([2,5,8,9])}
    cfg=numerical_cfg(geometry_tolerance=tolerance)
    r=fit_performance_uvarfs(x,v,cfg,gram)
    assert len(r['active'])==cfg['max_features'] and r['lambda']==0 and len(r['lambda_path'])==1
    assert r['weight_cap_factor']==4 and r['maximum_relative_weight']<=4/8+1e-7
    audit=audit_sparse_path(r,cfg)
    assert audit['selection_and_guards_verified'] and audit['feasibility_scope']=='fixed_budget_generated_pool'
    assert audit['lambda_path_feasible_points'] is None
    if tolerance==0: assert not r['feasible']
    eligible=[c for c in r['candidate_pool'] if c['geometry_error']<=tolerance]
    if eligible:
        assert r['objective']==min(c['base_objective'] for c in eligible)
    else: assert r['geometry_error']==min(c['geometry_error'] for c in r['candidate_pool'])
    for key,value in [('weight_cap_factor',99),('lambda',.001),('solver_converged',not r['solver_converged'])]:
        bad=copy.deepcopy(r);bad[key]=value
        with pytest.raises(ValueError): audit_sparse_path(bad,cfg)
    bad=copy.deepcopy(r);bad['candidate_pool'][0]['weights'][bad['candidate_pool'][0]['active'][0]]=1.
    with pytest.raises(ValueError,match='capped weights'): audit_sparse_path(bad,cfg)


def test_feasible_uniform_reference_is_kept_when_refinement_sacrifices_geometry():
    gen=torch.Generator().manual_seed(713)
    x=torch.randn(29,8,generator=gen,dtype=torch.float64)
    v=torch.ones(2,8,dtype=torch.float64);v[:,0]=0
    cfg=numerical_cfg(max_features=8,beta=1.,geometry_tolerance=1e-6)
    r=fit_performance_uvarfs(x,v,cfg,{'budgeted_weights':np.ones(8),'active':np.array([0,1,2])})
    assert r['feasible'] and r['candidate_type']=='uniform_reference'
    assert min(c['base_objective'] for c in r['candidate_pool'])<r['base_objective']
    assert audit_sparse_path(r,cfg)['selection_and_guards_verified']


def test_zero_row_candidates_fail_without_raw_detector_fallback():
    x=torch.eye(7,dtype=torch.float64);v=torch.zeros(2,7,dtype=torch.float64)
    with pytest.raises(RuntimeError,match='No valid fixed-budget support'):
        fit_performance_uvarfs(x,v,numerical_cfg(max_features=2),{'budgeted_weights':np.ones(7),'active':np.array([0,1])})


def performance_fixture(tmp_path,primary='none'):
    cfg,train,test=fixture(tmp_path,primary)
    cfg['methods'].extend(['asls_selected_raw','asls_compression_uvarfs'])
    cfg['uvarfs'].update(objective=PERFORMANCE,performance_weight_cap_factor=2.,geometry_tolerance=.8)
    return cfg,train,test


@pytest.mark.parametrize('primary',['none','l2'])
def test_160_method_fit_memory_evaluation_preserves_normal_protocol_and_controls(tmp_path,monkeypatch,primary):
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    monkeypatch.setattr('uvarfs.pipeline_eval.make_index',lambda memory,cfg:CPUIndex(memory))
    cfg,train,test=performance_fixture(tmp_path,primary);extractor=CountingExtractor();out=tmp_path/'fit'
    specs=fit_all_method_specs(extractor,train,cfg,out)
    assert len(specs)==160 and sorted(specs)==expected_method_names(cfg) and len(extractor.outputs)==10
    for branch in experiment_branches(cfg):
        prefix=branch['prefix'];own=specs[prefix+'main'];r=own['obj']
        control=specs[prefix+'asls_compression_uvarfs']['obj']
        assert control['objective_definition']==COSINE and audit_sparse_path(control,cfg['uvarfs'])['selection_and_guards_verified']
        if branch['objective']==PERFORMANCE:
            assert len(r['active'])==cfg['uvarfs']['max_features']
            assert r['effective_weight_dimension']>=len(r['active'])/2-1e-5
        assert audit_sparse_path(r,cfg['uvarfs'])['selection_and_guards_verified']
        assert specs[prefix+'asls_pca']['obj']['components'].shape[0]==len(r['active'])
        for seed in cfg['random_baseline_seeds']:
            assert len(specs[prefix+f'asls_random_seed{seed}']['obj'])==len(r['active'])
    memories=build_memories(extractor,train,specs,cfg,progress_path=out/'memory_progress.json')
    rows=pd.DataFrame(evaluate(extractor,test,specs,memories,cfg,progress_path=out/'eval_progress.json'))
    rows['layer_normalization']=[specs[n]['layer_normalization'] for n in rows.method]
    rows['objective_branch']=[specs[n]['objective_branch'] for n in rows.method]
    rows['feature_dim']=[memories[n].shape[1] for n in rows.method]
    assert rows[['image_auroc','image_auprc','pixel_auroc','pixel_auprc','aupro']].notna().all().all()
    records=audit_representation_protocol(out,cfg,rows,json.loads((out/'fit_manifest.json').read_text()))
    assert len(records)==4
    np.testing.assert_array_equal(memories['asls_compression_uvarfs'],memories['gram_asls_compression_uvarfs'])
    assert set(map(len,memories.values()))=={5}


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA runtime unavailable locally')
def test_bounded_projection_cpu_cuda_agree():
    values=torch.randn(256,dtype=torch.float64,generator=torch.Generator().manual_seed(715))
    torch.testing.assert_close(project_bounded_simplex(values.cuda()).cpu(),project_bounded_simplex(values),atol=1e-11,rtol=0)


def test_runner_reports_fixed_budget_identity_and_requires_compression_control(tmp_path,monkeypatch):
    import scripts.run_all as runner
    cfg,_,_=performance_fixture(tmp_path)
    cfg['paths']['processed_data']=str(tmp_path);cfg['results_dir']=str(tmp_path/'output')
    cfg['methods']=['main','asls_raw','asls_selected_raw','asls_compression_uvarfs','asls_pca','asls_random','all_raw']
    extractor=CountingExtractor()
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    monkeypatch.setattr(runner,'load_config',lambda _:cfg)
    monkeypatch.setattr(runner,'DINOv3Extractor',lambda *args:extractor)
    monkeypatch.setattr(runner,'discover_bmad_roots',lambda _: {'liver':tmp_path/'liver'})
    monkeypatch.setattr('uvarfs.pipeline_eval.make_index',lambda memory,cfg:CPUIndex(memory))
    monkeypatch.setattr(sys,'argv',['run_all','--dataset','liver']);runner.main()
    frame=pd.read_csv(tmp_path/'output'/'all_metrics.csv');rows=frame.set_index('method')
    assert frame.experiment_version.eq(EXPERIMENT_VERSION).all()
    assert rows.loc['main'].uvarfs_objective_definition==PERFORMANCE
    assert rows.loc['main'].uvarfs_selected_lambda==0
    assert rows.loc['main'].feature_dim==cfg['uvarfs']['max_features']
    assert rows.loc['main'].uvarfs_feasibility_scope=='fixed_budget_generated_pool'
    assert pd.isna(rows.loc['main'].uvarfs_lambda_path_feasible_points)
    assert rows.loc['asls_compression_uvarfs'].uvarfs_objective_definition==COSINE
    folder=tmp_path/'output'/'liver'
    records=audit_representation_protocol(folder,cfg,frame,json.loads((folder/'fit_manifest.json').read_text()))
    all_rows=pd.concat([frame.assign(dataset=name) for name in DATASETS],ignore_index=True)
    all_rows['method_family']=all_rows.method.str.replace(r'_seed\d+$','',regex=True)
    families=[]
    for (dataset,family),group in all_rows.groupby(['dataset','method_family']):
        families.append({'dataset':dataset,'method_family':family,'runs':len(group),
                         'image_auroc_mean':group.image_auroc.mean(),'image_auroc_std':group.image_auroc.std(ddof=1) if len(group)>1 else 0})
    report=tmp_path/'report';report.mkdir()
    paired=pd.DataFrame({'dataset':DATASETS,'target_method':['main']*6,'comparison':['asls_compression_uvarfs']*6,
                        'delta_auroc':[0.]*6,'ci95_lower':[0.]*6,'ci95_upper':[0.]*6})
    data={'experiment_version':EXPERIMENT_VERSION,'git_revision':'constructed-fixture','bootstrap_draws':100,
          'git_blob_code_verified':False,'diagnostics':[{'dataset':name,'representation_audit':records} for name in DATASETS]}
    write_cosine_report(report,all_rows,pd.DataFrame(families),paired,pd.DataFrame(),data)
    text=(report/'analysis.md').read_text(encoding='utf-8')
    assert 'cosine_simplex_fixed_budget' in text and 'asls_compression_uvarfs' in text
    scope=pd.read_csv(report/'cosine_feasibility_scope.csv')
    assert scope[scope.method.isin(['main','layer_l2_main'])].lambda_path_feasible_points.isna().all()
    control_path=folder/'uvarfs_asls_compression.json'
    original=json.loads(control_path.read_text())
    bad={**original,'objective_definition':PERFORMANCE};control_path.write_text(json.dumps(bad))
    with pytest.raises(ValueError,match='compression control differs'):
        audit_representation_protocol(folder,cfg,frame,json.loads((folder/'fit_manifest.json').read_text()))
    control_path.write_text(json.dumps(original))
    before=len(extractor.outputs);runner.main();assert len(extractor.outputs)==before
    (folder/'uvarfs_asls_compression.json').unlink();runner.main();assert len(extractor.outputs)>before
