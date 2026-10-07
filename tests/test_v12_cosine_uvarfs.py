import copy
import json
import sys

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F
from scipy.optimize import minimize

from uvarfs.cosine_uvarfs import CosineObjective, project_simplex, refit_simplex, fit_cosine_uvarfs
from uvarfs.objectives import COSINE, QUADRATIC, experiment_branches, objective_modes
from uvarfs.protocol import expected_method_names, EXPERIMENT_VERSION
from uvarfs.u_varfs import fit_uvarfs
from uvarfs.pipeline_fit import fit_all_method_specs
from uvarfs.pipeline_eval import build_memories, evaluate, transform
from uvarfs.representation import normalize_layer
from scripts.analyze_results import audit_sparse_path, audit_representation_protocol, write_cosine_report, DATASETS
from scripts.review_uvarfs_objective import proposed_loss_for_review
from test_v10_representation import make_fixture, CountingExtractor, CPUIndex


def test_simplex_projection_matches_independent_scalar_threshold():
    v=torch.tensor([-.7, .2, 2., -.1],dtype=torch.float64)
    floor=.03; low=-5.; high=5.
    for _ in range(100):
        middle=(low+high)/2
        total=sum(max(float(a)-middle,floor/len(v)) for a in v)
        if total>1: low=middle
        else: high=middle
    expected=[max(float(a)-(low+high)/2,floor/len(v)) for a in v]
    torch.testing.assert_close(project_simplex(v,floor),torch.tensor(expected,dtype=v.dtype),atol=1e-12,rtol=0)
    assert float(project_simplex(v,floor).sum())==pytest.approx(1)
    assert project_simplex(torch.tensor([7.]),floor).item()==1


@pytest.mark.parametrize('chunk',[1,5,40])
def test_analytic_loss_and_full_gradient_match_autograd_and_finite_differences(chunk):
    gen=torch.Generator().manual_seed(115)
    x=torch.randn(11,7,generator=gen,dtype=torch.float64)
    variability=torch.rand(3,7,generator=gen,dtype=torch.float64)
    active=torch.tensor([0,2,5]); p=torch.tensor([.2,.35,.45],dtype=torch.float64)
    engine=CosineObjective(x,variability,.007,chunk)
    value,gradient,terms=engine.evaluate(active,p,True,True)
    full=torch.zeros(7,dtype=torch.float64); full[active]=p; full.requires_grad_(True)
    # Sample-space oracle avoids differentiating sqrt at zero weights.
    X=F.normalize(x,dim=1); K=X@X.T
    h=(X.square()*full).sum(1)
    C=(X*full)@X.T/torch.sqrt(h[:,None]*h[None,:])
    P=engine.P
    oracle=.5*(K-C).square().sum()/K.square().sum()+.007*(P.T@full).square().sum()/(P.mean(0).square().sum())
    oracle.backward()
    assert float(value)==pytest.approx(float(oracle.detach()),rel=1e-12,abs=1e-13)
    torch.testing.assert_close(gradient,full.grad,atol=1e-11,rtol=1e-10)
    for j in range(len(p)):
        plus=p.clone(); minus=p.clone(); plus[j]+=1e-6; minus[j]-=1e-6
        numerical=float((engine.evaluate(active,plus)[0]-engine.evaluate(active,minus)[0])/2e-6)
        assert gradient[active[j]].item()==pytest.approx(numerical,abs=1e-9,rel=1e-7)
    proposal=proposed_loss_for_review(x.numpy(),full.detach().numpy(),engine.P.numpy(),.007,0)
    assert proposal['objective']==pytest.approx(float(value),abs=1e-12)


def test_fixed_support_descent_matches_independent_scipy_simplex_solve():
    gen=torch.Generator().manual_seed(134)
    x=torch.randn(12,4,generator=gen,dtype=torch.float64)
    v=torch.rand(2,4,generator=gen,dtype=torch.float64)
    engine=CosineObjective(x,v,.002,5); active=torch.tensor([0,1,3]); seed=torch.tensor([.1,.7,.2],dtype=x.dtype)
    cfg={'max_iter':800,'tol':1e-9,'cosine_weight_floor_mass':1e-4,'log_every':1000}
    p,terms,diag=refit_simplex(engine,active,seed,cfg,'oracle')
    def oracle(weights):
        full=np.zeros(4); full[active.numpy()]=weights
        full/=full.sum()  # SLSQP finite differences temporarily leave its equality constraint.
        return proposed_loss_for_review(x.numpy(),full,engine.P.numpy(),.002,0)['objective']
    reference=minimize(oracle,seed.numpy(),method='SLSQP',bounds=[(1e-4/3,1)]*3,
        constraints={'type':'eq','fun':lambda weights:weights.sum()-1},options={'ftol':1e-12,'maxiter':800})
    assert reference.success
    assert terms['representation_term']+terms['variability_term']==pytest.approx(reference.fun,abs=2e-7)
    assert float(p.sum())==pytest.approx(1,abs=1e-12)
    assert all(item['after']<item['before'] for item in diag['descent_history'])


def config(**overrides):
    return {'beta':.002,'lambda_grid':[.05,.001,.00001],'min_features':2,'max_features':4,
        'max_iter':100,'cosine_refit_max_iter':80,'cosine_support_step':1,'cosine_geometry_chunk':5,
        'cosine_exchange_max_steps':2,'cosine_exchange_candidates':4,
        'geometry_tolerance':.05,'tol':1e-6,'log_every':1000,**overrides}


def test_duplicate_columns_are_cosine_feasible_and_keep_original_control_unchanged():
    base=torch.tensor([[1.,.5],[.5,1.],[1.,-1.],[-1.,.3]],dtype=torch.float64)
    x=base.repeat(1,8); v=torch.ones(2,16,dtype=x.dtype)
    old={'active':np.array([0,1]),'budgeted_weights':np.array([1.,1.]+[0.]*14)}
    before=copy.deepcopy(old); cfg=config(beta=0.)
    result=fit_cosine_uvarfs(x,v,cfg,old)
    assert result['feasible'] and len(result['active'])==2 and result['geometry_error']<1e-10
    assert result['objective_definition']==COSINE and result['global_optimality_claimed'] is False
    assert 'box_optimality_gap' not in result and 'budget_geometry_audit' not in result
    for key in old: np.testing.assert_array_equal(old[key],before[key])
    assert audit_sparse_path(result,cfg)['selection_and_guards_verified']
    broken=copy.deepcopy(result); broken['lambda_path'][0]['sparsity_term']+=.1
    with pytest.raises(ValueError,match='objective'):
        audit_sparse_path(broken,cfg)
    broken=copy.deepcopy(result); broken['candidate_pool'][0]['weights'][0]*=.5
    with pytest.raises(ValueError,match='simplex'):
        audit_sparse_path(broken,cfg)


def test_generated_pool_fallback_is_explicit_and_zero_row_candidates_cannot_be_detector_fallbacks():
    x=torch.eye(6,dtype=torch.float64); v=torch.zeros(2,6,dtype=x.dtype)
    old={'active':np.array([0,1]),'budgeted_weights':np.array([1.,1.,0.,0.,0.,0.])}
    with pytest.raises(RuntimeError,match='No valid cosine'):
        fit_cosine_uvarfs(x,v,config(max_features=2),old)
    gen=torch.Generator().manual_seed(61)
    x=torch.randn(17,8,generator=gen,dtype=torch.float64)
    result=fit_cosine_uvarfs(x,torch.zeros(2,8,dtype=x.dtype),config(max_features=2,geometry_tolerance=0),
        {'active':np.array([0,1]),'budgeted_weights':np.array([1.,1.]+[0.]*6)})
    assert result['feasible'] is False and result['selection_reason']=='minimum_geometry_error_fallback'
    assert audit_sparse_path(result,config(max_features=2,geometry_tolerance=0))['selection_and_guards_verified']


@pytest.mark.parametrize('setting',[{'objective':'bad'},{'objective':COSINE,'ablation_objective':COSINE}])
def test_objective_protocol_rejects_invalid_configuration(setting):
    with pytest.raises(ValueError,match='objective'):
        objective_modes({'uvarfs':setting})


def fixture(tmp_path,primary='none'):
    cfg,train,test=make_fixture(tmp_path,primary)
    cfg['methods'].append('asls_exchange_refit_uvarfs')
    cfg['uvarfs'].update(objective=COSINE,ablation_objective=QUADRATIC,sparsity_strategy='objective_forward_refit',
        cosine_refit_max_iter=15,cosine_support_step=1,cosine_exchange_max_steps=1,
        cosine_exchange_candidates=3,cosine_geometry_chunk=5)
    return cfg,train,test


@pytest.mark.parametrize('primary',['none','l2'])
def test_paired_152_methods_use_one_normal_stream_and_matching_dimensions(tmp_path,monkeypatch,primary):
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    monkeypatch.setattr('uvarfs.pipeline_eval.make_index',lambda memory,cfg:CPUIndex(memory))
    cfg,train,test=fixture(tmp_path,primary); extractor=CountingExtractor(); out=tmp_path/'fit'
    specs=fit_all_method_specs(extractor,train,cfg,out)
    assert len(specs)==152 and sorted(specs)==expected_method_names(cfg)
    assert len(extractor.outputs)==10
    primary_gram=specs['gram_main']['obj']
    # Original solver is an exact same-data control, not a new loss relabeled.
    layers=specs['gram_main']['layers']
    normal=torch.cat([torch.cat([normalize_layer(extractor.outputs[o][l],primary).reshape(-1,4) for o in [0,5]]) for l in layers],1)
    manifest=json.loads((out/'fit_manifest.json').read_text())
    variations=torch.cat([torch.tensor(manifest['variability_vectors'][str(l)]) for l in layers],1)
    independent=fit_uvarfs(normal,variations,cfg['uvarfs'])
    for key in ['active','scales','budgeted_weights']:
        np.testing.assert_array_equal(primary_gram[key],independent[key])
    for branch in experiment_branches(cfg):
        folder=out/branch['folder']; prefix=branch['prefix']
        u=json.loads((folder/'uvarfs_main.json').read_text())
        assert u['objective_definition']==branch['objective']
        assert audit_sparse_path(u,cfg['uvarfs'])['selection_and_guards_verified']
        own=specs[prefix+'main']; dimension=len(own['obj']['active'])
        assert specs[prefix+'asls_pca']['obj']['components'].shape[0]==dimension
        for seed in cfg['random_baseline_seeds']:
            assert len(specs[prefix+f'asls_random_seed{seed}']['obj'])==dimension
        if branch['objective']==COSINE:
            assert sum(u['relative_weight_by_layer'].values())==pytest.approx(1,abs=1e-6)
    assert specs['asls_top_weights_uvarfs']['obj']['budgeted_weights'].tolist()==primary_gram['legacy_top_weights']['budgeted_weights'].tolist()
    memories=build_memories(extractor,train,specs,cfg,progress_path=out/'memory_progress.json')
    assert set(map(len,memories.values()))=={5}
    np.testing.assert_array_equal(memories['all_raw'],memories['gram_all_raw'])
    frame=pd.DataFrame(evaluate(extractor,test,specs,memories,cfg,progress_path=out/'eval_progress.json'))
    frame['layer_normalization']=[specs[name]['layer_normalization'] for name in frame.method]
    frame['feature_dim']=[memories[name].shape[1] for name in frame.method]
    assert frame[['image_auroc','image_auprc','pixel_auroc','pixel_auprc','aupro']].notna().all().all()
    records=audit_representation_protocol(out,cfg,frame,json.loads((out/'fit_manifest.json').read_text()))
    assert len(records)==4
    for record in records:
        if record['objective_definition']==COSINE:
            normal=record['normal_cosine_geometry']['methods']['main']
            assert normal['unrenormalized_weighted_gram_error'] is None


def test_runner_outputs_objective_scoped_metrics_and_resumes_only_complete_branches(tmp_path,monkeypatch):
    import scripts.run_all as runner
    cfg,train,test=fixture(tmp_path)
    cfg['paths']['processed_data']=str(tmp_path)
    cfg['results_dir']=str(tmp_path/'output')
    cfg['methods']=['main','asls_raw','asls_pca','asls_random','all_raw']
    extractor=CountingExtractor()
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    monkeypatch.setattr(runner,'load_config',lambda _:cfg)
    monkeypatch.setattr(runner,'DINOv3Extractor',lambda *args:extractor)
    monkeypatch.setattr(runner,'discover_bmad_roots',lambda _: {'liver':tmp_path/'liver'})
    monkeypatch.setattr('uvarfs.pipeline_eval.make_index',lambda memory,cfg:CPUIndex(memory))
    monkeypatch.setattr(sys,'argv',['run_all','--dataset','liver'])
    runner.main()
    frame=pd.read_csv(tmp_path/'output'/'all_metrics.csv')
    selected=frame[frame.method.isin(['main','gram_main'])].set_index('method')
    assert selected.loc['main'].uvarfs_objective_definition==COSINE
    assert pd.isna(selected.loc['main'].uvarfs_box_optimality_gap)
    assert selected.loc['gram_main'].uvarfs_objective_definition==QUADRATIC
    assert np.isfinite(selected.loc['gram_main'].uvarfs_box_optimality_gap)
    assert frame.experiment_version.eq(EXPERIMENT_VERSION).all()
    # Exercise the actual new report schema with clearly constructed six-dataset
    # copies, never presented as a BMAD performance experiment.
    out=tmp_path/'output'/'liver'
    records=audit_representation_protocol(out,cfg,frame,json.loads((out/'fit_manifest.json').read_text()))
    all_rows=pd.concat([frame.assign(dataset=name) for name in DATASETS],ignore_index=True)
    all_rows['method_family']=all_rows.method.str.replace(r'_seed\d+$','',regex=True)
    families=[]
    for (dataset,family),group in all_rows.groupby(['dataset','method_family']):
        families.append({'dataset':dataset,'method_family':family,'runs':len(group),
                         'image_auroc_mean':group.image_auroc.mean(),'image_auroc_std':group.image_auroc.std(ddof=1) if len(group)>1 else 0})
    report=tmp_path/'report'; report.mkdir()
    paired=pd.DataFrame({'dataset':DATASETS,'target_method':['main']*6,'comparison':['gram_main']*6,
                        'delta_auroc':[0.]*6,'ci95_lower':[0.]*6,'ci95_upper':[0.]*6})
    data={'experiment_version':EXPERIMENT_VERSION,'git_revision':'constructed-fixture',
          'bootstrap_draws':100,'git_blob_code_verified':False,
          'diagnostics':[{'dataset':name,'representation_audit':records} for name in DATASETS]}
    write_cosine_report(report,all_rows,pd.DataFrame(families),paired,pd.DataFrame(),data)
    assert (report/'analysis.md').is_file() and (report/'normal_cosine_geometry.csv').is_file()
    saved=pd.read_csv(report/'normal_cosine_geometry.csv')
    assert saved[saved.objective==COSINE].continuous_box_gap.isna().all()
    assert saved[saved.objective==QUADRATIC].continuous_box_gap.notna().all()
    before=len(extractor.outputs); runner.main()
    assert len(extractor.outputs)==before
    (tmp_path/'output'/'liver'/'gram'/'uvarfs_main.json').unlink()
    runner.main()
    assert len(extractor.outputs)>before


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA runtime unavailable locally')
def test_cosine_sample_space_gradient_cpu_cuda_agree():
    gen=torch.Generator().manual_seed(87)
    x=torch.randn(19,11,generator=gen,dtype=torch.float64); v=torch.rand(3,11,generator=gen,dtype=torch.float64)
    active=torch.tensor([1,4,7,8]); p=torch.tensor([.1,.3,.2,.4],dtype=x.dtype)
    cpu=CosineObjective(x,v).evaluate(active,p,True,True)
    gpu=CosineObjective(x.cuda(),v.cuda()).evaluate(active.cuda(),p.cuda(),True,True)
    torch.testing.assert_close(cpu[0],gpu[0].cpu(),atol=1e-11,rtol=1e-10)
    torch.testing.assert_close(cpu[1],gpu[1].cpu(),atol=1e-10,rtol=1e-9)
