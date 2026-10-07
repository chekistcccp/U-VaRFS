import copy
import itertools
import json

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F
from scipy.optimize import minimize_scalar

from uvarfs.budget_geometry import audit_budget_geometry
from uvarfs.sparse_support import objective_forward, forward_budget
from uvarfs.u_varfs import fit_uvarfs
from uvarfs.pipeline_fit import fit_all_method_specs
from uvarfs.pipeline_eval import build_memories, evaluate
from uvarfs.protocol import expected_method_names
from scripts.analyze_results import audit_sparse_path, audit_representation_protocol, paired_bootstrap
from test_v10_representation import make_fixture, CountingExtractor, CPUIndex


def test_forward_insertions_match_independent_sample_space_scalar_optimization():
    gen=torch.Generator().manual_seed(71)
    x=F.normalize(torch.randn(14,9,generator=gen,dtype=torch.float64),dim=1)
    h=(x.T@x).square(); h/=h.sum()
    p=torch.rand(9,2,generator=gen,dtype=torch.float64)
    beta=.017; scale=.3; lambdas=[.001,.1]
    actual,counts=objective_forward(h,p,beta,scale,lambdas,{'max_features':3},'oracle')
    X=x.numpy(); P=p.numpy(); K=X@X.T
    expected=np.zeros((9,2))
    for column,lam in enumerate(lambdas):
        def loss(w):
            difference=K-(X*w)@X.T
            return .5*np.square(difference).sum()/np.square(K).sum()+beta*scale*np.square(P.T@w).sum()+lam*w.sum()
        for _ in range(3):
            trials=[]
            for index in np.flatnonzero(expected[:,column]==0):
                def candidate(value):
                    w=expected[:,column].copy(); w[index]=value
                    return loss(w)
                optimum=minimize_scalar(candidate,bounds=(0,1),method='bounded',options={'xatol':1e-13})
                value=min([0.,1.,optimum.x],key=candidate)
                trials.append((candidate(value),index,value))
            objective,index,value=min(trials)
            if objective>=loss(expected[:,column]) or value<=1e-4:
                break
            expected[index,column]=value
    np.testing.assert_allclose(actual.numpy(),expected,atol=2e-7,rtol=2e-7)
    assert counts==np.count_nonzero(expected,axis=0).tolist()


def test_forward_candidate_can_replace_a_bad_support_without_changing_loss():
    h=torch.diag(torch.tensor([.9,.05,.05],dtype=torch.float64))
    p=torch.zeros(3,0,dtype=torch.float64)
    seed=torch.tensor([[0.],[1.],[0.]],dtype=torch.float64)
    cfg={'min_features':1,'max_features':1,'max_iter':200,'tol':1e-10}
    result,diagnostic=forward_budget(seed,torch.ones_like(seed),h,p,0,0,[.001],cfg,'test')
    assert torch.where(result[:,0]>0)[0].tolist()==[0]
    assert diagnostic[0]['forward_candidate_accepted']
    assert diagnostic[0]['forward_objective_improvement']>0
    assert diagnostic[0]['forward_geometry_improvement']>0


def test_global_budget_bound_is_tight_for_redundant_equal_columns_despite_cosine_preservation():
    x=torch.ones(5,8,dtype=torch.float64)/np.sqrt(8)
    h=(x.T@x).square(); h/=h.sum()
    audit=audit_budget_geometry(h,{'max_features':2,'geometry_tolerance':.05})
    assert audit['geometry_error_lower_bound']==pytest.approx(.75)
    assert audit['necessary_features_lower_bound']==8
    assert audit['budget_ruled_out_at_working_precision']
    selected=x[:,:2]
    assert float(((x@x.T)-(selected@selected.T)).norm()/(x@x.T).norm())==pytest.approx(.75)
    torch.testing.assert_close(F.normalize(x,dim=1)@F.normalize(x,dim=1).T,
                               F.normalize(selected,dim=1)@F.normalize(selected,dim=1).T)
    assert audit['selection_candidate'] is False


def test_global_bound_covers_all_small_supports_and_fractional_box_weights():
    gen=torch.Generator().manual_seed(82)
    x=torch.randn(7,6,generator=gen,dtype=torch.float64)
    h=(x.T@x).square(); h/=h.sum()
    audit=audit_budget_geometry(h,{'max_features':2,'geometry_tolerance':.1})
    reference=x@x.T
    for count in range(3):
        for support in itertools.combinations(range(6),count):
            for level in [0.,.3,1.]:
                weights=torch.zeros(6,dtype=torch.float64); weights[list(support)]=level
                actual=float((reference-(x*weights)@x.T).norm()/reference.norm())
                assert actual>=audit['geometry_error_lower_bound']-1e-12
    full=audit_budget_geometry(h,{'max_features':6,'geometry_tolerance':.1})
    assert full['geometry_error_lower_bound']==pytest.approx(0,abs=1e-12)
    zero=audit_budget_geometry(torch.zeros(3,3),{'max_features':2})
    assert zero['zero_reference_gram'] and zero['geometry_error_lower_bound']==0


@pytest.mark.parametrize('beta',[0.,.002,.2])
def test_forward_path_protects_exact_exchange_control_and_keeps_label_free_selection(beta):
    gen=torch.Generator().manual_seed(12)
    x=torch.randn(32,11,generator=gen); variability=torch.rand(3,11,generator=gen)
    cfg={'beta':beta,'lambda_grid':[.01,.001,.0001],'min_features':2,'max_features':4,
         'max_iter':100,'support_refit_max_iter':150,'support_exchange_max_steps':2,
         'geometry_tolerance':.03,'tol':1e-6,'diagnose_infeasible':True}
    result=fit_uvarfs(x,variability,cfg)
    old=fit_uvarfs(x,variability,{**cfg,'sparsity_strategy':'objective_exchange_refit'})
    assert result['sparsity_strategy']=='objective_forward_refit'
    assert result['exchange_refit']['lambda_path']==old['lambda_path']
    for field in ['active','scales','budgeted_weights']:
        np.testing.assert_array_equal(result['exchange_refit'][field],old[field])
    for point,previous in zip(result['lambda_path'],old['lambda_path']):
        assert point['budgeted_objective']<=previous['budgeted_objective']+1e-7
        assert point['geometry_error']<=previous['geometry_error']+1e-7
    assert audit_sparse_path(result,cfg)['selection_and_guards_verified']
    assert result['zero_lambda_diagnostic']['selection_candidate'] is False
    broken=copy.deepcopy(result); broken['budget_geometry_audit']['geometry_error_lower_bound']+=.1
    with pytest.raises(ValueError,match='lower bound'):
        audit_sparse_path(broken,cfg)


@pytest.mark.parametrize('primary',['none','l2'])
def test_76_method_paired_fit_and_matching_preserve_old_controls_and_new_audit(tmp_path,monkeypatch,primary):
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    monkeypatch.setattr('uvarfs.pipeline_eval.make_index',lambda memory,cfg:CPUIndex(memory))
    cfg,train,test=make_fixture(tmp_path,primary)
    cfg['uvarfs']['sparsity_strategy']='objective_forward_refit'
    cfg['methods'].append('asls_exchange_refit_uvarfs')
    out=tmp_path/'result'; extractor=CountingExtractor()
    specs=fit_all_method_specs(extractor,train,cfg,out)
    assert len(specs)==76 and sorted(specs)==expected_method_names(cfg)
    assert len(extractor.outputs)==10
    prefix='layer_l2_' if primary=='none' else 'raw_input_'
    for name in ['','layer_l2/' if primary=='none' else 'raw_input/']:
        u=json.loads((out/name/'uvarfs_main.json').read_text())
        saved=json.loads((out/name/'uvarfs_asls_exchange_refit.json').read_text())
        assert saved['sparsity_strategy']=='objective_exchange_refit'
        assert u['exchange_refit']['budgeted_weights']==saved['budgeted_weights']
        assert sum(u['normal_reference_alignment_by_layer'].values())==pytest.approx(1)
    assert specs[prefix+'asls_exchange_refit_uvarfs']['layer_normalization']!=primary
    memories=build_memories(extractor,train,specs,cfg,progress_path=out/'memory_progress.json')
    assert set(map(len,memories.values()))=={5}
    frame=pd.DataFrame(evaluate(extractor,test,specs,memories,cfg,progress_path=out/'eval_progress.json'))
    frame['layer_normalization']=[specs[m]['layer_normalization'] for m in frame.method]
    assert frame[['image_auroc','image_auprc','pixel_auroc','pixel_auprc','aupro']].notna().all().all()
    assert len(audit_representation_protocol(out,cfg,frame,json.loads((out/'fit_manifest.json').read_text())))==2


def test_branch_bootstrap_uses_declared_target_instead_of_primary_main():
    frame=pd.DataFrame({'label':[0,0,1,1],'main':[0.,1.,0.,1.],
                        'layer_l2_main':[0.,0.,1.,1.],'same':[0.,0.,1.,1.]})
    actual=paired_bootstrap(frame,{'same':['same']},draws=100,target='layer_l2_main')
    assert actual['same']==[0.,0.]


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA runtime unavailable locally')
def test_forward_support_cpu_cuda_agree():
    gen=torch.Generator().manual_seed(72)
    x=torch.randn(20,12,generator=gen,dtype=torch.float64)
    h=(x.T@x).square(); h/=h.sum(); p=torch.rand(12,2,generator=gen,dtype=torch.float64)
    cfg={'max_features':4}
    cpu,counts=objective_forward(h,p,.002,.3,[.001,.01],cfg,'cpu')
    gpu,gcounts=objective_forward(h.cuda(),p.cuda(),.002,.3,[.001,.01],cfg,'cuda')
    torch.testing.assert_close(cpu,gpu.cpu(),atol=1e-7,rtol=1e-6)
    assert counts==gcounts
