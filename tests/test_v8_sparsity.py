import numpy as np
import pytest
import torch
from scipy.optimize import minimize

from uvarfs.sparse_support import objective_prune, refit_fixed_support, refine_budget, top_weight_budget
from uvarfs.u_varfs import _objective_certificate, fit_uvarfs, apply_uvarfs
from uvarfs.asls import fit_asls, reselect_asls


def test_upgraded_layer_default_and_cached_prefix_respects_reselection_constraints():
    generator=torch.Generator().manual_seed(8)
    x=torch.randn(16,4,generator=generator)
    features={i:x.clone() for i in range(1,7)}
    cfg={'steps':3,'min_layers':2,'max_layers':6,'geometry_tolerance':.05}
    fitted=fit_asls(features,{i:0 for i in features},cfg)
    assert fitted['discrete_selection']=='geometry_search'
    assert len(fitted['gate_prefix_selection']['selected_layers'])==2
    original=reselect_asls(fitted,cfg,'gate_prefix')
    assert original['selected_layers']==fitted['gate_prefix_selection']['selected_layers']
    changed=reselect_asls(fitted,{**cfg,'min_layers':4},'gate_prefix')
    assert len(changed['selected_layers'])==4


def test_incremental_pruning_matches_independent_full_objective_deletions():
    generator=torch.Generator().manual_seed(37)
    x=torch.randn(15,7,generator=generator,dtype=torch.float64)
    h=(x.T@x).square(); h/=h.sum()
    p=torch.rand(7,2,generator=generator,dtype=torch.float64)
    weights=torch.rand(7,3,generator=generator,dtype=torch.float64)
    weights[2:,2]=0
    lambdas=[.01,.003,.001]; beta=.13; scale=.4
    actual,deletions=objective_prune(weights,h,p,beta,scale,lambdas,{'max_features':3},'test')
    expected=weights.clone()
    for column,lam in enumerate(lambdas):
        while int((expected[:,column]>0).sum())>3:
            candidates=[]
            for index in torch.where(expected[:,column]>0)[0].tolist():
                trial=expected[:,column].clone(); trial[index]=0
                difference=1-trial
                value=.5*difference@h@difference+beta*scale*(p.T@trial).square().sum()+lam*trial.sum()
                candidates.append((float(value),index))
            expected[min(candidates)[1],column]=0
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    assert deletions==[4,4,0]
    torch.testing.assert_close(actual[:,2],weights[:,2])


def test_fixed_support_refit_keeps_omitted_feature_cross_terms():
    h=torch.tensor([[1.,.4],[.4,1.]],dtype=torch.float64)
    p=torch.zeros(2,0,dtype=torch.float64)
    seed=torch.tensor([[.2],[0.]],dtype=torch.float64)
    cfg={'min_features':1,'max_iter':1000,'tol':1e-12,'check_every':5}
    fitted,info=refit_fixed_support(seed,torch.ones_like(seed),h,p,0,0,[.5],cfg,'test')
    # Correct target=(H*1)_0=1.4 gives .9; H_SS*1 would incorrectly give .5.
    assert float(fitted[0,0])==pytest.approx(.9,abs=1e-10)
    assert float(fitted[1,0])==0
    assert info['fixed_support_converged']==[True]


def test_batched_support_refit_matches_scipy_and_handles_padded_row_zero():
    h=torch.tensor([[1.,.4,.2],[.4,.8,.3],[.2,.3,.6]],dtype=torch.float64)
    p=torch.tensor([[1.,.3],[.2,.4],[.3,.8]],dtype=torch.float64)
    seed=torch.tensor([[.2,.3],[0.,0.],[.4,0.]],dtype=torch.float64)
    lambdas=[.2,.6]; beta=.25; scale=.8
    cfg={'min_features':1,'max_iter':3000,'tol':1e-10,'check_every':5}
    fitted,info=refit_fixed_support(seed,torch.ones_like(seed),h,p,beta,scale,lambdas,cfg,'test')
    H=h.numpy(); P=p.numpy()
    for j,ids in enumerate([[0,2],[0]]):
        def loss(z):
            full=np.zeros(3); full[ids]=z
            d=1-full
            return .5*d@H@d+beta*scale*np.square(P.T@full).sum()+lambdas[j]*full.sum()
        reference=minimize(loss,seed[ids,j].numpy(),bounds=[(0,1)]*len(ids),method='SLSQP',
                           options={'ftol':1e-13,'maxiter':1000})
        assert reference.success
        np.testing.assert_allclose(fitted[ids,j],reference.x,atol=2e-7,rtol=2e-7)
        assert loss(fitted[ids,j].numpy())==pytest.approx(reference.fun,abs=1e-11)
    assert float(fitted[0,1])>0  # Padded zero indices must not erase this coefficient.
    assert info['fixed_support_converged']==[True,True]


def test_objective_support_keeps_geometry_dominant_dimension_despite_smaller_weight():
    h=torch.diag(torch.tensor([.9,.05,.05],dtype=torch.float64))
    p=torch.tensor([[1.],[0.],[0.]],dtype=torch.float64)
    lam=1e-5; beta=.1
    weights=((h.diagonal()-lam)/(h.diagonal()+2*beta*p[:,0].square()))[:,None]
    cfg={'min_features':1,'max_features':1,'max_iter':1000,'tol':1e-10}
    old=top_weight_budget(weights,cfg)
    assert torch.where(old[:,0]>0)[0].tolist()==[1]
    retained,diagnostics=refine_budget(weights,h,p,beta,1,[lam],cfg,'test')
    assert torch.where(retained[:,0]>0)[0].tolist()==[0]
    assert diagnostics[0]['objective_improvement']>.3
    assert diagnostics[0]['geometry_improvement']>.5
    assert diagnostics[0]['budget_source']!='top_weights'


@pytest.mark.parametrize('beta',[0.,.002,.2])
def test_every_lambda_respects_budget_and_normal_objective_geometry_safeguard(beta):
    generator=torch.Generator().manual_seed(19)
    x=torch.randn(64,15,generator=generator)
    variability=torch.rand(3,15,generator=generator)
    cfg={'beta':beta,'lambda_grid':[.1,.01,.001,.0001],'min_features':2,'max_features':4,
         'geometry_tolerance':.2,'max_iter':200,'support_refit_max_iter':400,'tol':1e-6}
    result=fit_uvarfs(x,variability,cfg)
    old=result['legacy_top_weights']
    for point,reference in zip(result['lambda_path'],old['lambda_path']):
        assert point['lambda']==reference['lambda']
        assert point['budgeted_objective']<=reference['budgeted_objective']+1e-7
        assert point['geometry_error']<=reference['geometry_error']+1e-7
        assert 2<=point['retained_features']<=4
    assert result['geometry_error']<=old['geometry_error']+1e-7
    assert result['lambda'] in cfg['lambda_grid']
    assert np.count_nonzero(result['budgeted_weights'])<=4
    assert apply_uvarfs(x,result).shape[1]==len(result['active'])


def test_legacy_ablation_exactly_matches_explicit_original_strategy():
    generator=torch.Generator().manual_seed(4)
    x=torch.randn(32,8,generator=generator)
    variability=torch.rand(2,8,generator=generator)
    cfg={'beta':.002,'lambda_grid':[.01,.0001],'min_features':2,'max_features':4,
         'geometry_tolerance':.1,'max_iter':200,'tol':1e-6}
    upgraded=fit_uvarfs(x,variability,cfg)
    original=fit_uvarfs(x,variability,{**cfg,'sparsity_strategy':'top_weights'})
    old=upgraded['legacy_top_weights']
    assert old['lambda']==original['lambda']
    for field in ['weights','active','scales','budgeted_weights']:
        np.testing.assert_array_equal(old[field],original[field])
    assert old['lambda_path']==original['lambda_path']


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA runtime unavailable locally')
def test_cuda_support_search_matches_cpu():
    generator=torch.Generator().manual_seed(41)
    x=torch.randn(32,12,generator=generator,dtype=torch.float64)
    h=(x.T@x).square(); h/=h.sum()
    p=torch.rand(12,2,generator=generator,dtype=torch.float64)
    weights=torch.rand(12,3,generator=generator,dtype=torch.float64)
    cfg={'min_features':2,'max_features':4,'max_iter':500,'tol':1e-9}
    cpu,_=refine_budget(weights,h,p,.002,.3,[.01,.001,.0001],cfg,'cpu')
    gpu,_=refine_budget(weights.cuda(),h.cuda(),p.cuda(),.002,.3,[.01,.001,.0001],cfg,'cuda')
    torch.testing.assert_close(gpu.cpu(),cpu,rtol=1e-5,atol=1e-6)
