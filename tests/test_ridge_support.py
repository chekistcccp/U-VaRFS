import copy
import numpy as np
import pytest
import torch
from uvarfs.ridge_support import ridge_whiten_columns,ridge_starts,fit_ridge_extension,digest,greedy_projection_coordinates
from uvarfs.matching_uvarfs import fit_matching_uvarfs
from test_v20_local_selection import case
from test_v21_matching_distribution import settings,serialized


def test_ridge_leverage_matches_independent_svd_and_inverse_covariance():
    torch.manual_seed(12);x=torch.randn(19,11,dtype=torch.float64)
    z,d=ridge_whiten_columns(x,4)
    u,s,v=torch.linalg.svd(x,full_matrices=False)
    ridge=max(float(s[4:].square().sum()/4),float(s[0].square())*torch.finfo(torch.float64).eps*max(x.shape))
    oracle=(u*(s/(s.square()+ridge).sqrt()))@v
    torch.testing.assert_close(z,oracle,atol=1e-12,rtol=1e-11)
    cov=x.T@x;hat=torch.linalg.solve(cov+ridge*torch.eye(11,dtype=torch.float64),cov)
    torch.testing.assert_close(z.T@z,hat,atol=1e-12,rtol=1e-11)
    assert abs(d['effective_dimension']-float(hat.trace()))<1e-11
    assert d['ridge_lambda']==pytest.approx(ridge)


def test_whitening_exposes_weak_independent_coordinate_without_changing_inference_basis():
    # Five repeated high-energy coordinates can crowd out one independent weak axis.
    a=torch.zeros(12,6,dtype=torch.float64);a[:6,:5]=100.;a[6:,5]=.1
    z,d=ridge_whiten_columns(a,2)
    from uvarfs.matching_diversity import pivot_coordinates
    active,_=pivot_coordinates(z,2)
    assert 5 in active and len(active)==2
    assert d['numerical_rank']==2 and torch.isfinite(z).all()
    zz,dd=ridge_whiten_columns(a*13,2)
    torch.testing.assert_close(z,zz,atol=1e-9,rtol=1e-8)
    assert dd['ridge_lambda']==pytest.approx(d['ridge_lambda']*169,rel=.02)


@pytest.mark.parametrize('x,k',[(torch.ones(2,3),True),(torch.ones(2,3),0),(torch.empty(0,3),1),(torch.tensor([[float('nan')]]),1)])
def test_invalid_ridge_inputs_rejected(x,k):
    with pytest.raises(ValueError):ridge_whiten_columns(x,k)


def test_zero_and_rank_deficient_inputs_are_finite():
    for x in [torch.zeros(5,9),torch.ones(5,9)]:
        z,d=ridge_whiten_columns(x,4)
        assert torch.isfinite(z).all() and np.isfinite(d['ridge_lambda'])


def test_greedy_projection_matches_explicit_independent_residual_oracle():
    torch.manual_seed(19);x=torch.randn(21,10,dtype=torch.float64)
    active,info=greedy_projection_coordinates(x,5)
    residual=x.clone();selected=[];gains=[]
    for _ in range(5):
        choices=[]
        for j in range(x.shape[1]):
            if j in selected:continue
            q=residual[:,j]/residual[:,j].norm()
            choices.append((float((q@residual).square().sum()),j))
        gain,j=max(choices,key=lambda pair:(pair[0],-pair[1]));selected.append(j);gains.append(gain)
        q=residual[:,j]/residual[:,j].norm();residual-=q[:,None]*(q@residual)[None,:]
    assert active.tolist()==sorted(selected)
    np.testing.assert_allclose(info['greedy_actual_projection_gains'],gains,atol=1e-10)
    assert info['remaining_projection_energy']==pytest.approx(float(residual.square().sum()),abs=1e-10)
    full,_=greedy_projection_coordinates(torch.ones(5,9),4)
    assert len(full.unique())==4


def test_generation_preserves_old_pool_holdout_separation_beta0_and_objective():
    x,val,g,vg,paired=case()
    cfg={**settings(),'matching_exchange_max_steps':0,'matching_refit_max_iter':3,'beta':0.}
    solution=serialized(fit_matching_uvarfs(x,val,g,vg,paired,cfg))
    before=copy.deepcopy(solution)
    opts={'chunk':7,'rank':8,'refit_max_iter':3,'tol':1e-5,'polish_max_iter':3,'polish_ftol':1e-12}
    a=fit_ridge_extension(x,val,g,vg,None,solution,opts,lambda _:None)
    b=fit_ridge_extension(x,val*1.2,g,vg,(torch.full_like(x,float('nan')),torch.empty(0)),solution,opts,lambda _:None)
    assert solution==before and a['source']['source_pool_sha256']==solution['candidate_pool_fit_only_sha256']
    assert a['candidate_pool_fit_only_sha256']==b['candidate_pool_fit_only_sha256']
    assert a['objective_definition']==solution['objective_definition']
    assert digest(a['candidate_pool'][:len(solution['candidate_pool'])])==solution['candidate_pool_fit_only_sha256']
    assert not a['support_generation_uses_holdout']
    for c in a['candidate_pool']:
        p=np.array(c['weights']);k=len(p)
        assert len(set(c['active']))==k and abs(p.sum()-1)<1e-10
        assert p.min()>=solution['weight_floor_mass']/k-1e-12 and p.max()<=min(1,solution['weight_cap_factor']/k)+1e-12
        assert c['training_terms']['variability_term']==0


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_ridge_cpu_cuda_spectral_identity():
    torch.manual_seed(12);x=torch.randn(49,17,dtype=torch.float64)
    a,da=ridge_whiten_columns(x,5);b,db=ridge_whiten_columns(x.cuda(),5)
    torch.testing.assert_close(a,b.cpu(),atol=1e-10,rtol=1e-9)
    assert da['effective_dimension']==pytest.approx(db['effective_dimension'],rel=1e-10)
