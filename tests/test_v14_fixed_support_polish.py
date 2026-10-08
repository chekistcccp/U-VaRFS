"""Same-objective solver checks; synthetic normal features, no anomaly tuning."""
from types import SimpleNamespace
import copy

import numpy as np
import pytest
import torch

from uvarfs.cosine_uvarfs import CosineObjective, polish_simplex, project_simplex, refit_simplex, fit_cosine_uvarfs
from scripts.analyze_results import audit_sparse_path, audit_cosine_refit
from test_v12_cosine_uvarfs import config


def ill_conditioned_normal(dtype=torch.float32):
    rng=torch.Generator().manual_seed(41)
    latent=torch.randn(64,5,generator=rng,dtype=dtype)
    loadings=torch.randn(5,48,generator=rng,dtype=dtype)
    x=(latent@loadings+.08*torch.randn(64,48,generator=rng,dtype=dtype))
    x*=torch.tensor([1.]*16+[12.]*16+[70.]*16,dtype=dtype)
    variability=torch.rand(3,48,generator=rng,dtype=dtype)
    active=torch.arange(0,48,4)
    p=torch.rand(len(active),generator=rng,dtype=dtype); p/=p.sum()
    return CosineObjective(x,variability,chunk=64),active,p


def independent_loss(engine,active,p):
    # Separate sample-space expression, no row blocking or analytic-gradient code.
    x=engine.x.double(); weights=torch.zeros(x.shape[1],dtype=torch.float64)
    weights[active]=p.double()
    h=x.square()@weights
    actual=(x*weights)@x.T/torch.sqrt(h[:,None]*h[None,:])
    reference=engine.reference.double()
    variability=engine.P.double().T@weights
    term=variability.square().sum()/engine.var_base.double() if engine.var_base>0 else 0.
    return .5*(actual-reference).square().sum()/reference.square().sum()+engine.beta*term


def test_stalled_projected_solver_can_descend_further_without_changing_support_or_loss():
    engine,active,initial=ill_conditioned_normal()
    cfg={'cosine_refit_max_iter':2000,'cosine_polish_max_iter':0,'tol':1e-5,'log_every':10000}
    original,_,old=refit_simplex(engine,active,initial,cfg,'unpolished')
    assert not old['fixed_support_converged']
    before=float(independent_loss(engine,active,original))
    endpoint,terms,diag=polish_simplex(engine,active,original,{**cfg,'cosine_polish_max_iter':200},'polished')
    assert diag['accepted'] and diag['after']<diag['before']
    assert float(independent_loss(engine,active,endpoint))<before-1e-5
    assert float(independent_loss(engine,active,endpoint))==pytest.approx(terms['representation_term']+terms['variability_term'],abs=3e-8)
    assert endpoint.shape==original.shape and endpoint.dtype==original.dtype
    assert float(endpoint.sum())==pytest.approx(1,abs=1e-6)
    assert (endpoint>=1e-4/len(endpoint)).all()
    assert diag['objective_evaluations']>0 and diag['iterations']<=200
    # SciPy termination does not determine fixed-support stationarity.
    _,gradient,_=engine.evaluate(active,endpoint,True)
    residual=float((endpoint-project_simplex(endpoint-gradient)).norm())
    assert diag['residual_after']==pytest.approx(residual)


def small_engine():
    rng=torch.Generator().manual_seed(73)
    engine=CosineObjective(torch.randn(13,5,generator=rng,dtype=torch.float64),torch.zeros(2,5,dtype=torch.float64),chunk=4)
    return engine,torch.tensor([0,1,3]),torch.tensor([.2,.55,.25],dtype=torch.float64)


@pytest.mark.parametrize('kind',['worse','nan','wrong_shape','overflow'])
def test_optimizer_success_cannot_override_endpoint_descent_or_validity(monkeypatch,kind):
    import uvarfs.cosine_uvarfs as solver
    engine,active,p=small_engine()
    floor=1e-4/len(p)
    vertices=[torch.full_like(p,floor) for _ in p]
    for j,v in enumerate(vertices): v[j]=1-2*floor
    worst=max(vertices,key=lambda v:float(engine.evaluate(active,v)[0]))
    assert engine.evaluate(active,worst)[0]>engine.evaluate(active,p)[0]
    endpoint=worst.numpy() if kind=='worse' else np.full(3,np.nan) if kind=='nan' else np.ones(4) if kind=='wrong_shape' else np.full(3,1e300)
    if kind=='overflow':
        p=p.float(); engine=CosineObjective(engine.x.float(),torch.zeros(2,5),chunk=4)
    monkeypatch.setattr(solver,'minimize',lambda *a,**k:SimpleNamespace(x=endpoint,success=True,status=0,message='success',nit=1))
    result,_,diag=polish_simplex(engine,active,p,{'tol':1e-12},'reject')
    torch.testing.assert_close(result,p,rtol=0,atol=0)
    assert diag['optimizer_success'] is True and not diag['accepted']
    assert diag['after']==diag['before'] and diag['residual_after']==diag['residual_before']


def test_unsuccessful_optimizer_endpoint_can_be_accepted_only_for_actual_descent(monkeypatch):
    import uvarfs.cosine_uvarfs as solver
    engine,active,p=small_engine()
    _,gradient,_=engine.evaluate(active,p,True)
    better=project_simplex(p-.01*gradient)
    assert engine.evaluate(active,better)[0]<engine.evaluate(active,p)[0]
    def stopped(fun,initial,**kwargs):
        value,g=fun(initial)
        assert value==pytest.approx(float(independent_loss(engine,active,p)),abs=1e-12)
        torch.testing.assert_close(torch.tensor(g),gradient)
        assert kwargs['jac'] is True and kwargs['constraints']['fun'](initial)==pytest.approx(0)
        return SimpleNamespace(x=better.numpy(),success=False,status=9,message='iteration limit',nit=2)
    monkeypatch.setattr(solver,'minimize',stopped)
    result,_,diag=polish_simplex(engine,active,p,{},'iteration-limit')
    torch.testing.assert_close(result,better,atol=1e-15,rtol=0)
    assert diag['accepted'] and diag['optimizer_success'] is False


def test_success_flag_does_not_make_nonstationary_refit_converged(monkeypatch):
    import uvarfs.cosine_uvarfs as solver
    engine,active,p=small_engine()
    monkeypatch.setattr(solver,'minimize',lambda fun,initial,**k:SimpleNamespace(x=initial,success=True,status=0,message='success',nit=1))
    cfg={'cosine_refit_max_iter':1,'max_iter':1,'tol':1e-14}
    _,_,diag=refit_simplex(engine,active,p,cfg,'scope')
    assert diag['polish']['optimizer_success'] is True
    assert not diag['fixed_support_converged']
    assert diag['fixed_support_first_order_residual']>cfg['tol']
    audit_cosine_refit(diag,cfg)
    broken=copy.deepcopy(diag); broken['fixed_support_converged']=True
    with pytest.raises(ValueError,match='actual residual'): audit_cosine_refit(broken,cfg)


@pytest.mark.parametrize('setting',[{'cosine_polish_max_iter':-1},{'cosine_polish_ftol':0},{'cosine_polish_ftol':float('nan')}])
def test_invalid_numerical_polish_settings_fail(setting):
    engine,active,p=small_engine()
    with pytest.raises(ValueError,match='polish limits'): polish_simplex(engine,active,p,setting,'invalid')


def test_audit_includes_all_candidate_and_exchange_refits_and_rejects_missing_costs():
    rng=torch.Generator().manual_seed(115)
    x=torch.randn(17,8,generator=rng,dtype=torch.float64)
    v=torch.rand(2,8,generator=rng,dtype=torch.float64)
    cfg=config(cosine_polish_max_iter=12,cosine_polish_ftol=1e-12)
    old={'active':np.array([0,1,3]),'budgeted_weights':np.array([.4,.3,0,.3,0,0,0,0])}
    result=fit_cosine_uvarfs(x,v,cfg,old,'audit')
    assert audit_sparse_path(result,cfg)['polish_guards_verified']
    refits=[c['initial_refit'] for c in result['candidate_pool']]+[e['refit'] for c in result['candidate_pool'] for e in c['exchange_history']]
    assert result['solver_iterations']==sum(r['solver_iterations'] for r in refits)
    assert result['polish_summary']['iterations']==sum(r['polish']['iterations'] for r in refits)
    for key in ['seconds','iterations','objective_evaluations']:
        broken=copy.deepcopy(result); broken['polish_summary'][key]+=1
        with pytest.raises(ValueError,match='totals omit'): audit_sparse_path(broken,cfg)
    broken=copy.deepcopy(result); del broken['solver_manifest']['polish_method']
    with pytest.raises(ValueError,match='manifest is missing'): audit_sparse_path(broken,cfg)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA runtime unavailable locally')
def test_cpu_controlled_polish_keeps_cuda_objective_and_improves_actual_loss():
    engine,active,p=small_engine()
    gpu=CosineObjective(engine.x.cuda(),torch.zeros(2,5,dtype=torch.float64,device='cuda'),chunk=4)
    before=float(gpu.evaluate(active.cuda(),p.cuda())[0])
    result,_,diag=polish_simplex(gpu,active.cuda(),p.cuda(),{'tol':1e-10},'cuda')
    assert result.is_cuda and diag['after']<=before
    assert float(result.sum())==pytest.approx(1,abs=1e-12)
