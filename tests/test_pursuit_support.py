import copy,itertools
import numpy as np
import pytest
import torch
from uvarfs.pursuit_support import (project_sparse_capped_simplex,first_support_breakpoint,
    pursue_support,fit_pursuit_extension,digest,extend_fit_pool)
from uvarfs.matching_uvarfs import fit_matching_uvarfs,MatchingObjective
from test_v20_local_selection import case
from test_v21_matching_distribution import settings,serialized


def options():
    return {'chunk':7,'starts':2,'rounds':2,'breakpoint_quantiles':[0.,.25,.5,.75],
            'breakpoint_multiplier':1.05,'refit_max_iter':5,'tol':1e-5,'polish_max_iter':4,'polish_ftol':1e-12}


def independent_projection(v,lo,hi):
    # Independent scalar bisection, not implementation's breakpoint projection.
    left=float(v.min()-hi-2);right=float(v.max()-lo+2)
    for _ in range(150):
        mid=(left+right)/2
        if float((v-mid).clamp(lo,hi).sum())>1:left=mid
        else:right=mid
    return (v-(left+right)/2).clamp(lo,hi)


@pytest.mark.parametrize('seed',[1,12,38])
def test_sparse_projection_matches_exhaustive_support_bisection_oracle(seed):
    torch.manual_seed(seed);v=torch.randn(8,dtype=torch.float64)-2
    k=3;floor=.1;cap=1.7
    active,p=project_sparse_capped_simplex(v,k,floor,cap)
    dense=torch.zeros_like(v);dense[active]=p;actual=float((dense-v).square().sum())
    losses=[]
    for ids in itertools.combinations(range(len(v)),k):
        q=independent_projection(v[list(ids)],floor/k,cap/k)
        full=torch.zeros_like(v);full[list(ids)]=q;losses.append(float((full-v).square().sum()))
    assert actual==pytest.approx(min(losses),abs=1e-10)
    assert len(active)==k and p.min()>=floor/k-1e-12 and p.max()<=cap/k+1e-12


def test_projection_ties_shift_invariance_and_full_budget():
    active,p=project_sparse_capped_simplex(torch.zeros(7,dtype=torch.float64),3)
    assert active.tolist()==[0,1,2]
    torch.testing.assert_close(p,torch.full_like(p,1/3))
    v=torch.tensor([-3.,2.,-4.,7.,1.],dtype=torch.float64)
    a,p=project_sparse_capped_simplex(v,3);b,q=project_sparse_capped_simplex(v+31,3)
    assert a.tolist()==b.tolist();torch.testing.assert_close(p,q)
    a,p=project_sparse_capped_simplex(v,5);assert a.tolist()==list(range(5))


@pytest.mark.parametrize('v,k',[(torch.ones(3),True),(torch.ones(3),0),(torch.ones(3),4),(torch.tensor([float('nan')]),1)])
def test_projection_rejects_invalid(v,k):
    with pytest.raises(ValueError):project_sparse_capped_simplex(v,k)


def test_first_breakpoint_is_actual_ranking_crossing_and_has_no_label_inputs():
    a=torch.tensor([0,1,2]);p=torch.tensor([.2,.3,.5],dtype=torch.float64)
    g=torch.tensor([1.,2.,3.,-2.,-1.],dtype=torch.float64)
    t=first_support_breakpoint(a,p,g);assert t==pytest.approx(.2/3)
    dense=torch.zeros_like(g);dense[a]=p
    before,_=project_sparse_capped_simplex(dense-t*.999*g,3)
    after,_=project_sparse_capped_simplex(dense-t*1.001*g,3)
    assert before.tolist()==a.tolist() and after.tolist()!=a.tolist()
    assert first_support_breakpoint(a,p,g+13)==pytest.approx(t)
    assert first_support_breakpoint(a,p,torch.tensor([-3.,-3.,-3.,1.,2.])) is None
    assert first_support_breakpoint(torch.arange(3),p,g[:3]) is None


def test_joint_pursuit_changes_multiple_coordinates_and_never_accepts_bad_loss():
    class Quadratic:
        q=torch.empty(1,6,dtype=torch.float64);floor_mass=1e-4;cap_factor=4.
        target=torch.tensor([0.,0.,0.,.2,.3,.5],dtype=torch.float64)
        def project(self,p):
            from uvarfs.performance_uvarfs import project_bounded_simplex
            return project_bounded_simplex(p,self.floor_mass,self.cap_factor)
        def evaluate(self,a,p,gradient=False,full_gradient=False):
            v=torch.zeros(6,dtype=torch.float64);v[a]=p;g=v-self.target
            return float(g.square().sum()/2),g if full_gradient else g[a],{}
        def refit(self,a,p,maximum,tol):
            q=self.project(self.target[a]);return q,self.evaluate(a,q)[0],{}, {'iterations':1}
    e=Quadratic();records=[]
    def record(a,p,name,diag=None):
        records.append((a.clone(),p.clone(),name,copy.deepcopy(diag)));return len(records)-1
    a=torch.tensor([0,1,2]);p=torch.ones(3,dtype=torch.float64)/3
    b,q,d=pursue_support(e,a,p,options(),record,lambda _:None)
    assert b.tolist()==[3,4,5] and len(set(a.tolist())-set(b.tolist()))==3
    assert all(t['after']<t['before'] for t in d['accepted_trace'])
    assert all(len(r[0])==3 and float(r[1].min())>0 for r in records)


def test_fit_pool_holdout_independence_beta_zero_and_old_pool_preservation():
    x,v,g,vg,paired=case()
    cfg={**settings(),'matching_exchange_max_steps':0,'matching_refit_max_iter':3,'beta':0.}
    solution=serialized(fit_matching_uvarfs(x,v,g,vg,paired,cfg));before=copy.deepcopy(solution)
    a=fit_pursuit_extension(x,v,g,vg,None,solution,options(),lambda _:None)
    b=fit_pursuit_extension(x,v,g,vg,(torch.full_like(x,float('nan')),torch.empty(0)),solution,options(),lambda _:None)
    assert solution==before
    assert a['candidate_pool_fit_only_sha256']==b['candidate_pool_fit_only_sha256']
    _,fit_only,_=extend_fit_pool(x,g,None,solution,options(),lambda _:None)
    assert digest(fit_only)==a['candidate_pool_fit_only_sha256']
    import inspect
    assert 'heldout' not in inspect.signature(extend_fit_pool).parameters
    assert digest(a['candidate_pool'][:len(solution['candidate_pool'])])==solution['candidate_pool_fit_only_sha256']
    assert a['source']['start_policy']=='lowest_fit_loss_unique_supports'
    assert all(c['training_terms']['variability_term']==0 for c in a['candidate_pool'])
    for t in a['source']['attempts']:
        assert all(c['after']<c['before'] for c in t['pursuit']['accepted_trace'])
    with pytest.raises(ValueError,match='disjoint'):fit_pursuit_extension(x,v,g,g[:len(v)],None,solution,options())


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_cpu_cuda_projection_and_full_matching_pursuit_loss():
    torch.manual_seed(31);v=torch.randn(31,dtype=torch.float64)
    a,p=project_sparse_capped_simplex(v,9);b,q=project_sparse_capped_simplex(v.cuda(),9)
    torch.testing.assert_close(a,b.cpu());torch.testing.assert_close(p,q.cpu(),atol=1e-12,rtol=1e-11)
    x,h,g,hg,paired=case();cfg={**settings(),'matching_exchange_max_steps':0,'matching_refit_max_iter':3,'beta':0.}
    sol=serialized(fit_matching_uvarfs(x,h,g,hg,paired,cfg))
    result=fit_pursuit_extension(x.cuda(),h.cuda(),g.cuda(),hg.cuda(),None,sol,options(),lambda _:None)
    assert result['winner_candidate_id']>=0 and result['fixed_feature_budget']==sol['fixed_feature_budget']
