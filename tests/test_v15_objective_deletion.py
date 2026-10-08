"""Normal-only finite deletion checks; no BMAD test label guides pruning."""
import copy

import numpy as np
import pytest
import torch

from uvarfs.cosine_uvarfs import CosineObjective,prune_cosine_support,project_simplex,fit_cosine_uvarfs
from scripts.analyze_results import audit_sparse_path
from test_v12_cosine_uvarfs import config
from test_v14_fixed_support_polish import independent_loss


def stationary_reference():
    x=torch.tensor([[1.,1.,1.,0.],[1.,1.,0.,1.],[1.,1.,-1.,0.],[1.,1.,0.,-1.]],dtype=torch.float64)
    return CosineObjective(x,torch.zeros(2,4,dtype=x.dtype),beta=0),torch.arange(4),torch.full((4,),.25,dtype=x.dtype)


def test_stationary_first_order_ties_drop_an_informative_coordinate_but_finite_loss_resolves_it():
    engine,active,p=stationary_reference()
    before,gradient,_=engine.evaluate(active,p,True)
    assert float(before)==0 and torch.count_nonzero(gradient)==0
    cfg={'cosine_pruning_strategy':'exact_objective_deletion'}
    chosen,weights,diag=prune_cosine_support(engine,active,p,3,cfg,'stationary')
    assert diag['first_order_score_span']==0 and diag['fixed_support_first_order_residual']==0
    assert diag['legacy_proposal']['active']==[0,1,2]
    assert chosen.tolist()==[0,2,3] and diag['exact_proposal_accepted']
    assert diag['selected_initial_objective']<diag['legacy_proposal']['base_objective']
    assert float(independent_loss(engine,chosen,weights))==pytest.approx(1/24,abs=1e-12)
    assert len(diag['finite_deletions'])==4


def test_finite_scores_match_independent_full_sample_oracle_with_new_floor_and_variability():
    rng=torch.Generator().manual_seed(77)
    x=torch.randn(13,7,generator=rng,dtype=torch.float64)
    v=torch.rand(3,7,generator=rng,dtype=torch.float64)
    engine=CosineObjective(x,v,beta=.007,chunk=4)
    active=torch.tensor([0,2,4,5]);p=torch.tensor([.68,.2,.119975,.000025],dtype=x.dtype)
    before=float(independent_loss(engine,active,p))
    _,_,diag=prune_cosine_support(engine,active,p,2,{'cosine_pruning_strategy':'exact_objective_deletion'},'oracle')
    for item in diag['finite_deletions']:
        keep=active!=item['removed_feature']
        weights=project_simplex(p[keep]/p[keep].sum())
        oracle=float(independent_loss(engine,active[keep],weights))
        assert item['valid'] and item['base_objective']==pytest.approx(oracle,abs=1e-12)
        assert item['loss_delta']==pytest.approx(oracle-before,abs=1e-12)
        assert float(weights.sum())==pytest.approx(1,abs=1e-12)
        assert (weights>=1e-4/3).all()
    assert diag['objective_evaluations']>=len(active)+2 and diag['seconds']>=0


def test_joint_multi_deletion_must_retain_legacy_when_single_coordinate_ranking_is_worse():
    rng=torch.Generator().manual_seed(2)
    x=torch.randn(11,8,generator=rng,dtype=torch.float64);v=torch.rand(2,8,generator=rng,dtype=torch.float64)
    p=torch.rand(8,generator=rng,dtype=torch.float64);p/=p.sum()
    engine=CosineObjective(x,v,chunk=4)
    chosen,weights,diag=prune_cosine_support(engine,torch.arange(8),p,4,{'cosine_pruning_strategy':'exact_objective_deletion'},'joint')
    assert diag['exact_proposal']['base_objective']>diag['legacy_proposal']['base_objective']
    assert diag['selected_proposal']=='gradient_direction' and not diag['exact_proposal_accepted']
    assert chosen.tolist()==diag['legacy_proposal']['active']
    assert float(independent_loss(engine,chosen,weights))==pytest.approx(diag['legacy_proposal']['base_objective'],abs=1e-12)


def test_zero_row_deletions_are_invalid_and_cannot_be_normalized_into_valid_scores():
    x=torch.eye(4,dtype=torch.float64)
    engine=CosineObjective(x,torch.zeros(2,4,dtype=x.dtype),beta=0)
    active,p,diag=prune_cosine_support(engine,torch.arange(4),torch.full((4,),.25,dtype=x.dtype),3,
                                     {'cosine_pruning_strategy':'exact_objective_deletion'},'coverage')
    assert active is None and p is None and diag['selected_proposal']=='invalid'
    assert all(not item['valid'] and item['base_objective'] is None for item in diag['finite_deletions'])
    assert diag['legacy_proposal'] is None and diag['exact_proposal'] is None


@pytest.mark.parametrize('count',[0,4,5])
def test_invalid_pruning_budget_fails(count):
    engine,active,p=stationary_reference()
    with pytest.raises(ValueError,match='pruning strategy or count'):
        prune_cosine_support(engine,active,p,count,{},'invalid')


def test_legacy_rule_remains_an_explicit_normal_solver_control():
    engine,active,p=stationary_reference()
    chosen,_,diag=prune_cosine_support(engine,active,p,3,{'cosine_pruning_strategy':'gradient_direction'},'legacy')
    assert chosen.tolist()==[0,1,2]
    assert diag['finite_deletions']==[] and diag['exact_proposal'] is None
    with pytest.raises(ValueError,match='pruning strategy'):
        prune_cosine_support(engine,active,p,3,{'cosine_pruning_strategy':'labels'},'invalid')


def test_full_path_records_scoring_costs_and_rejects_forged_guards_or_missing_events():
    rng=torch.Generator().manual_seed(115)
    x=torch.randn(15,8,generator=rng,dtype=torch.float64);v=torch.rand(2,8,generator=rng,dtype=torch.float64)
    cfg=config(cosine_pruning_strategy='exact_objective_deletion',cosine_polish_max_iter=12)
    old={'active':np.array([0,1,3]),'budgeted_weights':np.array([.4,.3,0,.3,0,0,0,0])}
    result=fit_cosine_uvarfs(x,v,cfg,old,'full-path')
    assert audit_sparse_path(result,cfg)['pruning_guards_verified']
    assert result['pruning_events'] and result['pruning_summary']['objective_evaluations']>0
    for c in result['candidate_pool']:
        event=c['pruning_event_id']
        if event is not None:
            e=result['pruning_events'][event]
            assert e['parent_candidate_id']<c['candidate_id']
            assert c['base_objective']<=e['selected_initial_objective']+1e-8
    broken=copy.deepcopy(result);broken['pruning_events'][0]['exact_proposal_accepted']=not broken['pruning_events'][0]['exact_proposal_accepted']
    with pytest.raises(ValueError,match='joint objective guard'): audit_sparse_path(broken,cfg)
    broken=copy.deepcopy(result);broken['pruning_events'][0]['finite_deletions'].pop()
    with pytest.raises(ValueError,match='omit active coordinates'): audit_sparse_path(broken,cfg)
    broken=copy.deepcopy(result);broken['pruning_summary']['objective_evaluations']+=1
    with pytest.raises(ValueError,match='totals omit'): audit_sparse_path(broken,cfg)
    broken=copy.deepcopy(result);del broken['solver_manifest']['pruning_strategy']
    with pytest.raises(ValueError,match='manifest is missing'): audit_sparse_path(broken,cfg)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA runtime unavailable locally')
def test_cpu_cuda_finite_deletion_and_actual_joint_guard_agree():
    engine,active,p=stationary_reference()
    gpu=CosineObjective(engine.x.cuda(),torch.zeros(2,4,dtype=p.dtype,device='cuda'),beta=0)
    cfg={'cosine_pruning_strategy':'exact_objective_deletion'}
    a,q,d=prune_cosine_support(engine,active,p,3,cfg,'cpu')
    b,r,e=prune_cosine_support(gpu,active.cuda(),p.cuda(),3,cfg,'cuda')
    assert r.is_cuda
    torch.testing.assert_close(a,b.cpu());torch.testing.assert_close(q,r.cpu())
    assert d['selected_initial_objective']==pytest.approx(e['selected_initial_objective'],abs=1e-11)
