from itertools import combinations
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.analyze_results import validate_output
from uvarfs.asls import cosine_gram, fit_asls, reselect_asls, select_layer_subset
from uvarfs.u_varfs import _objective_certificate, _select_solution


def test_search_finds_minimum_feasible_subset_missed_by_gate_prefix():
    generator=torch.Generator().manual_seed(26)
    a=torch.randn(32,4,generator=generator)
    b=torch.randn(32,4,generator=generator)
    features={i:a if i<=3 else b for i in range(1,7)}
    grams=torch.stack([cosine_gram(x) for x in features.values()]).double()
    q=(grams.flatten(1)@grams.flatten(1).T).numpy()
    probabilities=[.999,.998,.997,.996,.995,.994]
    cfg={'min_layers':2,'max_layers':3,'geometry_tolerance':1e-6}
    prefix=select_layer_subset(list(features),probabilities,q,cfg,'gate_prefix')
    search=select_layer_subset(list(features),probabilities,q,cfg,'geometry_search')
    assert prefix['selected_layers']==[1,2,3] and not prefix['feasible']
    assert search['selected_layers']==[1,4] and search['feasible']
    assert search['evaluated_subsets']==15
    consensus=grams.mean(0)
    for point in search['selection_path']:
        indices=[layer-1 for layer in point['layers']]
        direct=float((grams[indices].mean(0)-consensus).norm()/consensus.norm())
        assert point['geometry_error']==pytest.approx(direct,abs=2e-8)


def test_search_infeasible_fallback_checks_entire_layer_budget():
    generator=torch.Generator().manual_seed(5)
    grams=torch.stack([cosine_gram(torch.randn(16,4,generator=generator)) for _ in range(12)]).double()
    q=(grams.flatten(1)@grams.flatten(1).T).numpy()
    cfg={'min_layers':2,'max_layers':6,'geometry_tolerance':1e-12}
    result=select_layer_subset(list(range(1,13)),np.linspace(.99,.98,12),q,cfg,'geometry_search')
    reference=min(float((grams[list(ids)].mean(0)-grams.mean(0)).norm()/grams.mean(0).norm())
                  for k in range(2,7) for ids in combinations(range(12),k))
    assert result['evaluated_subsets']==2497
    assert not result['feasible']
    assert result['geometry_error']==pytest.approx(reference,rel=1e-10)
    assert result['selection_reason']=='minimum_geometry_error_fallback'


@pytest.mark.parametrize('mode',['pooled','patch'])
def test_saved_normal_geometry_reuses_fit_gates_and_original_prefix(mode):
    generator=torch.Generator().manual_seed(18)
    features={i:torch.randn(16,4,generator=generator) for i in range(1,7)}
    cfg={'steps':5,'geometry_representation':mode,'min_layers':2,'max_layers':4,'geometry_tolerance':.05}
    fitted=fit_asls(features,{i:i/6 for i in features},cfg)
    replay=reselect_asls(fitted,cfg,'gate_prefix')
    assert fitted['discrete_selection']=='gate_prefix'
    assert replay['selected_layers']==fitted['selected_layers']
    assert replay['geometry_error']==pytest.approx(fitted['geometry_error'],abs=1e-7)
    search=reselect_asls(fitted,cfg,'geometry_search')
    assert search['geometry_representation']==mode
    assert search['geometry_error']<=replay['geometry_error']+1e-7
    assert fitted['probabilities']=={str(i):fitted['probabilities'][str(i)] for i in features}


def test_budgeted_certificate_distinguishes_converged_continuous_solution():
    h=torch.diag(torch.tensor([.6,.3,.1],dtype=torch.float64))
    p=torch.zeros(3,0,dtype=torch.float64)
    lambdas=[.01]
    weights=(1-.01/h.diagonal())[:,None]
    certificate=_objective_certificate(weights,h,p,0,0,lambdas)
    solver={'relative_change':[0.],'projected_residual':[0.],'converged':[True],
            'objective_converged':[True],**{k:v.tolist() for k,v in certificate.items()}}
    cfg={'min_features':1,'max_features':1,'geometry_tolerance':.05,'tol':1e-8}
    chosen,active,retained,path=_select_solution(weights,h,lambdas,cfg,solver,(p,0,0))
    assert chosen==0 and active.tolist()==[0]
    point=path[0]
    assert point['objective_converged']
    assert point['objective_certificate_scope']=='continuous_full_weights'
    assert not point['budgeted_objective_converged']
    assert point['budgeted_box_optimality_gap']>.3
    assert point['fixed_support_geometry_floor']==pytest.approx(np.sqrt(.4))
    assert not point['fixed_support_can_meet_tolerance']
    expected=.5*((1-retained)@h@(1-retained))+.01*retained.sum()
    assert point['budgeted_objective']==pytest.approx(float(expected))
    torch.testing.assert_close(retained[active],weights[active,0])
    assert not point['feasible']


def test_fixed_support_floor_is_lower_bound_only_on_that_support():
    h=torch.tensor([[.6,.1,.05],[.1,.3,.02],[.05,.02,.1]],dtype=torch.float64)
    weights=torch.tensor([[.7],[.2],[.1]],dtype=torch.float64)
    solver={'relative_change':[0.],'projected_residual':[0.],'converged':[True]}
    _,active,_,path=_select_solution(weights,h,[.001],{'min_features':1,'max_features':1},solver)
    support=torch.zeros(3,dtype=torch.float64); support[active]=1
    floor=path[0]['fixed_support_geometry_floor']
    for value in torch.linspace(0,1,20):
        difference=1-value*support
        assert float(torch.sqrt(difference@h@difference/h.sum()))>=floor-1e-12


def test_analysis_cannot_overwrite_original_or_tracked_historical_report(tmp_path):
    with pytest.raises(ValueError,match='outside the original'):
        validate_output(tmp_path,tmp_path/'nested')
    repo=Path(__file__).resolve().parents[1]
    with pytest.raises(ValueError,match='tracked historical'):
        validate_output(tmp_path,repo/'reports'/'2026-10-06-v4-analysis')
