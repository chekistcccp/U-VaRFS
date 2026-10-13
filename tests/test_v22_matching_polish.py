import copy
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from uvarfs.matching_uvarfs import MatchingObjective, fit_matching_uvarfs, generate_fit_only_pool
from uvarfs.matching_polish import polish_fixed_support
from uvarfs.matching_diversity import pivot_coordinates, diverse_starts
from uvarfs.protocol import expected_method_names
from uvarfs.objectives import experiment_branches
from uvarfs.utils import load_config
from test_v20_local_selection import case
from test_v21_matching_distribution import settings, serialized, matching_fixture
from test_v10_representation import CountingExtractor, CPUIndex


def test_polish_preserves_feasibility_true_loss_and_reports_own_residual():
    x,_,g,_,paired=case()
    engine=MatchingObjective(x,x,g,g,.1,paired,beta=.002,chunk=7)
    active=torch.tensor([0,2,4,6,9]);initial=torch.full((5,),.2,dtype=torch.float64)
    before=engine.evaluate(active,initial)[0]
    p,value,terms,diag=polish_fixed_support(engine,active,initial,40)
    actual,gradient,actual_terms=engine.evaluate(active,p,True)
    assert actual==value and actual_terms==terms and value<=before+1e-10
    assert abs(float(p.sum())-1)<1e-12 and float(p.min())>=1e-4/5-1e-12 and float(p.max())<=4/5+1e-12
    assert diag['residual']==float((p-engine.project(p-gradient)).abs().max())
    assert diag['scipy_success'] in (True,False)  # Not a stationarity certificate.
    assert np.all(np.diff(diag['training_history'])<=1e-12)


def test_failed_nonfinite_optimizer_keeps_initial_feasible_reference(monkeypatch):
    x,_,g,_,_=case();engine=MatchingObjective(x,x,g,g,.1,beta=0.,chunk=7)
    monkeypatch.setattr('scipy.optimize.minimize',lambda *args,**kwargs:
        SimpleNamespace(x=np.full(4,np.nan),success=False,status=9,message='injected failure',nit=3))
    initial=torch.full((4,),.25,dtype=torch.float64)
    p,value,_,diag=polish_fixed_support(engine,[0,2,4,6],initial)
    torch.testing.assert_close(p,initial)
    assert value==engine.evaluate([0,2,4,6],initial)[0] and not diag['scipy_success']


@pytest.mark.parametrize('kwargs',[{'maximum':0},{'maximum':True},{'ftol':float('nan')}])
def test_invalid_polish_budget_is_rejected(kwargs):
    x,_,g,_,_=case();engine=MatchingObjective(x,x,g,g,.1,beta=0.,chunk=7)
    with pytest.raises(ValueError,match='budget'):
        polish_fixed_support(engine,[0,2],torch.tensor([.5,.5]),**kwargs)


@pytest.mark.parametrize('diversity',[False,True])
def test_original_pool_retained_and_polish_generation_is_holdout_and_beta0_independent(diversity):
    from scripts.analyze_local_results import audit_matching_solution
    x,val,g,vg,paired=case();old={**settings(),'matching_exchange_max_steps':0}
    cfg={**old,'matching_polish_max_iter':8,'matching_diverse_starts':diversity}
    _,base,_=generate_fit_only_pool(x,g,paired,old)
    result=serialized(fit_matching_uvarfs(x,val,g,vg,paired,cfg))
    n=result.get('matching_v21_candidate_count',result['matching_base_candidate_count'])
    removed={'validation_valid','validation_error','validation_objective','validation_terms','selection_score'}
    actual=[{k:v for k,v in c.items() if k not in removed} for c in result['candidate_pool'][:n]]
    assert base==actual
    original_hash=result.get('matching_v21_candidate_pool_fit_only_sha256',result['matching_base_candidate_pool_fit_only_sha256'])
    assert original_hash==hashlib.sha256(json.dumps(base,sort_keys=True,allow_nan=False).encode()).hexdigest()
    assert audit_matching_solution(result,cfg)
    bad=copy.deepcopy(result);bad['matching_polish_attempts']=[]
    with pytest.raises(ValueError,match='attempt costs'):audit_matching_solution(bad,cfg)
    changed=serialized(fit_matching_uvarfs(x,val*torch.arange(1,14),g,vg,paired,cfg))
    assert result['candidate_pool_fit_only_sha256']==changed['candidate_pool_fit_only_sha256']
    beta0={**cfg,'beta':0.}
    _,a,_=generate_fit_only_pool(x,g,paired,beta0)
    _,b,_=generate_fit_only_pool(x,g,(torch.full_like(x,float('nan')),torch.empty(0)),beta0)
    assert a==b
    bad=copy.deepcopy(result);bad['matching_base_candidate_pool_fit_only_sha256']='tampered'
    with pytest.raises(ValueError,match='original v21'):audit_matching_solution(bad,cfg)
    n=result['matching_base_candidate_count']
    if n<len(result['candidate_pool']):
        bad=copy.deepcopy(result);bad['candidate_pool'][n]['diagnostics']['parent_candidate_id']=n
        with pytest.raises(ValueError,match='polish'):audit_matching_solution(bad,cfg)


def test_v21_full_matching_control_branches_do_not_drop_original_methods():
    cfg=load_config(Path(__file__).resolve().parents[1]/'configs/default.yaml')
    old=load_config(Path(__file__).resolve().parents[1]/'configs/v21_control.yaml')
    new=copy.deepcopy(old);new['uvarfs'].update(matching_polish_max_iter=4,matching_v21_control=True)
    names=set(expected_method_names(new));assert len(names)==352
    for method in expected_method_names(old):
        if method.startswith(('gram_','raw_input_gram_','local_','raw_input_local_')):
            assert method in names
        else:
            renamed='raw_input_v21_'+method.removeprefix('raw_input_') if method.startswith('raw_input_') else 'v21_'+method
            assert renamed in names
    assert all(b['matching_polish_max_iter']==0 for b in experiment_branches(new) if b['prefix'].endswith('v21_'))
    assert all(b['matching_diverse_starts'] is False for b in experiment_branches(new) if b['prefix'].endswith('v21_'))
    assert len(expected_method_names(old))==262 and old['results_dir'].endswith('v21-normal-matching-distribution')
    for value in [0,'4',True]:
        bad=copy.deepcopy(new);bad['uvarfs']['matching_polish_max_iter']=value
        with pytest.raises(ValueError,match='v21 solver control'):experiment_branches(bad)


def test_polished_full_worker_and_old_control_share_fit_memory_and_strict_audit(tmp_path,monkeypatch):
    import scripts.run_all as runner
    import scripts.analyze_local_results as analyzer
    cfg,_,_=matching_fixture(tmp_path,True)
    cfg['uvarfs'].update(matching_polish_max_iter=4,matching_v21_control=True,matching_diverse_starts=True)
    cfg['paths']['processed_data']=str(tmp_path);cfg['results_dir']=str(tmp_path/'output')
    extractor=CountingExtractor()
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    monkeypatch.setattr(runner,'load_config',lambda _:copy.deepcopy(cfg))
    monkeypatch.setattr(runner,'DINOv3Extractor',lambda *args:extractor)
    monkeypatch.setattr(runner,'discover_bmad_roots',lambda _: {'liver':tmp_path/'liver'})
    monkeypatch.setattr('uvarfs.pipeline_eval.make_index',lambda memory,cfg:CPUIndex(memory))
    monkeypatch.setattr(sys,'argv',['run_all','--dataset','liver','--dataset-worker'])
    runner.main();root=tmp_path/'output';folder=root/'liver'
    assert len(pd.read_csv(folder/'metrics.csv'))==352
    assert not (root/'all_metrics.csv').exists()
    baseline=json.loads((folder/'v21'/'uvarfs_main.json').read_text())
    main=json.loads((folder/'uvarfs_main.json').read_text())
    assert 'matching_polish_strategy' not in baseline and main['matching_base_candidate_count']>0
    assert main['candidate_pool'][:main['matching_v21_candidate_count']]==baseline['candidate_pool']
    first=len(extractor.outputs);runner.main();assert len(extractor.outputs)==first
    monkeypatch.setattr(analyzer,'verify_recorded_code',lambda _:False)
    report=analyzer.audit(root,tmp_path/'audit',True,50)
    assert report['completed_datasets']==['liver']
    (folder/'v21'/'uvarfs_main.json').unlink()
    runner.main();assert len(extractor.outputs)>first


def test_rank_revealing_starts_are_real_coordinates_deterministic_and_beta0_ignores_nuisance():
    x,_,g,_,paired=case()
    a,info=diverse_starts(x,g,paired,0.,4,8)
    b,other=diverse_starts(x,g,(torch.full_like(x,float('nan')),torch.empty(0)),0.,4,8)
    assert info==other
    for (name,left),(same,right) in zip(a,b):
        assert name==same and torch.equal(left,right) and len(left)==len(left.unique())==4
        assert left.min()>=0 and left.max()<x.shape[1]
    zero,diag=pivot_coordinates(torch.zeros(20,13),4)
    assert zero.tolist()==[0,1,2,3] and diag['rank_steps']==0
    for bad in [torch.tensor([[float('nan'),1.]]),torch.ones(2)]:
        with pytest.raises(ValueError,match='diversity'):pivot_coordinates(bad,1)
