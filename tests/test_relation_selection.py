import copy
from pathlib import Path
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from uvarfs.relation_selection import (SETTINGS,VERSION,RelationObjective,clean_graph,fit_branch,
    expected_methods,rankings,complementary_support,scientific_digest)
from uvarfs.utils import load_config

def data(device='cpu'):
    generator=torch.Generator().manual_seed(195)
    x=torch.randn(24,12,generator=generator,dtype=torch.float64).to(device)
    g=torch.arange(8,device=device).repeat_interleave(3)
    v=torch.randn(6,12,generator=generator,dtype=torch.float64).to(device)
    vg=torch.arange(20,22,device=device).repeat_interleave(3)
    paired=(x[:12],(x[:12][None]+.03*torch.randn(2,12,12,generator=generator,dtype=torch.float64).to(device)))
    settings=copy.deepcopy(SETTINGS);settings['graph'].update(neighbors=4,minimum_votes=1)
    settings['feature_budget']=6;settings['common_weight_refit']['maximum']=3
    settings['support_search'].update(rounds=1,refit_max_iter=2)
    return x,g,v,vg,paired,settings

def oracle(x,active,p,graph,g,tau,paired,beta):
    z=x[:,active]*p.sqrt();z=F.normalize(z,dim=1);cos=z@z.T
    positive=graph>0;allowed=g[:,None]!=g[None,:];valid=graph.sum(1)>0
    lw=torch.where(positive,graph.clamp_min(1e-300).log(),torch.zeros_like(graph))
    logits=(cos/tau+lw).masked_fill(~allowed,-torch.inf)
    value=(torch.logsumexp(logits[valid],1)-torch.logsumexp(logits[valid].masked_fill(~positive[valid],-torch.inf),1)).mean()
    if beta:
        a,b=paired;aa=a.repeat(len(b),1);bb=b.reshape(-1,x.shape[1])
        scale=(1-F.cosine_similarity(aa,bb)).mean().clamp_min(1e-4)
        sa=F.normalize(aa[:,active]*p.sqrt(),dim=1);sb=F.normalize(bb[:,active]*p.sqrt(),dim=1)
        value+=beta*(1-(sa*sb).sum(1)).mean()/scale
    return value

def test_fixed_config_roster_entrypoints_and_plan():
    from scripts.run_relation_validation import entrypoint_paths,task_plan,validate_run_root
    root=Path(__file__).resolve().parents[1]
    assert load_config(root/'configs/relation_validation.yaml')==SETTINGS
    assert len(expected_methods())==41 and 'same_support_beta_zero' in expected_methods()
    assert len(task_plan())==len(set(task_plan()))==18 and task_plan()[0]==(42,'liver')
    assert all(p.is_file() for p in entrypoint_paths(root).values())

@pytest.mark.parametrize('beta',[0.,.002])
def test_independent_autograd_and_full_coordinate_gradient(beta):
    x,g,_,_,paired,s=data();graph,_=clean_graph(x,x,g,g,s['graph'],42,chunk=5)
    engine=RelationObjective(x,x,g,g,.1,graph,paired if beta else object(),beta,chunk=5)
    ids=torch.tensor([0,2,4,7,9,11]);p=torch.tensor([.1,.2,.15,.18,.17,.2],requires_grad=True,dtype=torch.float64)
    expected=oracle(x,ids,p,graph,g,.1,paired,beta);expected.backward()
    value,grad,terms=engine.evaluate(ids,p.detach(),True)
    assert value==pytest.approx(float(expected.detach()),abs=1e-12)
    assert torch.allclose(grad,p.grad,atol=2e-11,rtol=2e-10)
    _,full,_=engine.evaluate(ids,p.detach(),True,True)
    assert torch.allclose(full[ids],grad,atol=2e-11,rtol=2e-10)
    assert terms['normal_relation_loss']>=0
    # Independent feasible directional finite difference checks omitted coordinates too.
    dense=torch.zeros(x.shape[1],dtype=torch.float64);dense[ids]=p.detach()
    for j in [1,3]:
        direction=torch.zeros_like(dense);direction[j]=1;direction[ids[1]]=-1
        h=1e-7;vp=engine.evaluate(range(len(dense)),dense+h*direction)[0]
        assert (vp-value)/h==pytest.approx(float(full@direction),abs=1e-3,rel=2e-4)

def test_graph_reproducible_clean_cross_image_and_subsets_recorded():
    x,g,_,_,_,s=data();a,info=clean_graph(x,x,g,g,s['graph'],42,chunk=5)
    b,other=clean_graph(x,x,g,g,s['graph'],42,chunk=7)
    assert torch.equal(a,b) and info==other
    assert not (a[g[:,None]==g[None,:]]!=0).any()
    assert info['valid_queries']>0 and info['same_image_edges']==0 and not info['nuisance_used']
    assert len(info['coordinate_subsets'])==3

@pytest.mark.parametrize('bad',['negative','outside','nonfinite','empty','overone'])
def test_invalid_graph_is_rejected(bad):
    x,g,_,_,_,s=data();a,_=clean_graph(x,x,g,g,s['graph'],42)
    if bad=='negative':a[0,4]=-1
    if bad=='outside':a[0,1]=1
    if bad=='nonfinite':a[0,4]=float('nan')
    if bad=='empty':a.zero_()
    if bad=='overone':a[0,4]=2
    with pytest.raises(ValueError):RelationObjective(x,x,g,g,.1,a,beta=0)

def test_actual_refit_preserves_feasibility_and_monotonicity():
    x,g,_,_,pairs,s=data();graph,_=clean_graph(x,x,g,g,s['graph'],42)
    e=RelationObjective(x,x,g,g,.1,graph,pairs,.002,chunk=7)
    ids=list(range(6));p0=torch.full((6,),1/6,dtype=torch.float64)
    p,value,terms,d=e.refit(ids,p0,maximum=8)
    assert abs(float(p.sum())-1)<1e-10 and p.min()>=1e-4/6-1e-12 and p.max()<=4/6+1e-12
    assert value<=e.evaluate(ids,p0)[0]+1e-12
    assert all(b<=a+1e-12 for a,b in zip(d['training_history'],d['training_history'][1:]))
    assert d['loss_gradient_and_value_calls']>0 and d['stop_status'] in {'residual_tolerance','iteration_budget','line_search_stalled'}

def test_common_fit_graph_and_candidates_ignore_selection_until_freeze_and_beta0_does_not_read_pairs():
    x,g,v,vg,_,s=data();s['support_search']['rounds']=0
    a,_=fit_branch(x,v,g,vg,object(),s,42,0.)
    b,_=fit_branch(x,v+3,g,vg,None,s,42,0.)
    assert a['fit_only_sha256']==b['fit_only_sha256']
    assert scientific_digest(a['support_pool'])==scientific_digest(b['support_pool']) and a['graph']==b['graph']
    for item in [*a['learned'],*a['random'].values(),*a['baseline'].values(),a['dense']]:
        assert item['initialization']=='uniform' and item['training_terms']['variability_term']==0
    assert a['selection_used_only_after_full_fit']

def test_simple_selectors_and_complementary_budget_are_clean_and_deterministic():
    x,g,_,_,_,s=data();graph,_=clean_graph(x,x,g,g,s['graph'],42)
    lap,repeat=rankings(x,g,graph)
    # Independent scalar Laplacian-score oracle for each coordinate.
    w=(graph+graph.T)/2;degree=w.sum(1)
    for j in range(x.shape[1]):
        col=x[:,j];center=col-(degree*col).sum()/degree.sum()
        expected=(w*(col[:,None]-col[None,:]).square()).sum()/2/(degree*center.square()).sum()
        assert float(lap[j])==pytest.approx(float(expected),abs=1e-12)
    ids=complementary_support(x,-lap,6)
    assert len(ids)==len(set(ids))==6 and ids==complementary_support(x,-lap,6)
    assert torch.isfinite(repeat).all()

def test_full_fitter_serialized_independent_audit_and_same_support_beta0(tmp_path):
    import json
    from uvarfs.relation_selection import fit_design
    from scripts.evaluate_relation_validation import audit_fit
    gen=torch.Generator().manual_seed(619)
    x=torch.randn(24,384,generator=gen);v=torch.randn(6,384,generator=gen)
    g=torch.arange(8).repeat_interleave(3);vg=torch.arange(20,22).repeat_interleave(3)
    paired=({1:x[:12]},{1:x[:12][None]+.02*torch.randn(2,12,384,generator=gen)})
    s=copy.deepcopy(SETTINGS);s['common_weight_refit']['maximum']=1;s['support_search'].update(rounds=0)
    raw=dict(layers=[1],layer_normalization='l2',kind='raw')
    old={name:copy.deepcopy(raw) for name in ['asls_raw','asls_pca','fixed4_raw','all_raw','last_raw',
        'mechanism_sparse','mechanism_beta_zero','dense_weighted',*[f'randomk_raw_seed{s}' for s in SETTINGS['random_seeds']]]}
    result=fit_design({1:x},{1:v},g,vg,paired,[1],old,s,42)
    assert set(result['specs'])==expected_methods()
    assert result['specs']['same_support_beta_zero']['obj']['active']==result['specs']['relation_sparse']['obj']['active']
    folder=tmp_path/'selection';folder.mkdir()
    (folder/'representation-specs.json').write_text(json.dumps(result['specs']))
    for name,b in result['branches'].items():(folder/(name+'-fit.json')).write_text(json.dumps(b))
    audit_fit(tmp_path)
    corrupt=json.loads((folder/'nuisance-fit.json').read_text());corrupt['learned'][0]['weights'][0]+=.01
    (folder/'nuisance-fit.json').write_text(json.dumps(corrupt))
    with pytest.raises(ValueError,match='freeze'):audit_fit(tmp_path)

@pytest.mark.parametrize('bad',['existing','outside','prefix'])
def test_unsafe_scheduler_root_rejected(tmp_path,bad):
    from scripts.run_relation_validation import validate_run_root
    source=tmp_path/'results/source';out=tmp_path/'results/remote-experiments/relation-validation-fixed'
    if bad=='existing':out.mkdir(parents=True)
    elif bad=='outside':out=tmp_path/'outside/relation-validation-fixed'
    else:out=out.with_name('mechanism-validation-old')
    with pytest.raises(ValueError):validate_run_root(tmp_path,source,out)

@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_actual_cuda_graph_objective_and_refit_match_cpu():
    x,g,_,_,paired,s=data();graph,_=clean_graph(x,x,g,g,s['graph'],42)
    gpu,ginfo=clean_graph(x.cuda(),x.cuda(),g.cuda(),g.cuda(),s['graph'],42)
    assert torch.equal(graph,gpu.cpu())
    a=RelationObjective(x,x,g,g,.1,graph,paired,.002,chunk=7)
    b=RelationObjective(x.cuda(),x.cuda(),g.cuda(),g.cuda(),.1,gpu,(paired[0].cuda(),paired[1].cuda()),.002,chunk=7)
    ids=list(range(6));p=torch.full((6,),1/6,dtype=torch.float64)
    av,ag,_=a.evaluate(ids,p,True);bv,bg,_=b.evaluate(ids,p.cuda(),True)
    assert av==pytest.approx(bv,abs=1e-10) and torch.allclose(ag,bg.cpu(),atol=1e-9,rtol=1e-8)
    aa=a.refit(ids,p,3);bb=b.refit(ids,p.cuda(),3)
    assert aa[1]==pytest.approx(bb[1],abs=1e-8)
