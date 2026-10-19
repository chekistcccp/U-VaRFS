import copy,json
from pathlib import Path
import numpy as np
import pytest
import torch
from uvarfs.relation_precision import (VERSION,SETTINGS,weighted_names,methods,
    frozen_parent,frozen_graph,continue_parents,audit_record)
from uvarfs.relation_selection import RelationObjective,clean_graph,SETTINGS as GRAPH_SETTINGS
from uvarfs.mechanism_validation import artifact_digest
from uvarfs.performance_uvarfs import project_bounded_simplex

def test_fixed_design_identity_and_all_controls():
    from uvarfs.utils import load_config
    from scripts.run_relation_precision import tasks,entrypoints
    root=Path(__file__).resolve().parents[1]
    assert load_config(root/'configs/relation_precision.yaml')==SETTINGS
    assert len(weighted_names())==19 and len(methods())==22
    assert len(tasks())==len(set(tasks()))==18 and tasks()[0]==(42,'liver')
    assert len(entrypoints(root))==7
    assert 'precision_same_support_beta_zero' in weighted_names()
    for stem in ['laplacian','repeatability']:
        assert all('precision_'+stem+'_'+suffix in weighted_names() for suffix in ['weighted','beta_zero'])

def engine_fixture(device='cpu',beta=0.):
    torch.manual_seed(23)
    x=torch.randn(16,11,dtype=torch.float64).to(device)
    groups=torch.arange(4,device=device).repeat_interleave(4)
    graph,_=clean_graph(x,x,groups,groups,GRAPH_SETTINGS['graph'],42)
    pairs=(x[:4],torch.stack([x[:4]+.01])) if beta else None
    return RelationObjective(x,x,groups,groups,.1,graph,pairs,beta,chunk=7)

@pytest.mark.parametrize('beta',[0.,.002])
def test_actual_relation_continuation_preserves_support_and_original_loss(beta):
    e=engine_fixture(beta=beta);ids=[0,2,3,5,7,9];p=torch.ones(6,dtype=torch.float64)/6
    v,g,_=e.evaluate(ids,p,True)
    parent=dict(active=ids,weights=p.tolist(),training_objective=v,diagnostics={'residual':float((p-e.project(p-g)).abs().max())})
    original=copy.deepcopy(parent)
    cfg=dict(SETTINGS,maximum_iterations=40,maximum_evaluations=300)
    c=continue_parents(e,{'fixed':parent},cfg,log=None)['fixed']
    assert parent==original and c['active']==ids and c['training_objective']<=v
    q=torch.tensor(c['weights'],dtype=torch.float64);_,gg,_=e.evaluate(ids,q,True)
    assert np.allclose(c['final_gradient_p'],gg,atol=1e-12)
    assert c['diagnostics']['objective_gradient_calls']<=300
    assert c['diagnostics']['total_actual_calls']==c['diagnostics']['objective_gradient_calls']+3
    parent['training_objective']+=.01
    with pytest.raises(ValueError,match='does not reproduce'):continue_parents(e,{'bad':parent},cfg,log=None)

def test_mass_change_of_variables_on_smooth_relation_matches_finite_difference():
    e=engine_fixture();ids=[0,2,3,5,7,9];k=len(ids);p=torch.ones(k,dtype=torch.float64)/k
    _,g,_=e.evaluate(ids,p,True);d=torch.tensor([1.,-1.,.5,-.5,.25,-.25],dtype=torch.float64)
    h=1e-6
    fd=(e.evaluate(ids,p+h*d/k)[0]-e.evaluate(ids,p-h*d/k)[0])/(2*h)
    assert fd==pytest.approx(float(g@d/k),rel=1e-5,abs=1e-8)

def test_graph_reproduction_both_betas_and_tamper_rejection():
    e=engine_fixture();_,info=clean_graph(e.q,e.r,e.qg,e.rg,GRAPH_SETTINGS['graph'],42)
    branch=dict(graph_weights_sha256=artifact_digest(e.graph),graph=info,temperature_fit_only=.1)
    branches={b:copy.deepcopy(branch) for b in ['nuisance','beta_zero']}
    graph,record=frozen_graph(e.q,e.qg,branches,42)
    assert torch.equal(graph,e.graph) and record==info
    branches['beta_zero']['graph_weights_sha256']='bad'
    with pytest.raises(ValueError,match='does not reproduce'):frozen_graph(e.q,e.qg,branches,42)

def test_frozen_parent_resolves_every_kind_without_reselection():
    b=dict(learned=[{'id':'lose'},{'id':'win'}],winner_index=1,random={str(s):{'id':s} for s in [42,123,3407,2026,2027]},
           baseline={'laplacian':{'id':'lap'},'repeatability':{'id':'rep'}},dense={'id':'dense'})
    same={'id':'same'}
    for old in weighted_names().values():
        c=frozen_parent(b,old,same)
        assert c['id']!='lose'
    assert frozen_parent(b,'relation_sparse')['id']=='win'
    assert frozen_parent(b,'same_support_beta_zero',same)==same
    with pytest.raises(ValueError):frozen_parent(b,'same_support_beta_zero')

def record_fixture(k=256,fail=False):
    p=torch.ones(k,dtype=torch.float64)/k;g=torch.zeros(k,dtype=torch.float64)
    if fail:g[0]=1.;g[1]=-1.
    residual=float((p-project_bounded_simplex(p-g,1e-4,4.)).abs().max())
    return dict(active=list(range(k)),weights=p.tolist(),training_objective=1.,parent_training_objective=1.,parent_residual=.1,
        parent_weights_sha256=artifact_digest(p.tolist()),uniform_training_objective=1.,
        training_terms={'variability_term':0.},final_gradient_p=g.tolist(),
        diagnostics=dict(support_changed=False,precision_pass=residual<=1e-5,residual=residual,
            training_history=[1.],residual_history=[residual],iterations=0,objective_gradient_calls=1,
            termination='maximum_evaluations' if fail else 'residual_met',seconds=1.,
            total_actual_calls=4,certificate='smooth_fixed_graph_fixed_support_projected_residual_not_global'))

@pytest.mark.parametrize('change',['residual','cost','history','support','nuisance','budget','nonfinite'])
def test_independent_audit_rejects_false_scientific_records(change):
    c=record_fixture();name='precision_beta_zero'
    if change=='residual':c['diagnostics']['residual']=1.
    if change=='cost':c['diagnostics']['total_actual_calls']=0
    if change=='history':c['diagnostics']['training_history']=[1.,1.]
    if change=='support':c['active'][1]=0
    if change=='nuisance':c['training_terms']['variability_term']=.1
    if change=='budget':c['diagnostics']['objective_gradient_calls']=12001
    if change=='nonfinite':c['final_gradient_p'][0]=float('nan')
    with pytest.raises(ValueError):audit_record(name,c)

def materialize_normal(root,parent,seed,dataset,fail=False):
    from scripts.relation_precision import atomic,sha256
    block=root/'results'/f'seed{seed}'/dataset;block.mkdir(parents=True)
    old=parent/'results'/f'seed{seed}'/dataset/'selection';old.mkdir(parents=True)
    fits={n:record_fixture(384 if 'dense_' in n else 256,fail and n=='precision_sparse') for n in weighted_names()}
    specs={n:dict(layers=[1],obj={'active':c['active'],'scales':np.sqrt(c['weights']).tolist()}) for n,c in fits.items()}
    specs['asls_raw']=dict(layers=[1],kind='raw')
    oldspec={weighted_names()[n]:s for n,s in specs.items() if n!='asls_raw'}
    p=record_fixture();branch=dict(learned=[p],winner_index=0,random={str(s):record_fixture() for s in [42,123,3407,2026,2027]},
        baseline={n:record_fixture() for n in ['laplacian','repeatability']},dense=record_fixture(384))
    for c in [p,*branch['random'].values(),*branch['baseline'].values(),branch['dense']]:c['diagnostics']['residual']=.1
    for b in ['nuisance','beta_zero']:atomic(old/(b+'-fit.json'),branch)
    atomic(old/'same_support_beta_zero-fit.json',p);atomic(old/'representation-specs.json',oldspec)
    # Child parent records match stored parent losses and residuals exactly.
    for n,c in fits.items():c['parent_residual']=.1
    atomic(block/'normal-fit.json',fits);atomic(block/'representation-specs.json',specs)
    atomic(block/'normal-role-split.json',{});atomic(block/'normal-source-sha256.json',{})
    atomic(block/'fit-stage.json',dict(stage='normal_fit_complete'))
    atomic(block/'graph-reproduction.json',dict(both_parent_beta_graphs_exact=True,graph={'nuisance_used':False,'same_image_edges':0}))
    atomic(block/'run_metadata.json',dict(evaluation_version=VERSION,fit_seed=seed,dataset=dataset,head='H',code_fingerprint='F',worker_sha256='W',normal_fit_seconds=1.))
    atomic(block/'normal-files-sha256.json',{p.name:{'sha256':sha256(p),'bytes':p.stat().st_size} for p in block.iterdir() if p.is_file()})
    atomic(block/'normal-complete.json',dict(evaluation_version=VERSION,precision_pass_count=sum(c['diagnostics']['precision_pass'] for c in fits.values()),frozen_representation_sha256=sha256(block/'representation-specs.json')))
    return block

def test_all_342_normal_audit_stops_without_reading_test_tables(tmp_path,monkeypatch):
    import scripts.analyze_relation_precision as a
    from scripts.relation_precision import atomic,sha256
    monkeypatch.setattr(a,'__file__',str(tmp_path/'scripts/analyze_relation_precision.py'))
    batch=tmp_path/'results/remote-experiments/relation-precision-fixed';parent=tmp_path/'results/parent';parent.mkdir(parents=True);batch.mkdir(parents=True)
    atomic(parent/'state.json',dict(phase='complete'))
    atomic(batch/'startup.json',dict(parent_relative='results/parent',parent_state_sha256=sha256(parent/'state.json'),head='H',code_fingerprint='F',worker_sha256='W'))
    from scripts.run_relation_precision import tasks
    for i,(seed,dataset) in enumerate(tasks()):materialize_normal(batch,parent,seed,dataset,fail=i==0)
    atomic(batch/'global-normal-gate.json',dict(all_342_pass=False,precision_pass_count=341))
    def forbidden(*args,**kwargs):raise AssertionError('test table parsed before global precision gate')
    monkeypatch.setattr(a.pd,'read_csv',forbidden)
    out=tmp_path/'reports/final';a.analyze(batch,out)
    proof=a.read(out/'audit-provenance.json')
    assert proof['normal_weighted_fits']==342 and proof['precision_pass_count']==341
    assert not proof['test_data_parsed'] and proof['detector_evaluations']==0

def test_global_gate_refuses_count_inflation_and_modified_files(tmp_path):
    from scripts.relation_precision import validate_global_gate,atomic,sha256
    from scripts.run_relation_precision import global_gate,tasks
    batch=tmp_path/'results/remote-experiments/relation-precision-fixed';parent=tmp_path/'results/parent'
    for seed,dataset in tasks():materialize_normal(batch,parent,seed,dataset,fail=False)
    gate=global_gate(batch);assert gate['precision_pass_count']==342
    validate_global_gate(gate,tmp_path)
    gate['precision_pass_count']=341
    with pytest.raises(ValueError,match='not passed'):validate_global_gate(gate,tmp_path)
    gate['precision_pass_count']=342
    (tmp_path/next(iter(gate['normal_fit_sha256']))).write_text('{}')
    with pytest.raises(ValueError,match='differs'):validate_global_gate(gate,tmp_path)

def test_conditional_all_six_analysis_retains_parent_and_all_new_controls(tmp_path,monkeypatch):
    import pandas as pd
    import scripts.analyze_relation_precision as a
    from scripts.relation_precision import atomic,sha256,METRICS
    from scripts.run_relation_precision import tasks,global_gate
    from uvarfs.relation_selection import expected_methods
    monkeypatch.setattr(a,'__file__',str(tmp_path/'scripts/analyze_relation_precision.py'))
    batch=tmp_path/'results/remote-experiments/relation-precision-fixed';parent=tmp_path/'results/parent'
    parent.mkdir(parents=True);atomic(parent/'state.json',dict(phase='complete'))
    for seed,dataset in tasks():
        block=materialize_normal(batch,parent,seed,dataset)
        previous=parent/'results'/f'seed{seed}'/dataset
        for base,names in [(block,methods()),(previous,expected_methods())]:
            pd.DataFrame([dict(method=n,dataset=dataset,fit_seed=seed,**{m:1. if m in METRICS[:2] or dataset in {'brain','liver','resc'} else np.nan for m in METRICS}) for n in names]).to_csv(base/'metrics.csv',index=False)
            pd.DataFrame(dict(label=[0,1,0,1],**{n:[.1,.9,.2,.8] for n in names})).to_csv(base/'image_predictions.csv',index=False)
    atomic(batch/'startup.json',dict(parent_relative='results/parent',parent_state_sha256=sha256(parent/'state.json'),head='H',code_fingerprint='F',worker_sha256='W'))
    atomic(batch/'global-normal-gate.json',global_gate(batch))
    monkeypatch.setattr(a,'strict_evaluation',lambda p:None) # Inference is covered by separate anchor/protocol tests.
    out=tmp_path/'reports/complete';a.analyze(batch,out)
    proof=a.read(out/'audit-provenance.json')
    assert proof['test_data_parsed'] and proof['detector_evaluations']==396
    assert len(pd.read_csv(out/'previous_relation_738_metrics.csv'))==738
    delta=pd.read_csv(out/'paired_differences.csv')
    assert 'precision_same_support_beta_zero' in set(delta.comparison)
    assert 'precision_laplacian_weighted' in set(delta.comparison)
    assert 'parent_same_support_beta_zero' in set(delta.comparison)
    assert len(pd.read_csv(out/'five_random_within_fit_mean_sd.csv'))==180

@pytest.mark.parametrize('bad',[None,'prediction','metric','memory'])
def test_three_relation_anchors_validate_exact_parent_identity(bad):
    import pandas as pd
    from scripts.relation_precision import validate_anchors,ANCHORS
    identity={'image_path':['a','b'],'label':[0,1],'mask_path':['','m']}
    a=pd.DataFrame({**identity,**{n:[.1,.9] for n in ANCHORS}})
    b=pd.DataFrame({**identity,**{n:[.1,.9] for n in ANCHORS.values()}})
    ma=pd.DataFrame([{'method':n,'image_auroc':.8,'image_auprc':.9} for n in ANCHORS])
    mb=pd.DataFrame([{'method':n,'image_auroc':.8,'image_auprc':.9} for n in ANCHORS.values()])
    mem={'normal_train_indices':[1],'normal_train_paths':['normal'],'patch_batches':[1],'memory_capacity':20000,'reservoir_seed':42,'shared_sampling_across_methods':True,'memory_rows':{'any':20000}}
    oldmem=copy.deepcopy(mem)
    if bad=='prediction':a.loc[0,'previous_sparse_anchor']+=.01
    if bad=='metric':ma.loc[0,'image_auroc']+=.01
    if bad=='memory':mem['reservoir_seed']=0
    if bad:
        with pytest.raises(ValueError):validate_anchors(a,b,ma,mb,mem,oldmem)
    else:assert validate_anchors(a,b,ma,mb,mem,oldmem)['predictions_exact']

@pytest.mark.parametrize('bad',['outside','duplicate','prefix'])
def test_new_batch_paths_fail_closed(tmp_path,bad):
    from scripts.run_relation_precision import validate_paths
    parent=tmp_path/'results/parent';cache=tmp_path/'results/cache';root=tmp_path/'results/remote-experiments/relation-precision-fixed'
    if bad=='outside':cache=tmp_path/'outside'
    if bad=='duplicate':root.mkdir(parents=True)
    if bad=='prefix':root=root.with_name('precision-validation-old')
    with pytest.raises(ValueError):validate_paths(tmp_path,parent,cache,root)

@pytest.mark.skipif(not torch.cuda.is_available(),reason='real CUDA required')
def test_actual_relation_precision_cpu_cuda_agree():
    answers=[]
    for device in ['cpu','cuda']:
        e=engine_fixture(device,beta=.002);ids=[0,2,3,5,7,9];p=torch.ones(6,device=device,dtype=torch.float64)/6
        v,g,_=e.evaluate(ids,p,True);parent=dict(active=ids,weights=p.cpu().tolist(),training_objective=v,diagnostics={'residual':float((p-e.project(p-g)).abs().max())})
        answers.append(continue_parents(e,{'small':parent},dict(SETTINGS,maximum_iterations=20),log=None)['small'])
    assert answers[0]['active']==answers[1]['active']
    assert answers[0]['training_objective']==pytest.approx(answers[1]['training_objective'],rel=1e-6,abs=1e-8)
    assert np.allclose(answers[0]['weights'],answers[1]['weights'],atol=1e-6,rtol=1e-5)
