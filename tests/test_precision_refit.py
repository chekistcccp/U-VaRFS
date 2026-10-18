import copy
import numpy as np
import pytest
import torch
from uvarfs.precision_refit import SETTINGS,weighted_names,methods,refit_fixed_support
from uvarfs.performance_uvarfs import project_bounded_simplex
from uvarfs.matching_uvarfs import MatchingObjective,normal_temperature

class Quadratic:
    def __init__(self,device='cpu'):
        self.q=torch.zeros(1,5,device=device);self.floor_mass=1e-4;self.cap_factor=4.
        self.h=torch.tensor([1.,2.,3.,4.,5.],device=device,dtype=torch.float64)
        self.target=torch.tensor([.05,.1,.15,.3,.4],device=device,dtype=torch.float64)
    def project(self,p):return project_bounded_simplex(p,self.floor_mass,self.cap_factor)
    def evaluate(self,ids,p,gradient=False):
        diff=p-self.target;value=.5*(self.h*diff**2).sum()
        return float(value),self.h*diff if gradient else None,{'quadratic':float(value)}

def settings(**kwargs):return dict(SETTINGS,**kwargs)

def test_fixed_identity_and_yaml():
    from uvarfs.utils import load_config
    from pathlib import Path
    assert load_config(Path(__file__).resolve().parents[1]/'configs/fixed_support_precision.yaml')==SETTINGS
    assert len(weighted_names())==14 and len(methods())==17

def test_independent_convex_oracle_and_monotone_actual_loss():
    engine=Quadratic();p,v,terms,d=refit_fixed_support(engine,range(5),torch.ones(5).double()/5)
    assert np.allclose(p,engine.target,atol=1e-5) and v<1e-10 and d['precision_pass']
    assert d['residual']<=1e-5 and d['termination']=='residual_met'
    assert all(b<a for a,b in zip(d['training_history'],d['training_history'][1:]))
    assert abs(float(p.sum())-1)<1e-10

def test_bound_active_optimum_against_independent_scipy_oracle():
    from scipy.optimize import minimize
    e=Quadratic();e.target=torch.tensor([-.3,.1,.3,.5,.8],dtype=torch.float64)
    p,v,_,d=refit_fixed_support(e,range(5),torch.ones(5).double()/5)
    result=minimize(lambda x:.5*np.dot(e.h.numpy(),(x-e.target.numpy())**2),np.ones(5)/5,
                    jac=lambda x:e.h.numpy()*(x-e.target.numpy()),method='SLSQP',
                    bounds=[(1e-4/5,4/5)]*5,constraints={'type':'eq','fun':lambda x:x.sum()-1,'jac':lambda x:np.ones(5)},
                    options={'ftol':1e-13,'maxiter':200})
    assert result.success and d['precision_pass'] and np.allclose(p,result.x,atol=2e-5)

def test_fixed_budget_failure_is_not_marked_converged():
    e=Quadratic();p,v,_,d=refit_fixed_support(e,range(5),torch.ones(5).double()/5,settings(maximum_evaluations=1))
    assert not d['precision_pass'] and d['termination']=='maximum_evaluations' and d['objective_gradient_calls']==1
    assert torch.allclose(p,torch.ones(5).double()/5)

@pytest.mark.parametrize('change',[{'maximum_iterations':0},{'maximum_evaluations':False},{'residual_tolerance':float('nan')},{'history_size':0}])
def test_bad_budget_rejected(change):
    with pytest.raises(ValueError):refit_fixed_support(Quadratic(),range(5),torch.ones(5).double()/5,settings(**change))

def test_mass_gradient_and_real_matching_descent_without_other_data():
    torch.manual_seed(7);x=torch.randn(18,9,dtype=torch.float64);g=torch.arange(6).repeat_interleave(3)
    e=MatchingObjective(x,x,g,g,normal_temperature(x,g),beta=0.,chunk=7)
    ids=torch.tensor([0,2,3,7,8]);initial=torch.ones(5,dtype=torch.float64)/5
    v,gp,_=e.evaluate(ids,initial,True);direction=torch.tensor([1.,-1.,.5,-.5,0.],dtype=torch.float64)
    h=1e-6
    difference=(e.evaluate(ids,initial+h*direction/5)[0]-e.evaluate(ids,initial-h*direction/5)[0])/(2*h)
    assert difference==pytest.approx(float(gp@direction/5),abs=1e-7,rel=1e-5)
    p,new,_,d=refit_fixed_support(e,ids,initial,settings(maximum_iterations=30))
    assert new<=v and d['iterations']<=30 and len(p)==len(ids)
    assert p.min()>=e.floor_mass/5-1e-12 and p.max()<=e.cap_factor/5+1e-12

def test_stationary_dense_beta_zero_has_no_unnecessary_steps():
    torch.manual_seed(8);x=torch.randn(12,7,dtype=torch.float64);g=torch.arange(4).repeat_interleave(3)
    e=MatchingObjective(x,x,g,g,normal_temperature(x,g),beta=0.)
    p,v,_,d=refit_fixed_support(e,range(7),torch.ones(7).double()/7)
    assert d['precision_pass'] and d['iterations']==0 and v<1e-12

@pytest.mark.skipif(not torch.cuda.is_available(),reason='real CUDA required')
def test_actual_cpu_cuda_feasible_secant_agreement():
    out=[]
    for device in ['cpu','cuda']:
        e=Quadratic(device);out.append(refit_fixed_support(e,range(5),torch.ones(5,device=device).double()/5))
    assert np.allclose(out[0][0].cpu(),out[1][0].cpu(),atol=1e-9) and out[1][3]['precision_pass']

def test_continue_parents_reproduces_original_and_verifies_final_gradient():
    from scripts.fixed_support_precision import continue_parents
    e=Quadratic();initial=np.ones(5)/5;value=e.evaluate(range(5),torch.tensor(initial))[0]
    parent={'active':list(range(5)),'weights':initial.tolist(),'training_objective':value,'diagnostics':{'residual':1.}}
    result=continue_parents(e,{'fixed':copy.deepcopy(parent)})['fixed']
    assert result['active']==parent['active'] and parent['weights']==initial.tolist()
    assert result['diagnostics']['precision_pass'] and result['final_verification_calls']==1
    assert np.allclose(result['final_gradient_p'],e.h.numpy()*(np.array(result['weights'])-e.target.numpy()))
    parent['training_objective']+=.1
    with pytest.raises(ValueError,match='does not reproduce'):continue_parents(e,{'bad':parent})

def test_fixed_entrypoints_and_full_six_three_fit_plan():
    import importlib.util
    from pathlib import Path
    from scripts.run_fixed_support_precision import entrypoints,tasks,WORKER,ANALYZER
    project=Path(__file__).resolve().parents[1]
    assert len(entrypoints(project))==5 and importlib.util.find_spec(WORKER) and importlib.util.find_spec(ANALYZER)
    assert len(tasks())==len(set(tasks()))==18 and tasks()[0]==(42,'liver')

@pytest.mark.parametrize('kind',['duplicate','outside','prefix'])
def test_precision_batch_paths_reject_duplicate_or_escape(tmp_path,kind):
    from scripts.run_fixed_support_precision import validate_paths
    parent=tmp_path/'results/parent';cache=tmp_path/'results/cache';root=tmp_path/'results/remote-experiments/precision-validation-one'
    if kind=='duplicate':root.mkdir(parents=True)
    if kind=='outside':cache=tmp_path/'outside'
    if kind=='prefix':root=root.with_name('other')
    with pytest.raises(ValueError):validate_paths(tmp_path,parent,cache,root)

def test_detector_gate_refuses_partial_counts_or_missing_artifact_coverage(tmp_path):
    from scripts.fixed_support_precision import validate_global_gate
    gate={'evaluation_version':SETTINGS['evaluation_version'],'all_252_pass':False,'blocks':18,'precision_pass_count':251,'normal_fit_sha256':{}}
    with pytest.raises(ValueError,match='not passed'):validate_global_gate(gate,tmp_path)
    gate.update(all_252_pass=True,precision_pass_count=252)
    with pytest.raises(ValueError,match='all 18'):validate_global_gate(gate,tmp_path)

def test_failed_normal_gate_reports_all_252_without_parsing_any_test_csv(tmp_path,monkeypatch):
    import scripts.analyze_fixed_support_precision as a
    from scripts.fixed_support_precision import atomic,sha256
    from uvarfs.mechanism_validation import artifact_digest
    monkeypatch.setattr(a,'__file__',str(tmp_path/'scripts/analyze_fixed_support_precision.py'))
    project=tmp_path;batch=tmp_path/'results/batch';parent=tmp_path/'results/parent'
    parent.mkdir(parents=True);atomic(parent/'state.json',{'phase':'complete'})
    batch.mkdir();atomic(batch/'startup.json',{'parent_relative':parent.relative_to(project).as_posix(),'parent_state_sha256':sha256(parent/'state.json'),'head':'H','code_fingerprint':'F','worker_sha256':'W'})
    count=0
    for seed in [42,123,3407]:
        for dataset in ['brain','liver','resc','oct2017','xray','camelyon16']:
            out=batch/'results'/f'seed{seed}'/dataset;out.mkdir(parents=True)
            previous=parent/'results'/f'seed{seed}'/dataset/'selection';previous.mkdir(parents=True)
            records={};specs={};oldspecs={}
            for name,old in weighted_names().items():
                k=384 if 'dense_' in name else 256;p=torch.ones(k,dtype=torch.float64)/k;g=torch.zeros(k,dtype=torch.float64)
                if count==0:g[0]=.1;g[1]=-.1
                residual=float((p-project_bounded_simplex(p-g,1e-4,4.)).abs().max());count+=1
                records[name]={'active':list(range(k)),'weights':p.tolist(),'final_gradient_p':g.tolist(),'parent_weights_sha256':artifact_digest(p.tolist()),'training_objective':.9,'parent_training_objective':1.,'training_terms':{'variability_term':0.},'parent_residual':1.,'diagnostics':{'support_changed':False,'precision_pass':residual<=1e-5,'residual':residual,'training_history':[1.,.9],'iterations':1,'objective_gradient_calls':2,'termination':'fixture','seconds':1.}}
                specs[name]={'layers':[1],'obj':{'active':list(range(k)),'scales':np.sqrt(p.numpy()).tolist()}};oldspecs[old]=specs[name]
            atomic(previous/'representation-specs.json',oldspecs);atomic(out/'normal-fit.json',records);atomic(out/'representation-specs.json',specs)
            atomic(out/'normal-role-split.json',{});atomic(out/'normal-source-sha256.json',{});atomic(out/'fit-stage.json',{'stage':'normal_fit_complete'})
            atomic(out/'run_metadata.json',{'evaluation_version':SETTINGS['evaluation_version'],'fit_seed':seed,'dataset':dataset,'head':'H','code_fingerprint':'F','worker_sha256':'W'})
            atomic(out/'normal-files-sha256.json',{p.name:{'sha256':sha256(p),'bytes':p.stat().st_size} for p in out.iterdir() if p.is_file()})
            atomic(out/'normal-complete.json',{'evaluation_version':SETTINGS['evaluation_version'],'frozen_representation_sha256':sha256(out/'representation-specs.json'),'precision_pass_count':sum(c['diagnostics']['precision_pass'] for c in records.values())})
    atomic(batch/'global-normal-gate.json',{'all_252_pass':False,'precision_pass_count':251})
    def forbidden(*args,**kwargs):raise AssertionError('test CSV accessed before full normal gate')
    monkeypatch.setattr(a.pd,'read_csv',forbidden)
    out=tmp_path/'reports/final';a.analyze(batch,out)
    proof=a.read(out/'audit-provenance.json')
    assert proof['normal_weighted_fits']==252 and proof['precision_pass_count']==251 and proof['detector_evaluations']==0 and not proof['test_data_parsed']

def test_true_gate_requires_exact_18_identities_and_immutable_bytes(tmp_path):
    from scripts.fixed_support_precision import validate_global_gate,sha256
    records={}
    for seed in [42,123,3407]:
        for dataset in ['brain','liver','resc','oct2017','xray','camelyon16']:
            for name in ['normal-fit.json','representation-specs.json','normal-complete.json','normal-files-sha256.json','normal-role-split.json']:
                path=tmp_path/'results/remote-experiments/precision-validation-fixed/results'/f'seed{seed}'/dataset/name
                path.parent.mkdir(parents=True,exist_ok=True);path.write_text('{}');records[path.relative_to(tmp_path).as_posix()]=sha256(path)
    gate={'evaluation_version':SETTINGS['evaluation_version'],'all_252_pass':True,'blocks':18,'precision_pass_count':252,'normal_fit_sha256':records}
    validate_global_gate(gate,tmp_path)
    (tmp_path/next(iter(records))).write_text('tampered')
    with pytest.raises(ValueError,match='differs'):validate_global_gate(gate,tmp_path)

@pytest.mark.parametrize('broken',[None,'prediction','metric','memory'])
def test_precision_three_anchors_reject_any_protocol_change(broken):
    import pandas as pd
    from scripts.fixed_support_precision import validate_anchors,ANCHORS
    identity={'image_path':['a','b'],'label':[0,1],'mask_path':['','m']}
    new=pd.DataFrame({**identity,**{n:[.1,.9] for n in ANCHORS}});old=pd.DataFrame({**identity,**{n:[.1,.9] for n in ANCHORS.values()}})
    nm=pd.DataFrame([{'method':n,'image_auroc':.7,'image_auprc':.8} for n in ANCHORS]);om=pd.DataFrame([{'method':n,'image_auroc':.7,'image_auprc':.8} for n in ANCHORS.values()])
    mem={'normal_train_indices':[1],'normal_train_paths':['normal'],'patch_batches':[1],'memory_capacity':20000,'reservoir_seed':42,'shared_sampling_across_methods':True,'memory_rows':{'any':20000}};oldmem=copy.deepcopy(mem)
    if broken=='prediction':new.loc[0,'previous_sparse_anchor']+=.01
    if broken=='metric':nm.loc[0,'image_auroc']+=.01
    if broken=='memory':mem['reservoir_seed']=1
    if broken:
        with pytest.raises(ValueError):validate_anchors(new,old,nm,om,mem,oldmem)
    else:assert validate_anchors(new,old,nm,om,mem,oldmem)['predictions_exact']

@pytest.mark.skipif(not torch.cuda.is_available(),reason='real matching CUDA required')
def test_actual_matching_precision_cpu_cuda_loss_and_support_agree():
    torch.manual_seed(9);x=torch.randn(18,9,dtype=torch.float64);g=torch.arange(6).repeat_interleave(3)
    tau=normal_temperature(x,g);answers=[]
    for device in ['cpu','cuda']:
        e=MatchingObjective(x.to(device),x.to(device),g.to(device),g.to(device),tau,beta=0.,chunk=7)
        answers.append(refit_fixed_support(e,[0,1,3,5,8],torch.ones(5,device=device).double()/5,settings(maximum_iterations=20)))
    assert answers[0][1]==pytest.approx(answers[1][1],abs=1e-8,rel=1e-7)
    assert np.allclose(answers[0][0].cpu(),answers[1][0].cpu(),atol=1e-6,rtol=1e-5)
