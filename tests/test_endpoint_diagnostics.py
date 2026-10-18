import numpy as np
import pytest
import torch
from uvarfs.matching_uvarfs import MatchingObjective
from uvarfs.endpoint_diagnostics import geometry,independent_slopes,feasible_directions,diagnose_endpoint,STEPS,DIRECTIONS

def engine(device='cpu',beta=.002):
    gen=torch.Generator().manual_seed(53);x=torch.randn(12,7,generator=gen,dtype=torch.float64).to(device);groups=torch.arange(12,device=device)//2
    clean=x[:4];perturb=(clean+.01*torch.randn(4,7,generator=gen,dtype=torch.float64).to(device))[None]
    return MatchingObjective(x,x,groups,groups,.3,(clean,perturb),beta,chunk=4)

@pytest.mark.parametrize('beta',[0.,.002])
def test_directional_derivative_matches_independent_finite_difference(beta):
    e=engine(beta=beta);ids=[0,1,2,4];p=torch.tensor([.15,.25,.3,.3],dtype=torch.float64);d=torch.tensor([.05,-.03,.01,-.03],dtype=torch.float64)
    loss,g,_=e.evaluate(ids,p,True);geo=geometry(e,ids,p);s=independent_slopes(e,p,d,geo,ids)
    finite=(e.evaluate(ids,p+1e-6*d)[0]-e.evaluate(ids,p-1e-6*d)[0])/2e-6
    assert float(g@d)==pytest.approx(s['independent_current_branch_slope'],rel=1e-9,abs=1e-10)
    assert finite==pytest.approx(float(g@d),rel=2e-5,abs=1e-7)
    assert s['observed_exact_tie_directional_slope']==pytest.approx(float(g@d),abs=1e-10)

def test_exact_tie_has_positive_actual_direction_despite_negative_chosen_branch():
    q=torch.tensor([[1.,1.,0.]],dtype=torch.float64);r=torch.tensor([[1.,0.,2.],[0.,1.,2.]],dtype=torch.float64)
    e=MatchingObjective(q,r,torch.tensor([0]),torch.tensor([1,2]),1.,beta=0.,distribution_weight=0.)
    p=torch.tensor([.5,.5],dtype=torch.float64);d=torch.tensor([-.2,.2],dtype=torch.float64);ids=[0,1]
    loss,g,_=e.evaluate(ids,p,True);geo=geometry(e,ids,p);s=independent_slopes(e,p,d,geo,ids)
    assert geo[3]['exact_tie_count'].item()==2 and float(g@d)<0
    finite=(e.evaluate(ids,p+1e-6*d)[0]-loss)/1e-6
    assert finite>0 and s['observed_exact_tie_directional_slope']>0
    assert finite==pytest.approx(s['observed_exact_tie_directional_slope'],rel=1e-4)
    # This is a counterexample to a single chosen-branch residual as a nonsmooth certificate.

def test_fixed_direction_plan_is_feasible_and_parent_unchanged():
    e=engine();p=torch.tensor([.25]*4,dtype=torch.float64);parent=torch.tensor([.1,.2,.3,.4],dtype=torch.float64);old=parent.clone();g=torch.tensor([1.,2.,-2.,0.],dtype=torch.float64)
    directions=feasible_directions(e,p,g,parent);assert tuple(directions)==DIRECTIONS and torch.equal(old,parent)
    for d in directions.values():
        for h in STEPS:
            candidate=p+h*d;assert abs(float(candidate.sum())-1)<1e-12 and candidate.min()>=e.floor_mass/4-1e-12 and candidate.max()<=1+1e-12

def test_read_only_probe_grid_loss_and_complete_branch_arrays():
    e=engine();ids=[0,1,2,4];p=torch.tensor([.15,.25,.3,.3],dtype=torch.float64);saved=p.clone()
    report,arrays=diagnose_endpoint(e,ids,p,p)
    assert torch.equal(p,saved) and not report['optimizer_run'] and not report['source_weights_modified']
    assert report['actual_objective_calls']==len(report['probes'])+2
    assert len(report['probes'])==7*sum(not d['zero_direction'] for d in report['directions'])
    assert len(arrays)==5+2*len(report['probes']) and report['normal_queries']==12
    for row in report['probes']:
        assert row['loss_difference']==pytest.approx(row['actual_loss']-report['loss'])
        assert np.isfinite(row['one_sided_quotient'])

def test_near_ties_not_promoted_and_invalid_endpoint_rejected():
    e=engine();p=torch.tensor([-.1,.3,.4,.4],dtype=torch.float64)
    with pytest.raises(ValueError,match='infeasible'):diagnose_endpoint(e,[0,1,2,4],p,p)

def test_worker_plan_and_source_paths_are_fixed(tmp_path):
    from scripts.run_normal_endpoint_diagnostics import tasks,validate_paths
    assert tasks()==['liver','brain','resc','xray','camelyon16','oct2017']
    with pytest.raises(ValueError):validate_paths(tmp_path,tmp_path/'results/old',tmp_path/'outside')
    with pytest.raises(ValueError):validate_paths(tmp_path,tmp_path/'outside',tmp_path/'results/remote-experiments/endpoint-diagnostics-one')

def test_coordinator_accepts_launcher_wrapper_logs_but_refuses_old_workers(tmp_path):
    from scripts.run_normal_endpoint_diagnostics import prepare_output_dirs
    root=tmp_path/'results/remote-experiments/endpoint-diagnostics-test';root.parent.mkdir(parents=True)
    logs=tmp_path/'logs/remote-experiments'/root.name;logs.mkdir(parents=True);(logs/'wrapper.log').write_text('original launcher')
    actual,reports=prepare_output_dirs(tmp_path,root)
    assert actual==logs and (actual/'jobs').is_dir() and reports.is_dir() and (logs/'wrapper.log').read_text()=='original launcher'
    with pytest.raises(ValueError):prepare_output_dirs(tmp_path,root)

def test_complete_worker_artifacts_and_independent_six_block_audit_without_test_reads(tmp_path,monkeypatch):
    from scripts.normal_endpoint_diagnostics import atomic,sha256,strict_complete
    import scripts.analyze_normal_endpoint_diagnostics as analyzer
    from uvarfs.endpoint_diagnostics import VERSION,METHODS
    monkeypatch.setattr(analyzer,'__file__',str(tmp_path/'scripts/analyze_normal_endpoint_diagnostics.py'))
    monkeypatch.setattr(analyzer.pd,'read_csv',lambda *a,**k:pytest.fail('normal diagnostic must not read test CSV'))
    source=tmp_path/'results/source';source.mkdir(parents=True);atomic(source/'state.json',{'phase':'precision_inconclusive'})
    batch=tmp_path/'results/batch';batch.mkdir();atomic(batch/'startup.json',{'source_relative':'results/source','source_state_sha256':sha256(source/'state.json'),'head':'H','code_fingerprint':'F','worker_sha256':'W','config_sha256':'C'})
    gen=torch.Generator().manual_seed(73);x=torch.randn(6,384,generator=gen,dtype=torch.float64);groups=torch.arange(6)
    results={}
    for method in METHODS:
        e=MatchingObjective(x,x,groups,groups,.4,beta=0.)
        ids=list(range(384 if 'dense_' in method else 256));p=torch.ones(len(ids),dtype=torch.float64)/len(ids)
        r,arrays=diagnose_endpoint(e,ids,p,p);r.update(source_method=method);results[method]=(r,arrays)
    for dataset in ['liver','brain','resc','xray','camelyon16','oct2017']:
        src=source/'results/seed42'/dataset;src.mkdir(parents=True);atomic(src/'normal-fit.json',{m:{'active':r['active'],'weights':r['weights']} for m,(r,a) in results.items()})
        out=batch/'results'/dataset;out.mkdir(parents=True)
        for method,(r,a) in results.items():atomic(out/(method+'.json'),r);np.savez_compressed(out/(method+'.npz'),**a)
        atomic(out/'source-sha256.json',{(src/'normal-fit.json').relative_to(tmp_path).as_posix():{'sha256':sha256(src/'normal-fit.json'),'bytes':(src/'normal-fit.json').stat().st_size}})
        atomic(out/'stage.json',{'stage':'complete'});atomic(out/'run_metadata.json',{'source_seed':42,'dataset':dataset,'head':'H','code_fingerprint':'F','worker_sha256':'W','config_sha256':'C'})
        atomic(out/'files-sha256.json',{p.name:{'sha256':sha256(p),'bytes':p.stat().st_size} for p in out.iterdir() if p.is_file()})
        atomic(out/'complete.json',{'evaluation_version':VERSION,'endpoints':5,'optimizer_run':False,'test_data_used':False});assert strict_complete(out)['endpoints']==5
    report=tmp_path/'reports/audit';analyzer.analyze(batch,report)
    proof=analyzer.read(report/'audit-provenance.json');assert proof['endpoints']==30 and not proof['test_data_used']
    # A source mutation is rejected independently of intact destination checksums.
    (source/'results/seed42/liver/normal-fit.json').write_text('{}')
    with pytest.raises(ValueError,match='source changed'):analyzer.analyze(batch,tmp_path/'reports/reject')

def test_strict_block_refuses_unlisted_and_tampered_artifacts(tmp_path):
    from scripts.normal_endpoint_diagnostics import atomic,strict_complete
    from uvarfs.endpoint_diagnostics import VERSION
    atomic(tmp_path/'complete.json',{'evaluation_version':VERSION,'endpoints':5,'optimizer_run':False,'test_data_used':False});atomic(tmp_path/'run_metadata.json',{'source_seed':42})
    atomic(tmp_path/'files-sha256.json',{'extra':{'sha256':'bad','bytes':0}})
    with pytest.raises(ValueError,match='artifact set'):strict_complete(tmp_path)

@pytest.mark.skipif(not torch.cuda.is_available(),reason='actual CUDA runtime required')
def test_actual_cuda_independent_directional_geometry():
    e=engine('cuda');p=torch.tensor([.15,.25,.3,.3],dtype=torch.float64,device='cuda');ids=[0,1,2,4];d=torch.tensor([.05,-.03,.01,-.03],dtype=torch.float64,device='cuda')
    _,g,_=e.evaluate(ids,p,True);s=independent_slopes(e,p,d,geometry(e,ids,p),ids)
    assert float(g@d)==pytest.approx(s['independent_current_branch_slope'],rel=1e-9,abs=1e-10)
