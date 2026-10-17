import copy,json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
from uvarfs.mechanism_validation import (SETTINGS,partition,common_refit,fit_branch,artifact_digest,
    expected_methods,RANDOM_SEEDS,jsonable,best_supports)
from uvarfs.matching_uvarfs import MatchingObjective,normal_temperature
from scripts.evaluate_mechanism_validation import validate_history_anchor,normal_audit_methods
from scripts.fit_mechanism_validation import audit_cached
from test_v20_local_selection import case
from test_v21_matching_distribution import settings


def test_fixed_config_and_method_roster():
    from uvarfs.utils import load_config
    assert load_config(Path(__file__).resolve().parents[1]/'configs/mechanism_validation.yaml')==SETTINGS
    assert len(expected_methods())==35 and len(normal_audit_methods(dict.fromkeys(expected_methods())))==23

def test_first_liver_is_counted_once_before_remaining_all_six_three_seed_plan():
    from scripts.run_mechanism_validation import task_plan
    tasks=task_plan()
    assert tasks[0]==(42,'liver') and len(tasks)==len(set(tasks))==18
    assert set(tasks)=={(s,d) for s in [42,123,3407] for d in ['brain','liver','resc','oct2017','xray','camelyon16']}
    assert (42,'liver') not in tasks[1:5]+tasks[5:]

def test_real_launcher_batch_prefix_is_accepted(tmp_path):
    from scripts.run_mechanism_validation import validate_run_root
    validate_run_root(tmp_path,tmp_path/'results/source',tmp_path/'results/remote-experiments/mechanism-validation-20261017T142627Z')

@pytest.mark.parametrize('bad',['existing','wrong-prefix','outside-source','outside-output'])
def test_scheduler_refuses_unsafe_or_duplicate_batch_roots(tmp_path,bad):
    from scripts.run_mechanism_validation import validate_run_root
    source=tmp_path/'results/source';root=tmp_path/'results/remote-experiments/mechanism-validation-safe'
    if bad=='existing':root.mkdir(parents=True)
    if bad=='wrong-prefix':root=root.with_name('other-batch')
    if bad=='outside-source':source=tmp_path/'not-results/source'
    if bad=='outside-output':root=tmp_path/'other/mechanism-validation-safe'
    with pytest.raises(ValueError,match='fresh project-local'):validate_run_root(tmp_path,source,root)

def source_fixture():
    g=torch.arange(192).repeat_interleave(16)
    return {'groups':g,'heldout_groups':torch.arange(192,256).repeat_interleave(16),
            'paired_clean':{1:torch.ones(1536,2)},'heldout':object()}

@pytest.mark.parametrize('seed',[42,123,3407])
def test_roles_image_disjoint_and_all_original_nuisance_fit(seed):
    source=source_fixture();a,b,info=partition(source,seed)
    sets=[set(info[k]) for k in ['fit_image_positions','selection_image_positions','audit_image_positions']]
    assert [len(s) for s in sets]==[144,48,64]
    assert not sets[0]&sets[1] and not sets[0]&sets[2] and not sets[1]&sets[2]
    assert set(range(96))<=sets[0] and int(a.sum())==144*16 and int(b.sum())==48*16
    source['heldout']=torch.full((1024,2),float('nan'))
    aa,bb,other=partition(source,seed)
    assert torch.equal(a,aa) and torch.equal(b,bb) and info==other

@pytest.mark.parametrize('change',['overlap','rows','nuisance'])
def test_role_mismatch_refused(change):
    s=source_fixture()
    if change=='overlap':s['heldout_groups']=s['heldout_groups']-192
    if change=='rows':s['groups']=s['groups'][:-1]
    if change=='nuisance':s['paired_clean'][1]=s['paired_clean'][1][:-1]
    with pytest.raises(ValueError):partition(s,42)

def test_generic_frozen_digest_survives_json_integer_keys():
    a={'random':{42:{'active':[1,2],'weights':[.3,.7]},123:{'active':[0,2],'weights':[.6,.4]}}}
    assert artifact_digest(a)==artifact_digest(json.loads(json.dumps(jsonable(a))))

@pytest.mark.parametrize('beta',[0.,.002])
def test_shared_solver_uniform_initialization_constraints_and_actual_loss(beta):
    x,_,g,_,paired=case();engine=MatchingObjective(x,x,g,g,normal_temperature(x,g),paired if beta else None,beta,chunk=7)
    for ids in [[0,2,5,7],list(range(x.shape[1]))]:
        result=common_refit(engine,ids,{'maximum':8,'tol':1e-5})
        p=np.array(result['weights']);value=engine.evaluate(ids,p)[0]
        assert result['initialization']=='uniform' and result['weight_solver']=='original_bounded_simplex_BB_Armijo'
        assert value==pytest.approx(result['training_objective'],abs=1e-12)
        assert value<=result['initial_uniform_objective']+1e-12
        assert abs(p.sum()-1)<1e-10 and p.min()>=1e-4/len(p)-1e-12 and p.max()<=min(1,4/len(p))+1e-12

def tiny_settings():
    s=copy.deepcopy(SETTINGS);s['common_weight_refit']['maximum']=3
    s['support_search'].update(rounds=1,refit_max_iter=2)
    return s

def test_fit_pool_and_random_supports_ignore_selection_values_and_beta0_perturbations():
    x,v,g,vg,_=case();cfg=settings();cfg['matching_refit_max_iter']=2;cfg['matching_exchange_max_steps']=0;cfg['matching_diverse_starts']=False;cfg['beta']=0.
    a=fit_branch(x,v,g,vg,object(),cfg,tiny_settings(),42)
    b=fit_branch(x,v+2,g,vg,None,cfg,tiny_settings(),42)
    assert a['support_pool_fit_only_sha256']==b['support_pool_fit_only_sha256']
    assert a['support_pool']==b['support_pool']
    for branch in [a,b]:
        for seed in RANDOM_SEEDS:
            c=branch['random'][seed];expected=np.sort(np.random.default_rng(seed).choice(x.shape[1],cfg['max_features'],replace=False))
            assert c['active']==expected.tolist() and c['training_terms']['variability_term']==0
        assert branch['dense']['active']==list(range(x.shape[1]))
        clean={k:copy.deepcopy(branch[k]) for k in ['support_pool','learned','random','dense']}
        for c in [*clean['learned'],*clean['random'].values(),clean['dense']]:
            for k in ['selection_objective','selection_terms','selection_score']:c.pop(k)
        assert artifact_digest(clean)==branch['fit_only_artifacts_sha256']

def test_final_support_starts_use_training_only_not_selection_scores():
    pool=[dict(candidate_id=i,active=a,training_objective=v,selection_score=-v) for i,a,v in [(0,[0,1],5.),(1,[0,1],2.),(2,[1,2],3.),(3,[0,2],9.)]]
    assert [c['candidate_id'] for c in best_supports(pool,2)]==[1,2]

@pytest.mark.parametrize('broken',[False,True])
def test_pursuit_anchors_checked_independently_of_v22_column_order(broken):
    identity={'image_path':['a','b'],'label':[0,1],'mask_path':['','m']}
    fresh=pd.DataFrame({**identity,'pursuit_anchor':[.2,.8],'pursuit_beta_zero_anchor':[.3,.7],
                        'v22_anchor':[.1,.9],'v22_beta_zero_anchor':[.4,.6]})
    old=pd.DataFrame({**identity,'pursuit_matching':[.2,.8],'pursuit_beta_zero':[.3,.7]})
    if broken:fresh.loc[0,'pursuit_anchor']+=.01
    metrics=['image_auroc','image_auprc','pixel_auroc','pixel_auprc','aupro']
    f=pd.DataFrame([{'method':m,**{k:.7 for k in metrics}} for m in ['pursuit_anchor','pursuit_beta_zero_anchor']])
    o=pd.DataFrame([{'method':m,**{k:.7 for k in metrics}} for m in ['pursuit_matching','pursuit_beta_zero']])
    mem={'normal_train_indices':[1],'normal_train_paths':['normal'],'patch_batches':[1],
         'memory_capacity':2,'reservoir_seed':42,'shared_sampling_across_methods':True,'memory_rows':{'any':2}}
    if broken:
        with pytest.raises(ValueError,match='anchor predictions'):validate_history_anchor(fresh,old,f,o,mem,mem)
    else:assert len(validate_history_anchor(fresh,old,f,o,mem,mem))==2

def test_cached_outer_audit_is_post_freeze_read_only_and_uses_dense_teacher():
    gen=torch.Generator().manual_seed(2);training={1:torch.randn(12,4,generator=gen)}
    source={'heldout':{1:torch.randn(8,4,generator=gen)},'heldout_groups':torch.arange(4).repeat_interleave(2)}
    specs={'asls_raw':dict(layers=[1],layer_normalization='l2',kind='raw'),
           'mechanism_sparse':dict(layers=[1],layer_normalization='l2',kind='uvarfs',obj={'active':[0,1],'scales':[.6,.8]})}
    before=copy.deepcopy(specs);a=audit_cached(source,training,specs,[1],.1,'cpu')
    assert specs==before and not a['used_for_selection'] and a['audit_query_rows']==8 and a['fit_reference_rows']==12
    raw=next(c for c in a['records'] if c['method']=='asls_raw')
    assert abs(raw['normal_neighbor_kl'])<1e-12 and raw['normal_patch_score_mse']<1e-12

@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_common_solver_actual_cuda_matches_cpu_objective():
    x,_,g,_,_=case();tau=normal_temperature(x,g);ids=[0,2,5,7]
    cpu=MatchingObjective(x,x,g,g,tau,beta=0.,chunk=7);gpu=MatchingObjective(x.cuda(),x.cuda(),g.cuda(),g.cuda(),tau,beta=0.,chunk=7)
    a=common_refit(cpu,ids,{'maximum':4,'tol':1e-5});b=common_refit(gpu,ids,{'maximum':4,'tol':1e-5})
    assert a['training_objective']==pytest.approx(b['training_objective'],rel=1e-8,abs=1e-9)


@pytest.mark.parametrize('leak',[False,True])
def test_full_image_normal_audit_excludes_every_query_from_memory(tmp_path,monkeypatch,leak):
    from types import SimpleNamespace
    import scripts.evaluate_mechanism_validation as w
    samples=[SimpleNamespace(image=Path('normal'+str(i)),label=0,split='train') for i in range(70)]
    paths=[str(s.image) for s in samples[:64]];calls=[]
    raw=dict(layers=[1],layer_normalization='l2',kind='raw')
    specs={'asls_raw':raw,'mechanism_sparse':dict(layers=[1],layer_normalization='l2',kind='selected_raw',obj={'active':[0,1]})}
    cfg={'seed':42,'eval':{'topk_fraction':.01}}
    def build(extractor,reference,ss,cfg,progress):
        assert len(reference)==6 and not set(paths)&{str(s.image) for s in reference}
        used=[str(s.image) for s in reference]+([paths[0]] if leak else [])
        w.atomic(progress.parent/'memory_manifest.json',dict(normal_train_paths=used,memory_rows={m:2 for m in ss}))
        calls.append('memory');return {m:np.ones((2,3)) for m in ss}
    monkeypatch.setattr(w,'build_memories',build)
    monkeypatch.setattr(w,'loader',lambda *a,**k:[{'image':torch.ones(8,1)} for _ in range(8)])
    class Index:
        def score_tensor(self,q):return 1-q[:,0]
    monkeypatch.setattr(w,'make_index',lambda *a:Index())
    def extract(image):calls.append('extract');return {1:torch.arange(1,1+len(image)*4*3).float().reshape(len(image),4,3)}
    if leak:
        with pytest.raises(ValueError,match='audit image present'):w.full_normal_audit(extract,samples,specs,cfg,paths,tmp_path/'audit')
        assert calls==['memory']
    else:
        w.full_normal_audit(extract,samples,specs,cfg,paths,tmp_path/'audit')
        frame=pd.read_csv(tmp_path/'audit/image_scores.csv')
        assert len(frame)==64 and frame.image_path.tolist()==paths
        protocol=w.read(tmp_path/'audit/protocol.json')
        assert protocol['full_grid_top1_percent'] and not protocol['used_for_selection']


def test_outer_cache_fitter_freezes_files_before_first_audit_feature_access(tmp_path,monkeypatch):
    import yaml
    import scripts.fit_mechanism_validation as fitter
    source=source_fixture();source['patches']={1:torch.ones(3072,2)};source['paired_clean']={1:torch.ones(1536,2)}
    source['paired_perturbations']={1:torch.ones(4,1536,2)}
    source['heldout']=object()  # Fitting must not index, normalize or inspect it.
    source['manifest']={'normal_selection_split':{'fit_patch_groups':source['groups'].tolist(),'holdout_patch_groups':source['heldout_groups'].tolist()}}
    cache=tmp_path/'results/cache.pt';cache.parent.mkdir();cache.write_bytes(b'immutable')
    manifest=tmp_path/'results/fit_manifest.json';manifest.write_text('{}')
    config=tmp_path/'fixed.yaml';config.write_text(yaml.safe_dump(SETTINGS))
    out=tmp_path/'results/new';fitted=[]
    monkeypatch.setattr(fitter.torch,'load',lambda *a,**k:{'l2':source})
    monkeypatch.setattr(fitter,'validate_normal_manifest',lambda *a:None)
    def fit(training,selection,g,sg,paired,cfg,settings,seed):
        assert len(g)==144*16 and len(sg)==48*16 and len(paired[0][1])==96*16
        assert not set(g.tolist())&set(sg.tolist());fitted.append(True)
        specs={m:dict(layers=[1],layer_normalization='l2',kind='raw') for m in expected_methods()-{'v22_anchor','v22_beta_zero_anchor','pursuit_anchor','pursuit_beta_zero_anchor'}}
        return dict(specs=specs,asls={'selected_layers':[1]},branches={'nuisance':{'temperature_fit_only':.1},'beta_zero':{}},new_fit_seconds=1.)
    monkeypatch.setattr(fitter,'fit_design',fit)
    def audit(*args):
        assert fitted and (tmp_path/'results/new.partial/representation-specs.json').is_file()
        return {'used_for_selection':False}
    monkeypatch.setattr(fitter,'audit_cached',audit)
    result=fitter.run(cache,manifest,{'seed':42},config,out,tmp_path,'cpu')
    assert not result['labels_or_masks_used'] and (out/'normal-cache-audit.json').is_file()
