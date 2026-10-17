import copy,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import torch,yaml
import scripts.evaluate_ridge_support as w
from scripts.fit_ridge_support import EXPECTED


@pytest.mark.parametrize('bad_anchor',[False,True])
def test_ten_method_worker_fits_before_test_and_preserves_source_cache(tmp_path,monkeypatch,bad_anchor):
    source=tmp_path/'results/source';source.mkdir(parents=True)
    cache=tmp_path/'results/cache';cache.mkdir();out=tmp_path/'results/new';cfgpath=tmp_path/'ridge.yaml'
    cfgpath.write_text(yaml.safe_dump(EXPECTED))
    cfg={'seed':42,'paths':{'processed_data':'data','model_dir':'models'},'model':{'input_size':224,'amp':False}}
    meta={'config':cfg,'git_head':'a'*40,'code_fingerprint':'source','dataset':'brain',
          'data_fingerprint':'data','experiment_version':w.SOURCE_VERSION}
    for name in w.INPUTS:w.save_json(meta if name=='run_metadata.json' else {},source/name)
    pred=pd.DataFrame({'image_path':['a','b'],'label':[0,1],'mask_path':['','m'],
        'main':[.2,.8],'asls_no_variability_uvarfs':[.3,.7]})
    pred.to_csv(source/'image_predictions.csv',index=False)
    names=list(w.ANCHORS.values())+['other'+str(i) for i in range(350)]
    old=pd.DataFrame([{'method':m,'dataset':'brain','experiment_version':w.SOURCE_VERSION,
                      **{k:.7 for k in w.METRICS}} for m in names]);old.to_csv(source/'metrics.csv',index=False)
    memory={'normal_train_indices':[1],'normal_train_paths':['train/good/a'],
        'patch_batches':[{'batch':1,'patch_indices':[0,1]}],'memory_capacity':2,'reservoir_seed':42,
        'shared_sampling_across_methods':True,'memory_rows':{m:2 for m in names}}
    w.save_json(memory,source/'memory_manifest.json')
    torch.save({'normal':torch.ones(2)},cache/'normal-inputs.pt')
    before={n:w.sha256(source/n) for n in w.INPUTS};cachehash=w.sha256(cache/'normal-inputs.pt')
    w.save_json({'dataset':'brain','fit_seed':42,'source_inputs_sha256':before},cache/'run_metadata.json')
    w.save_json({'normal-inputs.pt':{'sha256':cachehash}},cache/'files-sha256.json')
    pd.DataFrame([{'method':'tail'+str(i),'evaluation_version':'normal-tail-evaluation-v1'} for i in range(6)]).to_csv(cache/'metrics.csv',index=False)
    pred.to_csv(cache/'image_predictions.csv',index=False)
    w.save_json({p.name:{'sha256':w.sha256(p)} for p in cache.iterdir() if p.is_file() and p.name!='files-sha256.json'},cache/'files-sha256.json')
    fixed=[False]
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0');monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(torch.cuda,'reset_peak_memory_stats',lambda:None);monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda:0)
    monkeypatch.setattr(w,'code_fingerprint',lambda p:'eval')
    monkeypatch.setattr(w.subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=0))
    monkeypatch.setattr(w.subprocess,'check_output',lambda *a,**k:'head')
    monkeypatch.setattr('scripts.analyze_results.verify_recorded_code',lambda m:True)
    monkeypatch.setattr(w,'completed_result_compatible',lambda *a:True)
    monkeypatch.setattr(w,'discover_bmad_roots',lambda p:{'brain':'root'})
    monkeypatch.setattr(w,'DINOv3Extractor',lambda *a:object())
    monkeypatch.setattr(w,'data_fingerprint',lambda *a:'data')
    monkeypatch.setattr(w,'tail_complete',lambda p:fixed[0] or pytest.fail('old test parsed before new fit'))
    def fit(cachefile,solution,beta,config,output,root,device):
        assert not fixed[0] and cachefile==cache/'normal-inputs.pt';output.mkdir()
        specs={name:{'layers':[1],'kind':'uvarfs','obj':{'active':list(range(256)),'scales':[1/16]*256}} for name in w.METHODS}
        w.save_json(specs,output/'representation-specs.json');fixed[0]=True
        return {'new_fit_seconds':.1}
    monkeypatch.setattr(w,'fit_extension',fit)
    def scan(path,split,dataset):
        assert fixed[0];return [SimpleNamespace(label=0,split=split)]
    monkeypatch.setattr(w,'scan_split',scan)
    def build(e,t,s,c,progress):
        w.save_json({**memory,'memory_rows':{m:2 for m in s}},progress.parent/'memory_manifest.json')
        return {m:np.ones((2,256)) for m in s}
    monkeypatch.setattr(w,'build_memories',build)
    def evaluate(e,t,s,m,c,progress):
        frame=pred[['image_path','label','mask_path']].copy()
        for method in s:frame[method]=pred[w.ANCHORS.get(method,'main')]
        if bad_anchor:frame.loc[0,'v22_anchor']+=.01
        frame.to_csv(progress.parent/'image_predictions.csv',index=False)
        w.save_json({'stage':'complete','methods':10},progress)
        return [{'method':name,**{k:.7 for k in w.METRICS}} for name in s]
    monkeypatch.setattr(w,'evaluate',evaluate)
    monkeypatch.setattr(w,'__file__',str(tmp_path/'scripts/evaluate.py'))
    (tmp_path/'scripts').mkdir();(tmp_path/'scripts/evaluate.py').write_text('fake');(tmp_path/'scripts/fit_ridge_support.py').write_text('fake')
    if bad_anchor:
        with pytest.raises(ValueError,match='anchor predictions'):w.run(source,out,cfgpath,tmp_path,cache_source=cache)
        assert (out/'failure.json').is_file() and not (out/'complete.json').exists()
    else:
        w.run(source,out,cfgpath,tmp_path,cache_source=cache)
        assert w.strict_complete(out)['rows']==10
        with (cache/'normal-inputs.pt').open('ab') as f:f.write(b'x')
        with pytest.raises(ValueError,match='normal cache differs'):w.strict_complete(out)
    assert before=={n:w.sha256(source/n) for n in w.INPUTS}


def test_complete_all_six_report_keeps_previous_tail_and_no_mask_scope(tmp_path,monkeypatch):
    import scripts.analyze_ridge_support_validation as a
    from uvarfs.protocol import expected_method_names
    from uvarfs.utils import load_config
    names=expected_method_names(load_config(Path(__file__).resolve().parents[1]/'configs/default.yaml'))
    batch=tmp_path/'results/batch';batch.mkdir(parents=True)
    w.save_json({'head':'head','code_fingerprint':'eval','worker_sha256':'worker','fitter_sha256':'fit'},batch/'startup.json')
    monkeypatch.setattr(a,'__file__',str(tmp_path/'scripts/analyze.py'))
    monkeypatch.setattr(a,'strict_complete',lambda p:{'fit_seed':int(p.parent.name[4:]),'dataset':p.name})
    monkeypatch.setattr(a,'audit_fitted_extension',lambda p:None)
    tailnames=['nuisance_mean','nuisance_tail','nuisance_tail_uniform','beta_zero_mean','beta_zero_tail','beta_zero_tail_uniform']
    for seed in [42,123,3407]:
        for ds in ['brain','liver','resc','oct2017','xray','camelyon16']:
            p=batch/'results'/f'seed{seed}'/ds
            for d in ['controls','previous_tail','selection']:(p/d).mkdir(parents=True,exist_ok=True)
            metrics={'image_auroc':1.,'image_auprc':1.}
            if ds in {'brain','liver','resc'}:metrics.update(pixel_auroc=.7,pixel_auprc=.6,aupro=.5)
            pred={'image_path':['a','b'],'label':[0,1],'mask_path':['','m']}
            for folder,methods in [(p,w.METHODS),(p/'controls',names),(p/'previous_tail',tailnames)]:
                pd.DataFrame([{'method':m,**metrics,'dataset':ds,'fit_seed':seed} for m in methods]).to_csv(folder/'metrics.csv',index=False)
                pd.DataFrame({**pred,**{m:[.1,.9] for m in methods}}).to_csv(folder/'image_predictions.csv',index=False)
            for rel in ['report.json','nuisance-fit.json','beta_zero-fit.json']:w.save_json({},p/'selection'/rel)
            w.save_json({'evaluation_git_head':'head','evaluation_code_fingerprint':'eval','worker_script_sha256':'worker',
                'fitter_script_sha256':'fit','source_code_fingerprint':'source','source_config':{'seed':seed,'results_dir':'old','same':True}},p/'run_metadata.json')
    out=tmp_path/'reports/analysis';a.analyze(batch,out,draws=50)
    audit=w.read(out/'audit-provenance.json')
    assert audit['new_evaluations']==180 and audit['retained_control_rows']==6336 and audit['previous_tail_rows']==108
    frame=pd.read_csv(out/'paired_differences_by_fit.csv')
    assert frame[w.METRICS].fillna(0).eq(0).all().all()
    assert frame[frame.dataset.eq('xray')].pixel_auprc.isna().all()


def test_real_normal_cache_fitter_and_serialized_extension_audit(tmp_path,monkeypatch):
    import scripts.fit_ridge_support as fitter
    from scripts.analyze_ridge_support_validation import audit_fitted_extension
    from test_v20_local_selection import case
    from test_v21_matching_distribution import settings,serialized
    from uvarfs.matching_uvarfs import fit_matching_uvarfs
    x,val,g,vg,paired=case();source=tmp_path/'results/source';source.mkdir(parents=True)
    manifest={'normal_train_indices':[0],'normal_train_paths':['data/train/good/0.png'],
        'sampled_patches':[],'perturbation_rng':{},'normal_selection_split':{
        'holdout_used_for_gradients':False,'holdout_used_for_selection_reference':False,
        'fit_patch_groups':g.tolist(),'holdout_patch_groups':vg.tolist()}}
    for beta,name in [(.002,'uvarfs_main.json'),(0.,'uvarfs_asls_no_variability.json')]:
        cfg={**settings(),'max_features':x.shape[1],'matching_exchange_max_steps':0,'matching_refit_max_iter':3,'beta':beta}
        sol=serialized(fit_matching_uvarfs(x,val,g,vg,paired,cfg));sol.update(layers=[1],layer_normalization='l2')
        w.save_json(sol,source/name)
    w.save_json(manifest,source/'fit_manifest.json')
    data={'l2':{'patches':{1:x},'heldout':{1:val},'paired_clean':{1:paired[0]},
        'paired_perturbations':{1:paired[1]},'groups':g,'heldout_groups':vg,'manifest':manifest}}
    cache=source/'normal.pt';torch.save(data,cache)
    config=tmp_path/'ridge.yaml';config.write_text(yaml.safe_dump(EXPECTED));out=tmp_path/'results/run/selection'
    monkeypatch.setattr('scripts.run_normal_tail_pilot.subprocess.run',lambda *a,**k:SimpleNamespace(returncode=0))
    result=fitter.run(cache,source/'uvarfs_main.json',source/'uvarfs_asls_no_variability.json',config,out,tmp_path)
    assert set(w.read(out/'representation-specs.json'))==w.METHODS and not result['labels_or_masks_used']
    controls=out.parent/'controls';controls.mkdir()
    import shutil
    for name in ['uvarfs_main.json','uvarfs_asls_no_variability.json']:shutil.copy2(source/name,controls/name)
    audit_fitted_extension(out.parent)
    path=out/'nuisance-fit.json';bad=w.read(path);bad['candidate_pool_fit_only_sha256']='bad';w.save_json(bad,path)
    with pytest.raises(ValueError,match='hash'):audit_fitted_extension(out.parent)
