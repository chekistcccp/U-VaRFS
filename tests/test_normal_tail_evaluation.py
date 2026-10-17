import copy,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import torch
import yaml
import scripts.evaluate_normal_tail as w


def reference_case():
    predictions=pd.DataFrame({'image_path':['a','b'],'label':[0,1],'mask_path':['','m'],
        'main':[.2,.8],'asls_no_variability_uvarfs':[.3,.7]})
    new=predictions[['image_path','label','mask_path']].copy()
    for fresh,old in w.ANCHORS.items():new[fresh]=predictions[old]
    old=pd.DataFrame([{'method':name,**{m:.7 for m in w.METRICS}} for name in w.ANCHORS.values()])
    fresh=old.copy();fresh['method']=list(w.ANCHORS)
    memory={'normal_train_indices':[1],'normal_train_paths':['train/good/a'],
        'patch_batches':[{'batch':1,'patch_indices':[0,1]}],'memory_capacity':2,'reservoir_seed':42,
        'shared_sampling_across_methods':True,'memory_rows':{'main':2,'asls_no_variability_uvarfs':2}}
    newmemory={**memory,'memory_rows':{m:2 for m in w.METHODS}}
    return new,predictions,fresh,old,newmemory,memory


def test_exact_anchors_and_shared_sampling_guard_reused_controls():
    args=reference_case();result=w.validate_reference(*args)
    assert len(result)==2 and all(r['predictions_exact'] for r in result)
    for index,mutation in [(0,lambda a:a.__setitem__('nuisance_mean',[.2,.800001])),
                           (4,lambda a:a.__setitem__('reservoir_seed',123)),
                           (2,lambda a:a.__setitem__('pixel_auprc',[.6,.7]))]:
        bad=copy.deepcopy(args);mutation(bad[index])
        with pytest.raises(ValueError):w.validate_reference(*bad)
    bad=copy.deepcopy(args);bad[0].loc[0,'image_path']='different'
    with pytest.raises(ValueError,match='roster'):w.validate_reference(*bad)


@pytest.mark.parametrize('bad_anchor',[False,True])
def test_worker_keeps_sources_and_selection_before_test_scan(tmp_path,monkeypatch,bad_anchor):
    root=tmp_path;source=root/'results/source';source.mkdir(parents=True)
    out=root/'results/new';config=root/'selector.yaml'
    config.write_text(yaml.safe_dump({'pilot_version':'normal-tail-selection-pilot-v1','feature_budget':256,
        'selector':dict(top_fraction=.01,tail_fraction=.25,mean_se_multiplier=1.,chunk=256)}))
    cfg={'seed':42,'paths':{'processed_data':'unused','model_dir':'unused'},
         'model':{'input_size':448,'amp':'bf16'}}
    meta={'config':cfg,'dataset':'brain','experiment_version':w.SOURCE_VERSION,
          'code_fingerprint':'original','git_head':'old','data_fingerprint':'data'}
    _,pred,_,old,_,memory=reference_case()
    metrics=old.to_dict('records')+[{'method':f'control{i}',**{m:.7 for m in w.METRICS}} for i in range(350)]
    frame=pd.DataFrame(metrics);frame['dataset']='brain';frame['experiment_version']=w.SOURCE_VERSION
    frame.to_csv(source/'metrics.csv',index=False);pred.to_csv(source/'image_predictions.csv',index=False)
    for name,value in [('run_metadata.json',meta),('memory_manifest.json',memory),('eval_progress.json',{}),
                       ('fit_manifest.json',{}),('uvarfs_main.json',{}),('uvarfs_asls_no_variability.json',{})]:
        w.save_json(value,source/name)
    before={name:w.sha256(source/name) for name in w.INPUTS}
    monkeypatch.setattr(w.subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=0))
    monkeypatch.setattr(w.subprocess,'check_output',lambda *a,**k:'new-head')
    monkeypatch.setattr(w,'completed_result_compatible',lambda *a:True)
    monkeypatch.setattr(w,'code_fingerprint',lambda *a:'evaluation-code')
    import scripts.analyze_results
    monkeypatch.setattr(scripts.analyze_results,'verify_recorded_code',lambda *a:True)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
    for name,value in [('is_available',lambda:True),('reset_peak_memory_stats',lambda:None),
                       ('empty_cache',lambda:None),('max_memory_allocated',lambda:0)]:
        monkeypatch.setattr(w.torch.cuda,name,value)
    monkeypatch.setattr(w,'set_seed',lambda seed:None)
    monkeypatch.setattr(w,'DINOv3Extractor',lambda *a:object())
    monkeypatch.setattr(w,'discover_bmad_roots',lambda *a:{'brain':Path('unused')})
    fixed=[False]
    def scan(path,split,dataset):
        if split=='test':assert fixed[0], 'test opened before frozen normal selection'
        return [SimpleNamespace(label=0,split=split)]
    monkeypatch.setattr(w,'scan_split',scan)
    def normal(extractor,train,settings):
        assert all(s.label==0 and s.split=='train' for s in train)
        return {'normal':torch.ones(2)}
    monkeypatch.setattr(w,'collect_normal_inputs',normal)
    monkeypatch.setattr(w,'data_fingerprint',lambda *a:'data')
    def select(cache,solution,beta,config,output,project,device):
        assert not fixed[0];output.mkdir()
        specs={name:{'layers':[1],'kind':'uvarfs','obj':{'active':list(range(256)),
                 'scales':[1/16]*256}} for name in w.METHODS}
        w.save_json(specs,output/'representation-specs.json');fixed[0]=True
        return {'results':{'nuisance':{'selection_seconds':.1},'beta_zero':{'selection_seconds':.1}}}
    monkeypatch.setattr(w,'select_pilot',select)
    def memories(extractor,train,specs,cfg,progress):
        m={**memory,'memory_rows':{name:2 for name in specs}}
        w.save_json(m,progress.parent/'memory_manifest.json')
        return {name:np.ones((2,256)) for name in specs}
    monkeypatch.setattr(w,'build_memories',memories)
    def evaluate(extractor,test,specs,memories,cfg,progress):
        a=pred[['image_path','label','mask_path']].copy()
        for name in specs:a[name]=pred[w.ANCHORS.get(name,'main')]
        if bad_anchor:a.loc[0,'nuisance_mean']+=.01
        a.to_csv(progress.parent/'image_predictions.csv',index=False)
        w.save_json({'stage':'complete','methods':6},progress)
        return [{'method':name,**{m:.7 for m in w.METRICS}} for name in specs]
    monkeypatch.setattr(w,'evaluate',evaluate)
    if bad_anchor:
        with pytest.raises(ValueError,match='anchor predictions'):w.run(source,out,config,root)
        assert not (out/'complete.json').exists() and (out/'failure.json').is_file()
    else:
        w.run(source,out,config,root)
        assert w.strict_complete(out)['retained_control_rows']==352
        saved=w.read(out/'run_metadata.json')
        assert not saved['new_independent_fit_generated'] and saved['old_controls_reused_not_refitted']
        with (out/'metrics.csv').open('a') as f:f.write('\n')
        with pytest.raises(ValueError,match='hash mismatch'):w.strict_complete(out)
    assert before=={name:w.sha256(source/name) for name in w.INPUTS}


def test_cpu_cache_roundtrip_is_weights_only_safe(tmp_path):
    data={'mode':{'patches':{1:torch.ones(2,3)},'manifest':{'paths':['train/good/a']}}}
    p=tmp_path/'cache.pt';torch.save(w.cpu_tree(data),p)
    loaded=torch.load(p,weights_only=True)
    torch.testing.assert_close(loaded['mode']['patches'][1],data['mode']['patches'][1])


def test_all_six_three_fit_report_keeps_no_mask_missing_and_zero_paired_effect(tmp_path,monkeypatch):
    import scripts.analyze_normal_tail_validation as a
    from uvarfs.protocol import expected_method_names
    from uvarfs.utils import load_config
    cfg=load_config(Path(__file__).resolve().parents[1]/'configs/default.yaml')
    names=expected_method_names(cfg);assert len(names)==352
    batch=tmp_path/'results/batch';batch.mkdir(parents=True)
    w.save_json({'head':'head','code_fingerprint':'eval','worker_sha256':'worker'},batch/'startup.json')
    monkeypatch.setattr(a,'__file__',str(tmp_path/'scripts/analyze.py'))
    monkeypatch.setattr(a,'strict_complete',lambda p:{'fit_seed':int(p.parent.name[4:]),'dataset':p.name})
    for seed in [42,123,3407]:
        for dataset in ['brain','liver','resc','oct2017','xray','camelyon16']:
            folder=batch/'results'/f'seed{seed}'/dataset;(folder/'controls').mkdir(parents=True)
            metrics={'image_auroc':1.,'image_auprc':1.}
            if dataset in {'brain','liver','resc'}:metrics.update(pixel_auroc=.7,pixel_auprc=.6,aupro=.5)
            pd.DataFrame([{'method':m,**metrics,'dataset':dataset,'fit_seed':seed} for m in w.METHODS]).to_csv(folder/'metrics.csv',index=False)
            pd.DataFrame([{'method':m,**metrics,'dataset':dataset} for m in names]).to_csv(folder/'controls/metrics.csv',index=False)
            pred={'image_path':['a','b'],'label':[0,1],'mask_path':['','m']}
            pd.DataFrame({**pred,**{m:[.1,.9] for m in w.METHODS}}).to_csv(folder/'image_predictions.csv',index=False)
            pd.DataFrame({**pred,**{m:[.1,.9] for m in names}}).to_csv(folder/'controls/image_predictions.csv',index=False)
            w.save_json({'evaluation_git_head':'head','evaluation_code_fingerprint':'eval','worker_script_sha256':'worker',
                'source_code_fingerprint':'source','source_config':{'seed':seed,'results_dir':f'old/seed{seed}','same':True}},folder/'run_metadata.json')
    out=tmp_path/'reports/analysis';a.analyze(batch,out,draws=100)
    report=w.read(out/'audit-provenance.json')
    assert report['new_evaluations']==108 and report['retained_control_rows']==6336
    assert report['independent_source_fit_seeds']==[42,123,3407] and not report['new_fit_repeats_claimed']
    delta=pd.read_csv(out/'paired_differences_by_fit.csv')
    assert delta[w.METRICS].fillna(0).eq(0).all().all()
    assert delta[delta.dataset=='xray'].pixel_auprc.isna().all()
    intervals=pd.read_csv(out/'conditional_image_intervals.csv')
    assert intervals.ci95_lower.eq(0).all() and intervals.ci95_upper.eq(0).all()
