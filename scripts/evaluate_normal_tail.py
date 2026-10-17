#!/usr/bin/env python3
"""Paired detection validation of the opt-in, frozen-pool normal-tail selector."""
from __future__ import annotations
import argparse, hashlib, json, os, shutil, subprocess, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from scripts.run_normal_tail_pilot import run as select_pilot, project_path, sha256 as _sha256
from uvarfs.data import discover_bmad_roots, scan_split
from uvarfs.dinov3 import DINOv3Extractor
from uvarfs.pipeline_fit import collect_normal_inputs
from uvarfs.pipeline_eval import build_memories, evaluate
from uvarfs.protocol import code_fingerprint, data_fingerprint, completed_result_compatible
from uvarfs.utils import load_config, save_json, set_seed

VERSION = 'normal-tail-evaluation-v1'
SOURCE_VERSION = 'gpu-eval-v22-diverse-matching-support'
ANCHORS = {'nuisance_mean':'main', 'beta_zero_mean':'asls_no_variability_uvarfs'}
METHODS = {'nuisance_mean','nuisance_tail','nuisance_tail_uniform',
           'beta_zero_mean','beta_zero_tail','beta_zero_tail_uniform'}
METRICS = ['image_auroc','image_auprc','pixel_auroc','pixel_auprc','aupro']
INPUTS = ['run_metadata.json','metrics.csv','eval_progress.json','fit_manifest.json',
          'memory_manifest.json','image_predictions.csv','uvarfs_main.json',
          'uvarfs_asls_no_variability.json']


def sha256(path):return _sha256(Path(path))


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def atomic(path, data):
    path=Path(path);part=path.with_name(path.name+'.tmp')
    save_json(data,part);part.replace(path)


def cpu_tree(obj):
    if isinstance(obj,torch.Tensor):return obj.detach().cpu()
    if isinstance(obj,dict):return {k:cpu_tree(v) for k,v in obj.items()}
    if isinstance(obj,(tuple,list)):return [cpu_tree(v) for v in obj]
    return obj


def validate_reference(new_predictions, reference_predictions, new_metrics,
                       reference_metrics, new_memory, reference_memory):
    """Exact paired anchors guard reuse of all 352 frozen controls, not tune scores."""
    for key in ['normal_train_indices','normal_train_paths','patch_batches',
                'memory_capacity','reservoir_seed','shared_sampling_across_methods']:
        if new_memory.get(key)!=reference_memory.get(key):
            raise ValueError('normal memory protocol differs: '+key)
    count=set(reference_memory['memory_rows'].values())
    if len(count)!=1 or set(new_memory['memory_rows'].values())!=count:
        raise ValueError('normal memory row budgets differ')
    for column in ['image_path','label','mask_path']:
        if not new_predictions[column].fillna('').equals(reference_predictions[column].fillna('')):
            raise ValueError('evaluation roster differs: '+column)
    checked=[]
    new_metrics=new_metrics.set_index('method');reference_metrics=reference_metrics.set_index('method')
    for fresh,old in ANCHORS.items():
        if not np.array_equal(new_predictions[fresh].to_numpy(),reference_predictions[old].to_numpy()):
            raise ValueError('paired anchor predictions differ: '+fresh)
        columns=[m for m in METRICS if m in reference_metrics and pd.notna(reference_metrics.loc[old,m])]
        a=new_metrics.loc[fresh,columns].to_numpy(dtype=float)
        b=reference_metrics.loc[old,columns].to_numpy(dtype=float)
        if not np.allclose(a,b,rtol=0,atol=1e-12):
            raise ValueError('paired anchor metric differs: '+fresh)
        checked.append(dict(evaluated=fresh,source=old,predictions_exact=True,
                            metric_max_abs_difference=float(np.max(np.abs(a-b)))))
    return checked


def strict_complete(out, *, verify_files=True):
    out=Path(out);complete=read(out/'complete.json');meta=read(out/'run_metadata.json')
    if (out/'failure.json').exists():raise ValueError('failed tail evaluation cannot be complete')
    if (complete['evaluation_version']!=VERSION or meta['evaluation_version']!=VERSION or
        complete['stage']!='complete' or not complete['anchors_reproduced_exactly']):
        raise ValueError('incomplete or unverified tail evaluation')
    frame=pd.read_csv(out/'metrics.csv')
    if set(frame.method)!=METHODS or len(frame)!=6 or not frame.evaluation_version.eq(VERSION).all():
        raise ValueError('six predefined tail/mean/beta0/uniform representations required')
    expected=METRICS if meta['dataset'] in {'brain','liver','resc'} else METRICS[:2]
    if not np.isfinite(frame[expected]).all().all() or not frame[expected].apply(lambda c:c.between(0,1)).all().all():
        raise ValueError('incomplete detection/localization metrics')
    progress=read(out/'eval_progress.json')
    if progress['stage']!='complete' or progress['methods']!=6:
        raise ValueError('final evaluation protocol missing')
    original=pd.read_csv(out/'controls/metrics.csv')
    if len(original)!=352 or not original.experiment_version.eq(SOURCE_VERSION).all():
        raise ValueError('all 352 historical controls must be retained separately')
    if verify_files:
        for rel,record in read(out/'files-sha256.json').items():
            path=(out/rel).resolve()
            if not path.is_relative_to(out.resolve()) or not path.is_file() or path.stat().st_size!=record['bytes'] or sha256(path)!=record['sha256']:
                raise ValueError('tail artifact hash mismatch: '+rel)
    return {'rows':6,'retained_control_rows':352,'fit_seed':meta['fit_seed'],
            'dataset':meta['dataset'],'anchors_reproduced_exactly':True}


def run(source, output, selector_config, root, *, prepare_only=False, prepared_cache=None):
    root=Path(root).resolve();source=project_path(source,root)
    output=project_path(output,root,output=True);selector_config=project_path(selector_config,root)
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError('source and new evaluation outputs must be separate')
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in {'0','1','2','3'} or not torch.cuda.is_available():
        raise ValueError('assign exactly one authorized physical GPU 0--3')
    if subprocess.run(['git','check-ignore','-q','--',str(output.relative_to(root))],cwd=root).returncode:
        raise ValueError('evaluation output must be Git ignored')
    source_hashes={name:sha256(source/name) for name in INPUTS}
    cfg=read(source/'run_metadata.json')['config'];meta=read(source/'run_metadata.json')
    seed=int(cfg['seed']);dataset=meta['dataset']
    if meta['experiment_version']!=SOURCE_VERSION:
        raise ValueError('a complete frozen v22 dataset/fit is required')
    from scripts.analyze_results import verify_recorded_code
    if not verify_recorded_code(meta):raise ValueError('source revision fingerprint invalid')
    pilot_cfg=load_config(selector_config)
    if pilot_cfg!={'pilot_version':'normal-tail-selection-pilot-v1','feature_budget':256,
                   'selector':{'top_fraction':.01,'tail_fraction':.25,'mean_se_multiplier':1.,'chunk':256}}:
        raise ValueError('use the predeclared tail rule without parameter search')
    output.mkdir(parents=True)
    start=time.perf_counter();stage=lambda value:atomic(output/'stage.json',{'stage':value,'updated_unix':time.time()})
    base_meta={'evaluation_version':VERSION,'source_experiment_version':SOURCE_VERSION,
        'source_git_head':meta['git_head'],'source_code_fingerprint':meta['code_fingerprint'],
        'evaluation_git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        'evaluation_code_fingerprint':code_fingerprint(root),'worker_script_sha256':sha256(__file__),
        'selector_config_sha256':sha256(selector_config),'selector_config':pilot_cfg,
        'source_root':str(source),'source_inputs_sha256':source_hashes,'source_config':cfg,
        'fit_seed':seed,'dataset':dataset,'physical_gpu':int(os.environ['CUDA_VISIBLE_DEVICES']),
        'new_independent_fit_generated':False,'independent_source_fit_seeds':[42,123,3407],
        'registered_main_replaced':False,'old_controls_reused_not_refitted':True,
        'new_evaluations':6,'unique_new_representations':4,'source_data_fingerprint':meta['data_fingerprint']}
    atomic(output/'run_metadata.json',base_meta)
    try:
        set_seed(seed);torch.cuda.reset_peak_memory_stats();stage('normal_cache')
        roots=discover_bmad_roots(cfg['paths']['processed_data']);train=scan_split(roots[dataset],'train',dataset)
        extractor=DINOv3Extractor(cfg['paths']['model_dir'],int(cfg['model']['input_size']),cfg['model']['amp'])
        cache_path=output/'normal-inputs.pt';t=time.perf_counter()
        if prepared_cache is None:
            collected=collect_normal_inputs(extractor,train,cfg)
            torch.save(cpu_tree(collected),cache_path);del collected
        else:
            prepared_cache=project_path(prepared_cache,root)
            shutil.copy2(prepared_cache,cache_path)
            if sha256(cache_path)!=sha256(prepared_cache):raise ValueError('prepared normal cache copy differs')
        cache_seconds=time.perf_counter()-t;torch.cuda.empty_cache();stage('normal_tail_selection')
        selection=select_pilot(cache_path,source/'uvarfs_main.json',source/'uvarfs_asls_no_variability.json',
                               selector_config,output/'selection',root,device='cuda')
        specs=read(output/'selection/representation-specs.json')
        if set(specs)!=METHODS:raise ValueError('unexpected evaluation method set')
        selection_hash=sha256(output/'selection/representation-specs.json')
        if prepare_only:
            for name in INPUTS:
                if sha256(source/name)!=source_hashes[name]:raise ValueError('frozen preparation input changed: '+name)
            atomic(output/'engineering-preparation.json',{'stage':'normal_preparation_complete',
                'anomaly_performance_assessed':False,'normal_cache_seconds':cache_seconds,
                'representation_specs_sha256':selection_hash,'source_inputs_unchanged':True})
            return
        # Metric/prediction tables and test data are parsed only after selection.
        # Earlier byte hashes establish source identity, never selection inputs.
        frame=pd.read_csv(source/'metrics.csv')
        if (len(frame)!=352 or not frame.dataset.eq(dataset).all() or
            not completed_result_compatible(frame,meta,read(source/'eval_progress.json'),cfg,
                meta['code_fingerprint'],dataset in {'brain','liver','resc'},meta['data_fingerprint'])):
            raise ValueError('source control evaluation protocol is incomplete')
        test=scan_split(roots[dataset],'test',dataset)
        if data_fingerprint(train,test)!=meta['data_fingerprint']:
            raise ValueError('source data identity changed')
        stage('memory');t=time.perf_counter()
        memories=build_memories(extractor,train,specs,cfg,output/'memory_progress.json')
        memory_seconds=time.perf_counter()-t;stage('evaluation');t=time.perf_counter()
        rows=evaluate(extractor,test,specs,memories,cfg,output/'eval_progress.json')
        evaluation_seconds=time.perf_counter()-t
        fresh=pd.DataFrame(rows)
        anchors=validate_reference(pd.read_csv(output/'image_predictions.csv',float_precision='round_trip'),
            pd.read_csv(source/'image_predictions.csv',float_precision='round_trip'),fresh,frame,
            read(output/'memory_manifest.json'),read(source/'memory_manifest.json'))
        if sha256(output/'selection/representation-specs.json')!=selection_hash:
            raise ValueError('representations changed after selection')
        for name in INPUTS:
            if sha256(source/name)!=source_hashes[name]:raise ValueError('frozen input changed: '+name)
        controls=output/'controls';controls.mkdir()
        for name in INPUTS:shutil.copy2(source/name,controls/name)
        if any(sha256(controls/name)!=source_hashes[name] for name in INPUTS):raise ValueError('control copy failed')
        fresh['dataset']=dataset;fresh['fit_seed']=seed;fresh['evaluation_version']=VERSION
        fresh['feature_dim']=256;fresh['memory_size']=next(iter(read(output/'memory_manifest.json')['memory_rows'].values()))
        fresh.to_csv(output/'metrics.csv',index=False)
        atomic(output/'anchor-reproduction.json',{'anchors':anchors,'all_predictions_exact':True})
        atomic(output/'costs.json',{'normal_cache_regeneration_seconds':cache_seconds,
            'selection_seconds':sum(r['selection_seconds'] for r in selection['results'].values()),
            'memory_seconds':memory_seconds,'evaluation_seconds':evaluation_seconds,
            'wall_seconds':time.perf_counter()-start,'peak_gpu_gb':torch.cuda.max_memory_allocated()/1024**3,
            'scope':'new normal cache, selector and six evaluations; old full-fitting costs remain in controls'})
        stage('complete')
        atomic(output/'files-sha256.json',{p.relative_to(output).as_posix():{'sha256':sha256(p),'bytes':p.stat().st_size}
            for p in sorted(output.rglob('*')) if p.is_file()})
        atomic(output/'complete.json',{'evaluation_version':VERSION,'stage':'complete','rows':6,
            'retained_control_rows':352,'anchors_reproduced_exactly':True,'updated_unix':time.time()})
        strict_complete(output)
    except BaseException as error:
        import traceback
        atomic(output/'failure.json',{'stage':read(output/'stage.json')['stage'],
            'error':repr(error),'traceback':traceback.format_exc()})
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--selector-config',type=Path,default=Path('configs/normal_tail_pilot.yaml'))
    p.add_argument('--prepare-only',action='store_true');p.add_argument('--prepared-cache',type=Path)
    a=p.parse_args();run(a.source,a.out,a.selector_config,Path(__file__).resolve().parents[1],
                        prepare_only=a.prepare_only,prepared_cache=a.prepared_cache)


if __name__=='__main__':main()
