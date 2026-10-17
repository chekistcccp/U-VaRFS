#!/usr/bin/env python3
"""Paired detection validation of fit-only ridge coordinate supports."""
from __future__ import annotations
import argparse, hashlib, json, os, shutil, subprocess, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from scripts.fit_ridge_support import run as fit_extension, EXPECTED, METHODS
from scripts.run_normal_tail_pilot import project_path, sha256 as _sha256
from scripts.evaluate_normal_tail import strict_complete as tail_complete
from uvarfs.data import discover_bmad_roots, scan_split
from uvarfs.dinov3 import DINOv3Extractor
from uvarfs.pipeline_eval import build_memories, evaluate
from uvarfs.protocol import code_fingerprint, data_fingerprint, completed_result_compatible
from uvarfs.utils import load_config, save_json, set_seed

VERSION = 'ridge-support-continuation-v1'
SOURCE_VERSION = 'gpu-eval-v22-diverse-matching-support'
ANCHORS = {'v22_anchor':'main', 'v22_beta_zero_anchor':'asls_no_variability_uvarfs'}
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
    if (out/'failure.json').exists():raise ValueError('failed ridge evaluation cannot be complete')
    if (complete['evaluation_version']!=VERSION or meta['evaluation_version']!=VERSION or
        complete['stage']!='complete' or not complete['anchors_reproduced_exactly']):
        raise ValueError('incomplete or unverified ridge evaluation')
    frame=pd.read_csv(out/'metrics.csv')
    if set(frame.method)!=METHODS or len(frame)!=10 or not frame.evaluation_version.eq(VERSION).all():
        raise ValueError('ten predefined ridge/anchor/beta0/uniform representations required')
    expected=METRICS if meta['dataset'] in {'brain','liver','resc'} else METRICS[:2]
    if not np.isfinite(frame[expected]).all().all() or not frame[expected].apply(lambda c:c.between(0,1)).all().all():
        raise ValueError('incomplete detection/localization metrics')
    progress=read(out/'eval_progress.json')
    if progress['stage']!='complete' or progress['methods']!=10:
        raise ValueError('final evaluation protocol missing')
    original=pd.read_csv(out/'controls/metrics.csv')
    if len(original)!=352 or not original.experiment_version.eq(SOURCE_VERSION).all():
        raise ValueError('all 352 historical controls must be retained separately')
    previous=pd.read_csv(out/'previous_tail/metrics.csv')
    if len(previous)!=6 or not previous.evaluation_version.eq('normal-tail-evaluation-v1').all():
        raise ValueError('previous six-method tail control is incomplete')
    if verify_files:
        for rel,record in read(out/'files-sha256.json').items():
            path=(out/rel).resolve()
            if not path.is_relative_to(out.resolve()) or not path.is_file() or path.stat().st_size!=record['bytes'] or sha256(path)!=record['sha256']:
                raise ValueError('ridge artifact hash mismatch: '+rel)
    cache=read(out/'cache-reference.json')
    if verify_files:
        project=Path(__file__).resolve().parents[1];path=(project/cache['relative_path']).resolve()
        if (not path.is_relative_to((project/'results').resolve()) or path.stat().st_size!=cache['bytes'] or sha256(path)!=cache['sha256']):
            raise ValueError('referenced original normal cache differs')
    return {'rows':10,'retained_control_rows':352,'fit_seed':meta['fit_seed'],
            'dataset':meta['dataset'],'anchors_reproduced_exactly':True}


def run(source, output, selector_config, root, *, cache_source, prepare_only=False):
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
    if pilot_cfg!=EXPECTED:raise ValueError('fixed ridge extension configuration required')
    cache_source=project_path(cache_source,root)
    if output.is_relative_to(cache_source) or cache_source.is_relative_to(output):
        raise ValueError('source cache and new outputs must be separate')
    cache_path=cache_source/'normal-inputs.pt'
    cache_meta=read(cache_source/'run_metadata.json')
    cached_files=read(cache_source/'files-sha256.json')
    if cache_meta['dataset']!=dataset or cache_meta['fit_seed']!=seed or cache_meta['source_inputs_sha256']!=source_hashes:
        raise ValueError('normal cache belongs to a different source fit')
    cache_hash=sha256(cache_path)
    if cache_hash!=cached_files['normal-inputs.pt']['sha256']:raise ValueError('normal cache hash differs')
    previous_hashes={name:sha256(cache_source/name) for name in ['metrics.csv','image_predictions.csv','run_metadata.json']}
    if any(digest!=cached_files[name]['sha256'] for name,digest in previous_hashes.items()):
        raise ValueError('previous tail source files differ from their closed record')
    output.mkdir(parents=True)
    start=time.perf_counter();stage=lambda value:atomic(output/'stage.json',{'stage':value,'updated_unix':time.time()})
    base_meta={'evaluation_version':VERSION,'source_experiment_version':SOURCE_VERSION,
        'source_git_head':meta['git_head'],'source_code_fingerprint':meta['code_fingerprint'],
        'evaluation_git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        'evaluation_code_fingerprint':code_fingerprint(root),'worker_script_sha256':sha256(__file__),
        'fitter_script_sha256':sha256(root/'scripts/fit_ridge_support.py'),'selector_config_sha256':sha256(selector_config),'selector_config':pilot_cfg,
        'source_root':str(source),'source_inputs_sha256':source_hashes,'source_config':cfg,
        'previous_tail_inputs_sha256':previous_hashes,
        'fit_seed':seed,'dataset':dataset,'physical_gpu':int(os.environ['CUDA_VISIBLE_DEVICES']),
        'new_independent_fit_generated':False,'independent_source_fit_seeds':[42,123,3407],
        'registered_main_replaced':False,'old_controls_reused_not_refitted':True,
        'new_evaluations':10,'unique_new_representations':8,'source_data_fingerprint':meta['data_fingerprint']}
    atomic(output/'run_metadata.json',base_meta)
    try:
        set_seed(seed);torch.cuda.reset_peak_memory_stats();stage('ridge_normal_fitting')
        selection=fit_extension(cache_path,source/'uvarfs_main.json',source/'uvarfs_asls_no_variability.json',
            selector_config,output/'selection',root,device='cuda')
        specs=read(output/'selection/representation-specs.json')
        if set(specs)!=METHODS:raise ValueError('unexpected evaluation method set')
        selection_hash=sha256(output/'selection/representation-specs.json')
        if prepare_only:
            for name in INPUTS:
                if sha256(source/name)!=source_hashes[name]:raise ValueError('frozen preparation input changed: '+name)
            atomic(output/'engineering-preparation.json',{'stage':'normal_preparation_complete',
                'anomaly_performance_assessed':False,'normal_cache_reused':True,
                'representation_specs_sha256':selection_hash,'source_inputs_unchanged':True})
            return
        tail_complete(cache_source)
        roots=discover_bmad_roots(cfg['paths']['processed_data']);train=scan_split(roots[dataset],'train',dataset)
        extractor=DINOv3Extractor(cfg['paths']['model_dir'],int(cfg['model']['input_size']),cfg['model']['amp'])
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
        tail=output/'previous_tail';tail.mkdir()
        for name in ['metrics.csv','image_predictions.csv','run_metadata.json']:
            shutil.copy2(cache_source/name,tail/name)
            if sha256(cache_source/name)!=previous_hashes[name] or sha256(tail/name)!=previous_hashes[name]:
                raise ValueError('previous tail input changed')
        if sha256(cache_path)!=cache_hash:raise ValueError('source normal cache changed')
        atomic(output/'cache-reference.json',{'relative_path':cache_path.relative_to(root).as_posix(),'sha256':cache_hash,
            'bytes':cache_path.stat().st_size,'original_cache_reused':True})
        atomic(output/'costs.json',{'normal_cache_regeneration_seconds':0.,
            'selection_seconds':selection['new_fit_seconds'],
            'memory_seconds':memory_seconds,'evaluation_seconds':evaluation_seconds,
            'wall_seconds':time.perf_counter()-start,'peak_gpu_gb':torch.cuda.max_memory_allocated()/1024**3,
            'scope':'reused normal cache, new ridge fits and ten evaluations; old full-fitting costs remain in controls'})
        stage('complete')
        atomic(output/'files-sha256.json',{p.relative_to(output).as_posix():{'sha256':sha256(p),'bytes':p.stat().st_size}
            for p in sorted(output.rglob('*')) if p.is_file()})
        atomic(output/'complete.json',{'evaluation_version':VERSION,'stage':'complete','rows':10,
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
    p.add_argument('--selector-config',type=Path,default=Path('configs/ridge_support.yaml'))
    p.add_argument('--prepare-only',action='store_true');p.add_argument('--cache-source',type=Path,required=True)
    a=p.parse_args();run(a.source,a.out,a.selector_config,Path(__file__).resolve().parents[1],
                        prepare_only=a.prepare_only,cache_source=a.cache_source)


if __name__=='__main__':main()
