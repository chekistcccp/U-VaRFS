"""Fresh ASLS/sparse factorial fitting, isolated normal audit and fixed BMAD evaluation."""
from __future__ import annotations
import argparse,json,math,os,shutil,subprocess,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scripts.evaluate_pursuit_support import (read,atomic,sha256,INPUTS,METRICS,
    SOURCE_VERSION,validate_reference,strict_complete as pursuit_complete)
from scripts.fit_mechanism_validation import run as fit_extension
from scripts.evaluate_normal_tail import strict_complete as tail_complete
from scripts.run_normal_tail_pilot import project_path
from uvarfs.mechanism_validation import VERSION,SETTINGS,expected_methods
from uvarfs.pipeline_eval import build_memories,evaluate,loader,transform
from uvarfs.memory import make_index
from uvarfs.data import discover_bmad_roots,scan_split
from uvarfs.dinov3 import DINOv3Extractor
from uvarfs.protocol import code_fingerprint,data_fingerprint,completed_result_compatible
from uvarfs.utils import load_config,set_seed

METHODS=expected_methods()
HISTORY_ANCHORS={'pursuit_anchor':'pursuit_matching','pursuit_beta_zero_anchor':'pursuit_beta_zero'}

def normal_audit_methods(specs):
    return {name:spec for name,spec in specs.items() if name.startswith(('mechanism_','random_weighted_','random_uniform_','random_beta_zero_','dense_')) or name in {'asls_raw','asls_pca'}}

@torch.inference_mode()
def full_normal_audit(extractor,train,specs,cfg,audit_paths,out):
    """Actual full-grid scoring; all outer query images excluded from this memory."""
    if out.exists():raise ValueError('fresh normal audit output required')
    out.mkdir();paths=set(audit_paths)
    if len(paths)!=64:raise ValueError('64 distinct outer audit paths required')
    lookup={str(s.image):s for s in train};queries=[lookup[p] for p in audit_paths]
    reference=[s for s in train if str(s.image) not in paths]
    if any(s.label!=0 or s.split!='train' for s in [*queries,*reference]):raise ValueError('official normal train only')
    specs=normal_audit_methods(specs);set_seed(int(cfg['seed']))
    memories=build_memories(extractor,reference,specs,cfg,out/'memory_progress.json')
    manifest=read(out/'memory_manifest.json')
    if paths & set(manifest['normal_train_paths']):raise ValueError('audit image present in audit memory')
    indices={name:make_index(mem,cfg['eval']) for name,mem in memories.items()};scores={name:[] for name in specs}
    patch_sums={name:0. for name in specs};patch_count=0;seen=0
    for batch in loader(queries,cfg,with_mask=False,test=True):
        feats=extractor(batch['image']);cache={};values={};size=len(batch['image'])
        for name,spec in specs.items():
            z=transform(feats,spec,cache);index=indices[name]
            flat=z.reshape(-1,z.shape[-1])
            value=index.score_tensor(flat) if hasattr(index,'score_tensor') else torch.from_numpy(index.score(flat)).to(z.device)
            values[name]=value.reshape(size,-1)
            k=max(1,math.ceil(values[name].shape[1]*float(cfg['eval']['topk_fraction'])))
            scores[name].extend(values[name].topk(k,dim=1).values.mean(1).float().cpu().tolist())
        teacher=values['asls_raw'];patch_count+=teacher.numel()
        for name in specs:patch_sums[name]+=float((values[name]-teacher).square().sum())
        seen+=size;print(f'[mechanism:normal-audit] full images {seen}/64',flush=True)
    frame=pd.DataFrame({'image_path':audit_paths,**scores});frame.to_csv(out/'image_scores.csv',index=False)
    target=np.asarray(scores['asls_raw']);rows=[]
    for name,value in scores.items():
        value=np.asarray(value);rows.append(dict(method=name,normal_images=64,
            image_score_mse=float(np.mean((value-target)**2)),patch_score_mse=patch_sums[name]/patch_count,
            mean_normal_score=float(value.mean()),normal_image_score_relative_mse=float(np.mean((value-target)**2)/max(np.mean(target**2),1e-4))))
    pd.DataFrame(rows).to_csv(out/'normal_errors.csv',index=False)
    atomic(out/'protocol.json',dict(stage='complete',audit_images=64,methods=len(specs),
        full_grid_top1_percent=True,audit_queries_excluded_from_memory=True,used_for_selection=False,
        audit_paths=audit_paths,reference_normal_images=len(reference),memory_rows=manifest['memory_rows']))
    del memories,indices
    if torch.cuda.is_available():torch.cuda.empty_cache()


def validate_history_anchor(new_predictions,old_predictions,new_metrics,old_metrics,new_memory,old_memory):
    # Reuse the strict shared-memory/roster check, mapping two historical names.
    identity=['image_path','label','mask_path']
    a=new_predictions[identity+list(HISTORY_ANCHORS)].rename(columns={'pursuit_anchor':'v22_anchor','pursuit_beta_zero_anchor':'v22_beta_zero_anchor'})
    b=old_predictions[identity+list(HISTORY_ANCHORS.values())].rename(columns={'pursuit_matching':'main','pursuit_beta_zero':'asls_no_variability_uvarfs'})
    am=new_metrics[new_metrics.method.isin(HISTORY_ANCHORS)].copy()
    am['method']=am.method.map({'pursuit_anchor':'v22_anchor','pursuit_beta_zero_anchor':'v22_beta_zero_anchor'})
    bm=old_metrics.copy();bm['method']=bm.method.replace({'pursuit_matching':'main','pursuit_beta_zero':'asls_no_variability_uvarfs'})
    checked=validate_reference(a,b,am,bm,new_memory,old_memory)
    for item in checked:
        item['evaluated']='pursuit_anchor' if item['evaluated']=='v22_anchor' else 'pursuit_beta_zero_anchor'
    return checked


def strict_complete(out,*,verify_files=True):
    out=Path(out);complete=read(out/'complete.json');meta=read(out/'run_metadata.json')
    if (out/'failure.json').exists() or complete['evaluation_version']!=VERSION or meta['evaluation_version']!=VERSION or not complete['four_anchors_exact']:
        raise ValueError('incomplete mechanism run')
    f=pd.read_csv(out/'metrics.csv');needed=METRICS if meta['dataset'] in {'brain','liver','resc'} else METRICS[:2]
    if len(f)!=35 or set(f.method)!=METHODS or not f.evaluation_version.eq(VERSION).all() or not np.isfinite(f[needed]).all().all() or not f[needed].apply(lambda c:c.between(0,1)).all().all():
        raise ValueError('35 valid fixed representations required')
    split=read(out/'selection/normal-role-split.json')
    sets=[set(split[k]) for k in ['fit_image_positions','selection_image_positions','audit_image_positions']]
    if [len(s) for s in sets]!=[144,48,64] or any(sets[i]&sets[j] for i,j in [(0,1),(0,2),(1,2)]) or not set(split['nuisance_image_positions'])<=sets[0]:
        raise ValueError('normal role isolation differs')
    if read(out/'eval_progress.json')['methods']!=35 or read(out/'normal-full-audit/protocol.json')['stage']!='complete':raise ValueError('final evaluation or outer audit missing')
    if sha256(out/'selection/representation-specs.json')!=read(out/'selection/report.json')['representations_frozen_before_audit_sha256']:
        raise ValueError('frozen fitted representations changed')
    if verify_files:
        for rel,r in read(out/'files-sha256.json').items():
            p=(out/rel).resolve()
            if not p.is_relative_to(out.resolve()) or p.stat().st_size!=r['bytes'] or sha256(p)!=r['sha256']:raise ValueError('mechanism artifact differs: '+rel)
        project=Path(__file__).resolve().parents[1]
        for rel,r in read(out/'immutable-references.json').items():
            p=(project/rel).resolve()
            if not p.is_relative_to((project/'results').resolve()) or p.stat().st_size!=r['bytes'] or sha256(p)!=r['sha256']:raise ValueError('historical reference differs: '+rel)
    if not f[f.method.eq('mechanism_sparse')].feature_dim.eq(256).all():raise ValueError('main K256 differs')
    dense_dim=int(f.loc[f.method.eq('asls_raw'),'feature_dim'].iloc[0])
    if dense_dim<=256 or not f[f.method.isin(['dense_weighted','dense_beta_zero'])].feature_dim.eq(dense_dim).all() or not f.memory_size.eq(20000).all():
        raise ValueError('dense representation or shared memory budget differs')
    return dict(rows=35,retained_control_rows=352,fit_seed=meta['fit_seed'],dataset=meta['dataset'],anchors_reproduced_exactly=True)


def run(source,output,selector_config,root,*,cache_source,history_source,prepare_only=False):
    root=Path(root).resolve();source,cache_source,history_source,selector_config=[project_path(p,root) for p in [source,cache_source,history_source,selector_config]]
    output=project_path(output,root,output=True)
    if any(output.is_relative_to(p) or p.is_relative_to(output) for p in [source,cache_source,history_source]):raise ValueError('separate immutable sources required')
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in {'0','1','2','3'} or not torch.cuda.is_available():raise ValueError('one authorized CUDA GPU required')
    if load_config(selector_config)!=SETTINGS:raise ValueError('fixed mechanism settings required')
    if subprocess.run(['git','check-ignore','-q','--',str(output.relative_to(root))],cwd=root).returncode:raise ValueError('Git ignored result required')
    meta=read(source/'run_metadata.json');cfg=meta['config'];seed=int(cfg['seed']);dataset=meta['dataset']
    if meta['experiment_version']!=SOURCE_VERSION:raise ValueError('frozen v22 source required')
    from scripts.analyze_results import verify_recorded_code
    if not verify_recorded_code(meta):raise ValueError('original source revision identity invalid')
    source_hashes={name:sha256(source/name) for name in INPUTS};cache_path=cache_source/'normal-inputs.pt'
    cachemeta=read(cache_source/'run_metadata.json');cachefiles=read(cache_source/'files-sha256.json')
    if cachemeta['source_inputs_sha256']!=source_hashes or cachemeta['fit_seed']!=seed or cachemeta['dataset']!=dataset or sha256(cache_path)!=cachefiles['normal-inputs.pt']['sha256']:
        raise ValueError('normal cache/source pairing differs')
    refs={cache_path.relative_to(root).as_posix():{'sha256':sha256(cache_path),'bytes':cache_path.stat().st_size}}
    for folder,names in [(cache_source,['metrics.csv','image_predictions.csv','run_metadata.json']),
                         (history_source,['metrics.csv','image_predictions.csv','run_metadata.json','files-sha256.json'])]:
        for name in names:
            p=folder/name;refs[p.relative_to(root).as_posix()]={'sha256':sha256(p),'bytes':p.stat().st_size}
    output.mkdir(parents=True);started=time.perf_counter();stage=lambda v:atomic(output/'stage.json',{'stage':v,'updated_unix':time.time()})
    atomic(output/'run_metadata.json',dict(evaluation_version=VERSION,source_experiment_version=SOURCE_VERSION,
        source_git_head=meta['git_head'],source_code_fingerprint=meta['code_fingerprint'],
        evaluation_git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        evaluation_code_fingerprint=code_fingerprint(root),worker_script_sha256=sha256(__file__),
        fitter_script_sha256=sha256(root/'scripts/fit_mechanism_validation.py'),selector_config_sha256=sha256(selector_config),
        source_config=cfg,source_root=str(source),history_source_relative=history_source.relative_to(root).as_posix(),
        cache_source_relative=cache_source.relative_to(root).as_posix(),source_inputs_sha256=source_hashes,
        fit_seed=seed,dataset=dataset,physical_gpu=int(os.environ['CUDA_VISIBLE_DEVICES']),
        new_representation_fit=True,new_independent_normal_sampling_claimed=False,registered_main_replaced=False))
    try:
        set_seed(seed);torch.cuda.reset_peak_memory_stats();stage('fresh_normal_fitting')
        selection=fit_extension(cache_path,source/'fit_manifest.json',cfg,selector_config,output/'selection',root,'cuda')
        if prepare_only:
            atomic(output/'engineering-preparation.json',dict(stage='normal_preparation_complete',anomaly_performance_assessed=False,
                representations_frozen_before_audit_sha256=selection['representations_frozen_before_audit_sha256']))
            return
        # Historical numerical solutions are read only after the new representation fit.
        specs=read(output/'selection/representation-specs.json')
        for name,file in [('v22_anchor','uvarfs_main.json'),('v22_beta_zero_anchor','uvarfs_asls_no_variability.json')]:
            obj=read(source/file);specs[name]=dict(layers=obj['layers'],layer_normalization=obj['layer_normalization'],kind='uvarfs',obj={'active':obj['active'],'scales':obj['scales']})
        historical=read(history_source/'selection/representation-specs.json')
        for name,old in HISTORY_ANCHORS.items():specs[name]=historical[old]
        if set(specs)!=METHODS:raise ValueError('fixed 35 method set differs')
        atomic(output/'evaluation-specs.json',specs);frozen=sha256(output/'evaluation-specs.json')
        tail_complete(cache_source);pursuit_complete(history_source)
        roots=discover_bmad_roots(cfg['paths']['processed_data']);train=scan_split(roots[dataset],'train',dataset)
        extractor=DINOv3Extractor(cfg['paths']['model_dir'],int(cfg['model']['input_size']),cfg['model']['amp'])
        manifest=read(source/'fit_manifest.json');split=read(output/'selection/normal-role-split.json')
        audit_paths=[manifest['normal_train_paths'][i] for i in split['audit_image_positions']]
        stage('isolated_full_normal_audit');t=time.perf_counter()
        full_normal_audit(extractor,train,specs,cfg,audit_paths,output/'normal-full-audit');audit_seconds=time.perf_counter()-t
        # Test tables and test images are only parsed after normal fitting and audit freeze.
        frame=pd.read_csv(source/'metrics.csv');test=scan_split(roots[dataset],'test',dataset)
        if len(frame)!=352 or data_fingerprint(train,test)!=meta['data_fingerprint'] or not completed_result_compatible(frame,meta,read(source/'eval_progress.json'),cfg,meta['code_fingerprint'],dataset in {'brain','liver','resc'},meta['data_fingerprint']):
            raise ValueError('frozen source data/evaluation identity differs')
        stage('memory');t=time.perf_counter();set_seed(seed)
        memories=build_memories(extractor,train,specs,cfg,output/'memory_progress.json');memory_seconds=time.perf_counter()-t
        stage('evaluation');t=time.perf_counter();fresh=pd.DataFrame(evaluate(extractor,test,specs,memories,cfg,output/'eval_progress.json'));evaluation_seconds=time.perf_counter()-t
        pred=pd.read_csv(output/'image_predictions.csv',float_precision='round_trip');mem=read(output/'memory_manifest.json')
        anchors=validate_reference(pred,pd.read_csv(source/'image_predictions.csv',float_precision='round_trip'),fresh,frame,mem,read(source/'memory_manifest.json'))
        anchors+=validate_history_anchor(pred,pd.read_csv(history_source/'image_predictions.csv',float_precision='round_trip'),fresh,pd.read_csv(history_source/'metrics.csv'),mem,read(history_source/'memory_manifest.json'))
        if frozen!=sha256(output/'evaluation-specs.json'):raise ValueError('audit/evaluation changed representations')
        controls=output/'controls';controls.mkdir()
        for name in INPUTS:
            if sha256(source/name)!=source_hashes[name]:raise ValueError('source changed')
            shutil.copy2(source/name,controls/name)
        for rel,r in refs.items():
            if sha256(root/rel)!=r['sha256']:raise ValueError('immutable historical source changed')
        def dimension(spec):
            if spec['kind']=='uvarfs' or spec['kind']=='selected_raw':return len(spec['obj']['active'])
            if spec['kind']=='pca':return len(spec['obj']['components'])
            return len(spec['layers'])*int(extractor.hidden_dim)
        fresh['dataset']=dataset;fresh['fit_seed']=seed;fresh['evaluation_version']=VERSION
        fresh['feature_dim']=fresh.method.map(lambda n:dimension(specs[n]));fresh['memory_size']=fresh.method.map(mem['memory_rows']);fresh.to_csv(output/'metrics.csv',index=False)
        atomic(output/'anchor-reproduction.json',dict(anchors=anchors,all_predictions_exact=True))
        atomic(output/'immutable-references.json',refs)
        atomic(output/'costs.json',dict(selection_seconds=selection['new_fit_seconds'],cached_audit_inclusive_seconds=selection['fit_and_cached_audit_seconds'],
            full_normal_audit_seconds=audit_seconds,memory_seconds=memory_seconds,evaluation_seconds=evaluation_seconds,
            wall_seconds=time.perf_counter()-started,peak_gpu_gb=torch.cuda.max_memory_allocated()/1024**3,
            scope='fresh ASLS/support/common refits plus separate normal audit and 35 evaluations; old costs retained'))
        stage('complete');atomic(output/'files-sha256.json',{p.relative_to(output).as_posix():{'sha256':sha256(p),'bytes':p.stat().st_size} for p in sorted(output.rglob('*')) if p.is_file()})
        atomic(output/'complete.json',dict(evaluation_version=VERSION,stage='complete',rows=35,four_anchors_exact=True))
        strict_complete(output)
    except BaseException as error:
        import traceback
        atomic(output/'failure.json',dict(stage=read(output/'stage.json')['stage'],error=repr(error),traceback=traceback.format_exc()));raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['source','out','selector-config','cache-source','history-source']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--prepare-only',action='store_true');a=p.parse_args()
    run(a.source,a.out,a.selector_config,Path(__file__).resolve().parents[1],cache_source=a.cache_source,history_source=a.history_source,prepare_only=a.prepare_only)
if __name__=='__main__':main()
