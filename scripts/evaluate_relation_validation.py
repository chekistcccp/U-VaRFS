"""Fixed normal-relation selection and matched detector controls on one block."""
from __future__ import annotations
import argparse,copy,json,os,subprocess,time,traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from uvarfs.relation_selection import VERSION,SETTINGS,expected_methods,fit_design,scientific_digest
from uvarfs.mechanism_validation import partition,jsonable,artifact_digest
from uvarfs.protocol import code_fingerprint,data_fingerprint
from uvarfs.utils import load_config,set_seed
from uvarfs.pipeline_eval import build_memories,evaluate
from uvarfs.data import discover_bmad_roots,scan_split
from uvarfs.dinov3 import DINOv3Extractor
from scripts.evaluate_mechanism_validation import full_normal_audit,strict_complete as parent_complete
from scripts.evaluate_normal_tail import atomic,read,sha256,METRICS
from scripts.run_normal_tail_pilot import project_path,validate_normal_manifest

METHODS=expected_methods()
ANCHORS={'previous_sparse_anchor':'mechanism_sparse','previous_dense_anchor':'dense_weighted',
         'previous_beta_zero_anchor':'mechanism_beta_zero','asls_raw':'asls_raw','asls_pca':'asls_pca',
         'fixed4_raw':'fixed4_raw','all_raw':'all_raw','last_raw':'last_raw'}

def fit_normal(parent,cache,config,out,project,device='cuda'):
    """Only normal source files are read here; historical test tables are deferred."""
    if load_config(config)!=SETTINGS:raise ValueError('fixed relation settings required')
    meta=read(parent/'run_metadata.json');cfg=meta['source_config'];seed=int(meta['fit_seed'])
    if meta['evaluation_version']!='mechanism-factorial-audit-v1':raise ValueError('frozen mechanism parent required')
    if (cfg['uvarfs']['max_features']!=256 or cfg['uvarfs']['cosine_weight_floor_mass']!=1e-4 or
        cfg['uvarfs']['performance_weight_cap_factor']!=4 or cfg['data']['memory_size']!=20000 or
        cfg['data']['variability_images']!=96 or cfg['data']['patches_per_image']!=16 or
        cfg['representation']['layer_normalization']!='l2' or cfg['eval']['topk_fraction']!=.01):
        raise ValueError('frozen normal, feature or detector protocol differs')
    paths=[cache/'normal-inputs.pt',config,parent/'run_metadata.json',parent/'controls/fit_manifest.json']
    paths += [parent/'selection'/n for n in ['asls.json','representation-specs.json','normal-role-split.json']]
    refs={p.relative_to(project).as_posix():{'sha256':sha256(p),'bytes':p.stat().st_size} for p in paths}
    cachekey=(cache/'normal-inputs.pt').relative_to(project).as_posix()
    if refs[cachekey]!=read(parent/'immutable-references.json')[cachekey]:raise ValueError('immutable normal cache differs')
    cached=torch.load(cache/'normal-inputs.pt',map_location='cpu',weights_only=True)['l2']
    validate_normal_manifest(cached['manifest'],read(parent/'controls/fit_manifest.json'))
    fitmask,selectmask,split=partition(cached,seed)
    if split!=read(parent/'selection/normal-role-split.json'):raise ValueError('frozen roles differ')
    layers=read(parent/'selection/asls.json')['selected_layers'];old=read(parent/'selection/representation-specs.json')
    if old['asls_raw']['layers']!=layers:raise ValueError('frozen ASLS input differs')
    training={l:cached['patches'][l][fitmask].to(device) for l in layers}
    selection={l:cached['patches'][l][selectmask].to(device) for l in layers}
    paired=({l:cached['paired_clean'][l].to(device) for l in layers},
            {l:cached['paired_perturbations'][l].to(device) for l in layers})
    result=fit_design(training,selection,cached['groups'][fitmask].to(device),cached['groups'][selectmask].to(device),
                      paired,layers,old,SETTINGS,seed)
    selectionout=out/'selection';selectionout.mkdir()
    atomic(selectionout/'representation-specs.json',result['specs']);atomic(selectionout/'normal-role-split.json',split)
    atomic(selectionout/'asls.json',read(parent/'selection/asls.json'))
    for name,b in result['branches'].items():atomic(selectionout/(name+'-fit.json'),b)
    atomic(selectionout/'report.json',dict(new_fit_seconds=result['new_fit_seconds'],settings=SETTINGS,
        evaluation_version=VERSION,representations_frozen_before_audit_sha256=sha256(selectionout/'representation-specs.json'),
        asls_frozen=True,new_independent_normal_sampling_claimed=False,labels_or_masks_used=False))
    for rel,r in refs.items():
        if sha256(project/rel)!=r['sha256']:raise ValueError('normal source changed')
    return refs,result['new_fit_seconds'],split

def validate_anchors(pred,oldpred,metrics,oldmetrics,memory,oldmemory):
    for key in ['normal_train_indices','normal_train_paths','patch_batches','memory_capacity','reservoir_seed','shared_sampling_across_methods']:
        if memory.get(key)!=oldmemory.get(key):raise ValueError('shared memory changed: '+key)
    if set(memory['memory_rows'].values())!={20000}:raise ValueError('memory20000 required')
    for key in ['image_path','label','mask_path']:
        if not pred[key].fillna('').equals(oldpred[key].fillna('')):raise ValueError('evaluation roster differs')
    a,b=metrics.set_index('method'),oldmetrics.set_index('method')
    for new,old in ANCHORS.items():
        valid=[m for m in METRICS if m in b and pd.notna(b.loc[old,m])]
        if not np.array_equal(pred[new],oldpred[old]) or not np.allclose(a.loc[new,valid].astype(float),b.loc[old,valid].astype(float),atol=1e-12,rtol=0):
            raise ValueError('repeated parent anchor differs: '+new)
    return dict(anchors=ANCHORS,predictions_exact=True,metric_tolerance=1e-12)

def audit_fit(out):
    specs=read(out/'selection/representation-specs.json');width=len(specs['asls_raw']['layers'])*384
    records={label:read(out/'selection'/(label+'-fit.json')) for label in ['nuisance','beta_zero']}
    if records['nuisance']['graph_weights_sha256']!=records['beta_zero']['graph_weights_sha256']:raise ValueError('beta graphs differ')
    for label,b in records.items():
        clean={k:copy.deepcopy(b[k]) for k in ['support_pool','learned','random','baseline','dense']}
        clean['pool']=clean.pop('support_pool')
        for c in [*clean['learned'],*clean['random'].values(),*clean['baseline'].values(),clean['dense']]:
            for k in ['selection_objective','selection_terms','selection_score']:c.pop(k)
        if scientific_digest(clean)!=b['fit_only_sha256']:raise ValueError('fit-only candidate freeze differs')
        winner=min(range(len(b['learned'])),key=lambda i:(b['learned'][i]['selection_score'],i))
        if winner!=b['winner_index']:raise ValueError('normal selected candidate differs')
        main='relation_sparse' if label=='nuisance' else 'relation_beta_zero';c=b['learned'][winner]
        if specs[main]['obj']['active']!=c['active'] or not np.allclose(specs[main]['obj']['scales'],np.sqrt(c['weights']),rtol=1e-12,atol=1e-12):raise ValueError('winner representation differs')
        for c in [*b['learned'],*b['random'].values(),*b['baseline'].values(),b['dense']]:
            p=np.asarray(c['weights']);k=len(p)
            if (abs(p.sum()-1)>1e-10 or p.min()<1e-4/k-1e-12 or p.max()>min(1,4/k)+1e-12 or
                len(set(c['active']))!=k or c['training_objective']>c['initial_uniform_objective']+1e-10 or
                c['diagnostics']['iterations']>800 or (label=='beta_zero' and c['training_terms']['variability_term']!=0)):
                raise ValueError('common refit feasibility differs')
        for seed,c in b['random'].items():
            expected=np.sort(np.random.default_rng(int(seed)).choice(width,256,replace=False)).tolist()
            if c['active']!=expected:raise ValueError('fair Random support differs')
        if b['graph']['nuisance_used'] or b['graph']['same_image_edges'] or not b['graph']['valid_queries']:raise ValueError('clean cross-image graph differs')
    if specs['same_support_beta_zero']['obj']['active']!=specs['relation_sparse']['obj']['active']:raise ValueError('same-support beta0 changed support')
    if read(out/'selection/same_support_beta_zero-fit.json')['training_terms']['variability_term']!=0:raise ValueError('same support beta0 nuisance leaked')

def strict_complete(out,verify_files=True):
    out=Path(out);done=read(out/'complete.json');meta=read(out/'run_metadata.json')
    f=pd.read_csv(out/'metrics.csv');needed=METRICS if meta['dataset'] in {'brain','liver','resc'} else METRICS[:2]
    if (out/'failure.json').exists() or done['evaluation_version']!=VERSION or meta['evaluation_version']!=VERSION or not done['anchors_exact'] or len(f)!=len(METHODS) or set(f.method)!=METHODS or not np.isfinite(f[needed]).all().all() or not f[needed].apply(lambda c:c.between(0,1)).all().all():raise ValueError('incomplete relation run')
    if sha256(out/'selection/representation-specs.json')!=read(out/'selection/report.json')['representations_frozen_before_audit_sha256']:raise ValueError('frozen representations changed')
    if not f[f.method.eq('relation_sparse')].feature_dim.eq(256).all() or not f.memory_size.eq(20000).all():raise ValueError('fixed budgets changed')
    audit=read(out/'normal-full-audit/protocol.json')
    if audit['stage']!='complete' or not audit['audit_queries_excluded_from_memory'] or audit['used_for_selection']:raise ValueError('isolated full-grid audit required')
    audit_fit(out)
    if verify_files:
        project=Path(__file__).resolve().parents[1]
        for base,records in [(out,read(out/'files-sha256.json')),(project,read(out/'immutable-references.json'))]:
            for rel,r in records.items():
                p=(base/rel).resolve()
                if not p.is_relative_to(base.resolve()) or not p.is_file() or p.stat().st_size!=r['bytes'] or sha256(p)!=r['sha256']:raise ValueError('immutable relation artifact differs: '+rel)
    return dict(rows=len(METHODS),fit_seed=meta['fit_seed'],dataset=meta['dataset'],anchors_reproduced_exactly=True)

def run(source,output,selector_config,root,*,cache_source,history_source):
    root=Path(root).resolve();source,cache_source,history_source,selector_config=[project_path(p,root) for p in [source,cache_source,history_source,selector_config]]
    output=project_path(output,root,output=True)
    if output.exists() or any(output.is_relative_to(p) or p.is_relative_to(output) for p in [source,cache_source,history_source]):raise ValueError('fresh isolated output required')
    if not torch.cuda.is_available() or os.environ.get('CUDA_VISIBLE_DEVICES') not in {'0','1','2','3'}:raise ValueError('authorized single GPU required')
    if subprocess.run(['git','check-ignore','-q','--',str(output.relative_to(root))],cwd=root).returncode:raise ValueError('Git ignored output required')
    pm=read(history_source/'run_metadata.json');cfg=pm['source_config'];seed=int(pm['fit_seed']);dataset=pm['dataset']
    if read(source/'run_metadata.json')['config']!=cfg:raise ValueError('original source pairing differs')
    output.mkdir(parents=True);stage=lambda s:atomic(output/'stage.json',dict(stage=s,updated_unix=time.time()))
    started=time.perf_counter();set_seed(seed);torch.cuda.reset_peak_memory_stats()
    atomic(output/'run_metadata.json',dict(evaluation_version=VERSION,source_config=cfg,fit_seed=seed,dataset=dataset,
        parent_relative=history_source.relative_to(root).as_posix(),cache_relative=cache_source.relative_to(root).as_posix(),
        head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),code_fingerprint=code_fingerprint(root),
        worker_sha256=sha256(__file__),settings_sha256=sha256(selector_config),physical_gpu=int(os.environ['CUDA_VISIBLE_DEVICES']),
        new_independent_normal_sampling_claimed=False,registered_main_replaced=False))
    try:
        stage('clean_relation_fitting');refs,fit_seconds,split=fit_normal(history_source,cache_source,selector_config,output,root)
        specs=read(output/'selection/representation-specs.json');frozen=sha256(output/'selection/representation-specs.json')
        # No historical detector values are parsed until all new fitting is frozen.
        parent_complete(history_source)
        for name in ['metrics.csv','image_predictions.csv','memory_manifest.json','files-sha256.json']:
            p=history_source/name;refs[p.relative_to(root).as_posix()]={'sha256':sha256(p),'bytes':p.stat().st_size}
        roots=discover_bmad_roots(cfg['paths']['processed_data']);train=scan_split(roots[dataset],'train',dataset)
        extractor=DINOv3Extractor(cfg['paths']['model_dir'],int(cfg['model']['input_size']),cfg['model']['amp'])
        queries=[read(history_source/'controls/fit_manifest.json')['normal_train_paths'][i] for i in split['audit_image_positions']]
        stage('isolated_full_normal_audit');t=time.perf_counter()
        full_normal_audit(extractor,train,specs,cfg,queries,output/'normal-full-audit',methods_selector=lambda ss:{n:s for n,s in ss.items() if not n.startswith('previous_') and not n.startswith('randomk_') and n not in {'fixed4_raw','all_raw','last_raw'}})
        audit_seconds=time.perf_counter()-t
        test=scan_split(roots[dataset],'test',dataset)
        if data_fingerprint(train,test)!=read(source/'run_metadata.json')['data_fingerprint']:raise ValueError('official dataset changed')
        stage('memory');set_seed(seed);t=time.perf_counter();memories=build_memories(extractor,train,specs,cfg,output/'memory_progress.json');memory_seconds=time.perf_counter()-t
        stage('evaluation');t=time.perf_counter();fresh=pd.DataFrame(evaluate(extractor,test,specs,memories,cfg,output/'eval_progress.json'));eval_seconds=time.perf_counter()-t
        anchors=validate_anchors(pd.read_csv(output/'image_predictions.csv',float_precision='round_trip'),pd.read_csv(history_source/'image_predictions.csv',float_precision='round_trip'),fresh,pd.read_csv(history_source/'metrics.csv'),read(output/'memory_manifest.json'),read(history_source/'memory_manifest.json'))
        def dimension(s):
            if s['kind'] in {'uvarfs','selected_raw'}:return len(s['obj']['active'])
            if s['kind']=='pca':return len(s['obj']['components'])
            return len(s['layers'])*extractor.hidden_dim
        fresh['dataset']=dataset;fresh['fit_seed']=seed;fresh['evaluation_version']=VERSION
        fresh['feature_dim']=fresh.method.map(lambda n:dimension(specs[n]));fresh['memory_size']=20000;fresh.to_csv(output/'metrics.csv',index=False)
        if frozen!=sha256(output/'selection/representation-specs.json'):raise ValueError('test changed normal representation')
        atomic(output/'anchor-reproduction.json',anchors);atomic(output/'immutable-references.json',refs)
        atomic(output/'costs.json',dict(fit_seconds=fit_seconds,normal_audit_seconds=audit_seconds,memory_seconds=memory_seconds,evaluation_seconds=eval_seconds,wall_seconds=time.perf_counter()-started,peak_gpu_gb=torch.cuda.max_memory_allocated()/1024**3,scope='new relation fits and evaluations; inherited ASLS/PCA/cache/old-control cost excluded and retained separately'))
        stage('complete');atomic(output/'files-sha256.json',{p.relative_to(output).as_posix():{'sha256':sha256(p),'bytes':p.stat().st_size} for p in sorted(output.rglob('*')) if p.is_file()})
        atomic(output/'complete.json',dict(evaluation_version=VERSION,anchors_exact=True,rows=len(METHODS)))
        strict_complete(output)
    except BaseException as error:
        atomic(output/'failure.json',dict(stage=read(output/'stage.json')['stage'],error=repr(error),traceback=traceback.format_exc()));raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ['source','out','selector-config','cache-source','history-source']:p.add_argument('--'+arg,type=Path,required=True)
    a=p.parse_args();run(a.source,a.out,a.selector_config,Path(__file__).resolve().parents[1],cache_source=a.cache_source,history_source=a.history_source)
