"""Normal-only fixed-support continuation; detector phase requires a global gate."""
from __future__ import annotations
import argparse,copy,json,os,subprocess,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from uvarfs.precision_refit import VERSION,SETTINGS,weighted_names,methods,refit_fixed_support
from uvarfs.mechanism_validation import partition,jsonable,artifact_digest
from uvarfs.matching_uvarfs import MatchingObjective
from uvarfs.protocol import code_fingerprint,data_fingerprint
from uvarfs.utils import load_config,set_seed
from scripts.evaluate_mechanism_validation import full_normal_audit
from scripts.evaluate_normal_tail import atomic,read,sha256,METRICS
from scripts.run_normal_tail_pilot import project_path,validate_normal_manifest
from uvarfs.pipeline_eval import build_memories,evaluate
from uvarfs.data import discover_bmad_roots,scan_split
from uvarfs.dinov3 import DINOv3Extractor

ANCHORS={'previous_sparse_anchor':'mechanism_sparse','previous_dense_anchor':'dense_weighted','asls_raw':'asls_raw'}

def continue_parents(engine,parents,settings=SETTINGS):
    """Parents contain only immutable normal fits. No other data arguments."""
    result={}
    for name,parent in parents.items():
        initial=torch.tensor(parent['weights'],device=engine.q.device,dtype=torch.float64)
        value,_,_=engine.evaluate(parent['active'],initial)
        if not np.isclose(value,parent['training_objective'],atol=1e-10,rtol=1e-9):raise ValueError('original normal objective does not reproduce')
        print(f'[precision:{name}] fixed support K={len(initial)} old_loss={value:.9g}',flush=True)
        p,v,terms,diag=refit_fixed_support(engine,parent['active'],initial,settings,print)
        verified,gradient,verified_terms=engine.evaluate(parent['active'],p,True)
        residual=float((p-engine.project(p-gradient)).abs().max())
        if not np.isclose(verified,v,atol=1e-12,rtol=1e-10) or not np.isclose(residual,diag['residual'],atol=1e-12,rtol=1e-9):raise ValueError('independent final gradient/loss verification failed')
        result[name]=dict(active=parent['active'],weights=p.cpu().tolist(),training_objective=v,
            training_terms=terms,diagnostics=diag,parent_training_objective=value,parent_residual=parent['diagnostics']['residual'],
            parent_weights_sha256=artifact_digest(parent['weights']),final_gradient_p=gradient.cpu().tolist(),final_verification_calls=1)
        print(f'[precision:{name}] done loss={v:.9g} residual={diag["residual"]:.6g} pass={diag["precision_pass"]}',flush=True)
    return result

def strict_fit(out,verify_files=True):
    out=Path(out);done=read(out/'normal-complete.json');meta=read(out/'run_metadata.json');fits=read(out/'normal-fit.json')
    if (out/'failure.json').exists() or done['evaluation_version']!=VERSION or meta['evaluation_version']!=VERSION or set(fits)!=set(weighted_names()):raise ValueError('incomplete fixed normal fits')
    for name,c in fits.items():
        p=np.array(c['weights']);k=len(p);diag=c['diagnostics']
        if (len(set(c['active']))!=k or abs(p.sum()-1)>1e-10 or p.min()<1e-4/k-1e-12 or p.max()>min(1,4/k)+1e-12 or
            diag['support_changed'] or c['training_objective']>c['parent_training_objective']+1e-10 or
            diag['precision_pass']!=(diag['residual']<=SETTINGS['residual_tolerance'])):raise ValueError('normal precision feasibility or gate differs')
        if name in {'precision_sparse','precision_beta_zero'} and k!=256:raise ValueError('K256 differs')
        if 'beta_zero' in name and c['training_terms']['variability_term']!=0:raise ValueError('beta0 nuisance leaked')
    if sha256(out/'representation-specs.json')!=done['frozen_representation_sha256']:raise ValueError('normal representations changed')
    if verify_files:
        recorded=read(out/'normal-files-sha256.json')
        if set(recorded)!={'normal-fit.json','representation-specs.json','normal-role-split.json','normal-source-sha256.json','run_metadata.json','fit-stage.json'}:raise ValueError('complete frozen normal artifact set required')
        for rel,r in recorded.items():
            p=(out/rel).resolve()
            if not p.is_relative_to(out.resolve()) or sha256(p)!=r['sha256'] or p.stat().st_size!=r['bytes']:raise ValueError('normal fit file changed')
    passed=sum(c['diagnostics']['precision_pass'] for c in fits.values())
    if passed!=done['precision_pass_count']:raise ValueError('normal pass count differs')
    return dict(fits=14,precision_pass_count=passed,all_precision_pass=passed==14,fit_seed=meta['fit_seed'],dataset=meta['dataset'])

def fit(parent,cache,config,out,project):
    project=Path(project).resolve();parent,cache,config=[project_path(p,project) for p in [parent,cache,config]];out=project_path(out,project,output=True)
    if load_config(config)!=SETTINGS or out.exists() or any(out.is_relative_to(p) for p in [parent,cache]):raise ValueError('fresh fixed normal output/config required')
    if not torch.cuda.is_available() or os.environ.get('CUDA_VISIBLE_DEVICES') not in {'0','1','2','3'}:raise ValueError('authorized physical GPU required')
    meta=read(parent/'run_metadata.json');cfg=meta['source_config'];seed=int(meta['fit_seed']);dataset=meta['dataset']
    if meta['evaluation_version']!='mechanism-factorial-audit-v1':raise ValueError('frozen mechanism parent required')
    normal_paths=[parent/'selection'/n for n in ['asls.json','representation-specs.json','normal-role-split.json','nuisance-fit.json','beta_zero-fit.json']]
    normal_paths += [parent/'run_metadata.json',parent/'controls/fit_manifest.json',cache/'normal-inputs.pt',config]
    hashes={p.relative_to(project).as_posix():{'sha256':sha256(p),'bytes':p.stat().st_size} for p in normal_paths}
    if hashes[(cache/'normal-inputs.pt').relative_to(project).as_posix()]['sha256']!=read(parent/'immutable-references.json')[(cache/'normal-inputs.pt').relative_to(project).as_posix()]['sha256']:raise ValueError('frozen normal cache differs')
    out.mkdir(parents=True);atomic(out/'fit-stage.json',{'stage':'fixed_support_normal_fitting','updated_unix':time.time()})
    source=torch.load(cache/'normal-inputs.pt',map_location='cpu',weights_only=True)['l2'];validate_normal_manifest(source['manifest'],read(parent/'controls/fit_manifest.json'))
    mask,_,split=partition(source,seed)
    if split!=read(parent/'selection/normal-role-split.json'):raise ValueError('frozen normal roles differ')
    asls=read(parent/'selection/asls.json');layers=asls['selected_layers'];join=lambda tree:torch.cat([tree[l] for l in layers],-1).to('cuda')
    x=join({l:source['patches'][l][mask] for l in layers});groups=source['groups'][mask].to('cuda')
    # Selection and audit feature values are never indexed in this phase.
    oldspecs=read(parent/'selection/representation-specs.json');fits={};specs={};started=time.perf_counter();set_seed(seed)
    for branch,beta in [('nuisance',.002),('beta_zero',0.)]:
        old=read(parent/'selection'/(branch+'-fit.json'));pairs=(join(source['paired_clean']),join(source['paired_perturbations'])) if beta else None
        u=cfg['uvarfs'];engine=MatchingObjective(x,x,groups,groups,old['temperature_fit_only'],pairs,beta,
            int(u.get('matching_chunk',256)),float(u.get('matching_score_weight',1.)),float(u.get('matching_distribution_weight',1.)),old['floor_mass'],old['cap_factor'])
        parents={}
        for name,previous in weighted_names().items():
            zero=('beta_zero' in name)
            if zero!=(not bool(beta)):continue
            if 'random_' in name:original=old['random'][str(int(name.rsplit('seed',1)[1]))]
            elif 'dense_' in name:original=old['dense']
            else:original=old['learned'][old['winner_index']]
            spec=oldspecs[previous]
            if spec['obj']['active']!=original['active'] or not np.allclose(np.sqrt(original['weights']),spec['obj']['scales'],atol=1e-12,rtol=1e-12):raise ValueError('frozen parent support/weights differ')
            parents[name]=original
        result=continue_parents(engine,parents)
        for name,c in result.items():
            previous=weighted_names()[name];spec=copy.deepcopy(oldspecs[previous]);spec['obj']['scales']=np.sqrt(c['weights']).tolist();specs[name]=spec
        fits.update(result)
    specs['asls_raw']=oldspecs['asls_raw'];atomic(out/'fit-stage.json',{'stage':'normal_fit_complete','updated_unix':time.time()})
    atomic(out/'normal-fit.json',jsonable(fits));atomic(out/'representation-specs.json',specs);atomic(out/'normal-role-split.json',split)
    atomic(out/'normal-source-sha256.json',hashes)
    atomic(out/'run_metadata.json',dict(evaluation_version=VERSION,fit_seed=seed,dataset=dataset,
        head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=project,text=True).strip(),code_fingerprint=code_fingerprint(project),
        parent_relative=parent.relative_to(project).as_posix(),cache_relative=cache.relative_to(project).as_posix(),
        normal_fit_seconds=time.perf_counter()-started,supports_asls_and_temperature_frozen=True,new_independent_normal_sampling_claimed=False,
        evaluation_before_global_gate=False,worker_sha256=sha256(__file__),settings_sha256=sha256(config)))
    for rel,r in hashes.items():
        if sha256(project/rel)!=r['sha256']:raise ValueError('immutable normal source changed')
    atomic(out/'normal-files-sha256.json',{p.name:{'sha256':sha256(p),'bytes':p.stat().st_size} for p in out.iterdir() if p.is_file()})
    atomic(out/'normal-complete.json',dict(evaluation_version=VERSION,fits=14,precision_pass_count=sum(c['diagnostics']['precision_pass'] for c in fits.values()),
        frozen_representation_sha256=sha256(out/'representation-specs.json'),normal_only=True))
    strict_fit(out)

def validate_anchors(pred,oldpred,metrics,oldmetrics,memory,oldmemory):
    for key in ['normal_train_indices','normal_train_paths','patch_batches','memory_capacity','reservoir_seed','shared_sampling_across_methods']:
        if memory.get(key)!=oldmemory.get(key):raise ValueError('original shared memory differs: '+key)
    if set(memory['memory_rows'].values())!={20000} or set(oldmemory['memory_rows'].values())!={20000}:raise ValueError('memory budget differs')
    for col in ['image_path','label','mask_path']:
        if not pred[col].fillna('').equals(oldpred[col].fillna('')):raise ValueError('evaluation roster differs')
    a,b=metrics.set_index('method'),oldmetrics.set_index('method')
    for new,old in ANCHORS.items():
        valid=[m for m in METRICS if m in b and pd.notna(b.loc[old,m])]
        if not np.array_equal(pred[new],oldpred[old]) or not np.allclose(a.loc[new,valid].astype(float),b.loc[old,valid].astype(float),atol=1e-12,rtol=0):raise ValueError('historical anchor differs: '+new)
    return dict(anchors=ANCHORS,predictions_exact=True,metric_tolerance=1e-12)

def validate_global_gate(gate,project):
    if gate['evaluation_version']!=VERSION or not gate['all_252_pass'] or gate['blocks']!=18 or gate['precision_pass_count']!=252:raise ValueError('full normal precision gate not passed')
    if len(gate['normal_fit_sha256'])!=90:raise ValueError('all 18 frozen normal artifact groups required')
    expected={(f'seed{s}',d,n) for s in [42,123,3407] for d in ['brain','liver','resc','oct2017','xray','camelyon16'] for n in ['normal-fit.json','representation-specs.json','normal-complete.json','normal-files-sha256.json','normal-role-split.json']}
    paths=[Path(rel) for rel in gate['normal_fit_sha256']]
    if {tuple(p.parts[-3:]) for p in paths}!=expected or len({p.parent.parent.parent.parent for p in paths})!=1:raise ValueError('all 18 artifact identities must match one batch')
    for rel,h in gate['normal_fit_sha256'].items():
        path=(project/rel).resolve()
        if not path.is_relative_to((project/'results').resolve()) or sha256(path)!=h:raise ValueError('globally frozen normal fit differs')

def evaluate_block(out,gatepath,project):
    project=Path(project).resolve();out=project_path(out,project);gatepath=project_path(gatepath,project)
    gate=read(gatepath);validate_global_gate(gate,project);strict_fit(out)
    if gatepath.parent!=out.parents[2] or (out/'normal-fit.json').relative_to(project).as_posix() not in gate['normal_fit_sha256']:raise ValueError('normal gate belongs to another batch')
    if (out/'evaluation-complete.json').exists() or (out/'eval_progress.json').exists():raise ValueError('duplicate evaluation refused')
    meta=read(out/'run_metadata.json');parent=project/meta['parent_relative'];pm=read(parent/'run_metadata.json');cfg=pm['source_config'];seed=meta['fit_seed'];dataset=meta['dataset']
    from scripts.evaluate_mechanism_validation import strict_complete as parent_complete
    parent_complete(parent)
    specs=read(out/'representation-specs.json');old=read(parent/'selection/representation-specs.json')
    specs.update(previous_sparse_anchor=old['mechanism_sparse'],previous_dense_anchor=old['dense_weighted'])
    if set(specs)!=methods():raise ValueError('fixed 17 methods required')
    atomic(out/'evaluation-specs.json',specs);frozen=sha256(out/'evaluation-specs.json');started=time.perf_counter();set_seed(seed);torch.cuda.reset_peak_memory_stats()
    roots=discover_bmad_roots(cfg['paths']['processed_data']);train=scan_split(roots[dataset],'train',dataset)
    extractor=DINOv3Extractor(cfg['paths']['model_dir'],int(cfg['model']['input_size']),cfg['model']['amp'])
    split=read(out/'normal-role-split.json');manifest=read(parent/'controls/fit_manifest.json');queries=[manifest['normal_train_paths'][i] for i in split['audit_image_positions']]
    full_normal_audit(extractor,train,specs,cfg,queries,out/'normal-full-audit',methods_selector=lambda ss:{k:v for k,v in ss.items() if k in weighted_names() or k=='asls_raw'})
    test=scan_split(roots[dataset],'test',dataset);original=read(parent/'controls/run_metadata.json')
    if data_fingerprint(train,test)!=original['data_fingerprint']:raise ValueError('official data changed')
    set_seed(seed);memories=build_memories(extractor,train,specs,cfg,out/'memory_progress.json')
    fresh=pd.DataFrame(evaluate(extractor,test,specs,memories,cfg,out/'eval_progress.json'))
    pred=pd.read_csv(out/'image_predictions.csv',float_precision='round_trip')
    anchors=validate_anchors(pred,pd.read_csv(parent/'image_predictions.csv',float_precision='round_trip'),fresh,pd.read_csv(parent/'metrics.csv'),read(out/'memory_manifest.json'),read(parent/'memory_manifest.json'))
    fresh['dataset']=dataset;fresh['fit_seed']=seed;fresh['evaluation_version']=VERSION
    fresh['feature_dim']=fresh.method.map(lambda n:len(specs[n]['obj']['active']) if 'obj' in specs[n] else len(specs[n]['layers'])*extractor.hidden_dim)
    fresh['memory_size']=20000;fresh.to_csv(out/'metrics.csv',index=False)
    atomic(out/'anchor-reproduction.json',anchors);atomic(out/'evaluation-costs.json',dict(normal_audit_and_evaluation_seconds=time.perf_counter()-started,peak_gpu_gb=torch.cuda.max_memory_allocated()/1024**3))
    if frozen!=sha256(out/'evaluation-specs.json'):raise ValueError('fixed representations changed by audit/test')
    atomic(out/'evaluation-files-sha256.json',{p.relative_to(out).as_posix():{'sha256':sha256(p),'bytes':p.stat().st_size} for p in out.rglob('*') if p.is_file()})
    atomic(out/'evaluation-complete.json',dict(evaluation_version=VERSION,methods=17,anchors_exact=True,global_gate_sha256=sha256(gatepath)))
    strict_evaluation(out)

def strict_evaluation(out):
    out=Path(out);strict_fit(out);done=read(out/'evaluation-complete.json');meta=read(out/'run_metadata.json');f=pd.read_csv(out/'metrics.csv')
    metrics=METRICS if meta['dataset'] in {'brain','liver','resc'} else METRICS[:2]
    if done['evaluation_version']!=VERSION or not done['anchors_exact'] or len(f)!=17 or set(f.method)!=methods() or not np.isfinite(f[metrics]).all().all() or not f[metrics].apply(lambda c:c.between(0,1)).all().all():raise ValueError('incomplete fixed detector evaluation')
    weighted=f[f.method.isin(weighted_names())];dense=f[f.method.isin(['precision_dense_weighted','precision_dense_beta_zero'])]
    if not weighted[~weighted.method.str.contains('dense_')].feature_dim.eq(256).all() or not dense.feature_dim.eq(int(f.loc[f.method.eq('asls_raw'),'feature_dim'].iloc[0])).all() or not f.memory_size.eq(20000).all():raise ValueError('fixed feature or memory budget differs')
    if sha256(out.parents[2]/'global-normal-gate.json')!=done['global_gate_sha256']:raise ValueError('global gate changed after evaluation')
    for rel,r in read(out/'evaluation-files-sha256.json').items():
        p=(out/rel).resolve()
        if not p.is_relative_to(out.resolve()) or sha256(p)!=r['sha256'] or p.stat().st_size!=r['bytes']:raise ValueError('evaluation artifact changed')
    return dict(rows=17,anchors_exact=True,fit_seed=meta['fit_seed'],dataset=meta['dataset'])

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',choices=['fit','evaluate'],required=True)
    for arg in ['parent','cache','config','out','gate']:p.add_argument('--'+arg,type=Path)
    a=p.parse_args();project=Path(__file__).resolve().parents[1]
    # The coordinator owns failure/exit logs. Never write to an input or a
    # duplicate output merely because argument/path validation failed.
    if a.phase=='fit':fit(a.parent,a.cache,a.config,a.out,project)
    else:evaluate_block(a.out,a.gate,project)
