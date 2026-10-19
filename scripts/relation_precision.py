"""Normal-only fixed-support continuation; detector phase requires a global gate."""
from __future__ import annotations
import argparse,copy,json,os,subprocess,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from uvarfs.relation_precision import VERSION,SETTINGS,weighted_names,methods,ANCHORS,continue_parents,audit_record,frozen_parent,frozen_graph
from uvarfs.relation_selection import RelationObjective
from uvarfs.mechanism_validation import partition,jsonable,artifact_digest
from uvarfs.protocol import code_fingerprint,data_fingerprint
from uvarfs.utils import load_config,set_seed
from scripts.evaluate_mechanism_validation import full_normal_audit
from scripts.evaluate_normal_tail import atomic,read,sha256,METRICS
from scripts.run_normal_tail_pilot import project_path,validate_normal_manifest
from uvarfs.pipeline_eval import build_memories,evaluate
from uvarfs.data import discover_bmad_roots,scan_split
from uvarfs.dinov3 import DINOv3Extractor


def strict_fit(out,verify_files=True):
    out=Path(out);done=read(out/'normal-complete.json');meta=read(out/'run_metadata.json');fits=read(out/'normal-fit.json')
    if (out/'failure.json').exists() or done['evaluation_version']!=VERSION or meta['evaluation_version']!=VERSION or set(fits)!=set(weighted_names()):raise ValueError('incomplete fixed normal fits')
    for name,c in fits.items():
        audit_record(name,c)
        p=np.array(c['weights']);k=len(p);diag=c['diagnostics']
        if (len(set(c['active']))!=k or abs(p.sum()-1)>1e-10 or p.min()<1e-4/k-1e-12 or p.max()>min(1,4/k)+1e-12 or
            diag['support_changed'] or c['training_objective']>c['parent_training_objective']+1e-10 or
            diag['precision_pass']!=(diag['residual']<=SETTINGS['residual_tolerance'])):raise ValueError('normal precision feasibility or gate differs')
        if name in {'precision_sparse','precision_beta_zero'} and k!=256:raise ValueError('K256 differs')
        if 'beta_zero' in name and c['training_terms']['variability_term']!=0:raise ValueError('beta0 nuisance leaked')
    graph=read(out/'graph-reproduction.json')
    if not graph['both_parent_beta_graphs_exact'] or graph['graph']['nuisance_used'] or graph['graph']['same_image_edges']:
        raise ValueError('frozen clean graph protocol differs')
    specs=read(out/'representation-specs.json')
    if set(specs)!=set(weighted_names())|{'asls_raw'}:raise ValueError('normal representation roster differs')
    for name,c in fits.items():
        if specs[name]['obj']['active']!=c['active'] or not np.allclose(specs[name]['obj']['scales'],np.sqrt(c['weights']),atol=1e-12,rtol=1e-12):
            raise ValueError('normal weights do not match frozen representations')
    if fits['precision_sparse']['active']!=fits['precision_same_support_beta_zero']['active']:
        raise ValueError('same-support beta0 changed the Main support')
    if sha256(out/'representation-specs.json')!=done['frozen_representation_sha256']:raise ValueError('normal representations changed')
    if verify_files:
        recorded=read(out/'normal-files-sha256.json')
        if set(recorded)!={'normal-fit.json','representation-specs.json','normal-role-split.json','normal-source-sha256.json','run_metadata.json','fit-stage.json','graph-reproduction.json'}:raise ValueError('complete frozen normal artifact set required')
        for rel,r in recorded.items():
            p=(out/rel).resolve()
            if not p.is_relative_to(out.resolve()) or sha256(p)!=r['sha256'] or p.stat().st_size!=r['bytes']:raise ValueError('normal fit file changed')
    passed=sum(c['diagnostics']['precision_pass'] for c in fits.values())
    if passed!=done['precision_pass_count']:raise ValueError('normal pass count differs')
    return dict(fits=19,precision_pass_count=passed,all_precision_pass=passed==19,fit_seed=meta['fit_seed'],dataset=meta['dataset'])

def fit(parent,cache,config,out,project):
    project=Path(project).resolve();parent,cache,config=[project_path(p,project) for p in [parent,cache,config]];out=project_path(out,project,output=True)
    if load_config(config)!=SETTINGS or out.exists() or any(out.is_relative_to(p) for p in [parent,cache]):raise ValueError('fresh fixed normal output/config required')
    if not torch.cuda.is_available() or os.environ.get('CUDA_VISIBLE_DEVICES') not in {'0','1','2','3'}:raise ValueError('authorized physical GPU required')
    started=time.perf_counter()
    meta=read(parent/'run_metadata.json');cfg=meta['source_config'];seed=int(meta['fit_seed']);dataset=meta['dataset']
    if meta['evaluation_version']!='clean-relation-sparse-validation-v1':raise ValueError('frozen clean relation parent required')
    mechanism=project_path(meta['parent_relative'],project)
    if project_path(meta['cache_relative'],project)!=cache:raise ValueError('cache source pairing differs')
    normal_paths=[parent/'selection'/n for n in ['asls.json','representation-specs.json','normal-role-split.json','nuisance-fit.json','beta_zero-fit.json','same_support_beta_zero-fit.json']]
    normal_paths += [parent/'run_metadata.json',mechanism/'controls/fit_manifest.json',cache/'normal-inputs.pt',config]
    hashes={p.relative_to(project).as_posix():{'sha256':sha256(p),'bytes':p.stat().st_size} for p in normal_paths}
    if hashes[(cache/'normal-inputs.pt').relative_to(project).as_posix()]!=read(parent/'immutable-references.json')[(cache/'normal-inputs.pt').relative_to(project).as_posix()]:raise ValueError('frozen normal cache differs')
    out.mkdir(parents=True);atomic(out/'fit-stage.json',{'stage':'fixed_support_relation_fitting','updated_unix':time.time()})
    source=torch.load(cache/'normal-inputs.pt',map_location='cpu',weights_only=True)['l2'];validate_normal_manifest(source['manifest'],read(mechanism/'controls/fit_manifest.json'))
    mask,_,split=partition(source,seed)
    if split!=read(parent/'selection/normal-role-split.json'):raise ValueError('frozen normal roles differ')
    layers=read(parent/'selection/asls.json')['selected_layers'];join=lambda tree:torch.cat([tree[l] for l in layers],-1).to('cuda')
    x=join({l:source['patches'][l][mask] for l in layers});groups=source['groups'][mask].to('cuda')
    # No selection/audit values or support search are accessed in this phase.
    branches={b:read(parent/'selection'/(b+'-fit.json')) for b in ['nuisance','beta_zero']}
    graph,graphinfo=frozen_graph(x,groups,branches,seed)
    oldspecs=read(parent/'selection/representation-specs.json');fits={};specs={};set_seed(seed)
    same=read(parent/'selection/same_support_beta_zero-fit.json')
    for branch,beta in [('nuisance',.002),('beta_zero',0.)]:
        old=branches[branch];pairs=(join(source['paired_clean']),join(source['paired_perturbations'])) if beta else None
        engine=RelationObjective(x,x,groups,groups,old['temperature_fit_only'],graph,pairs,beta)
        parents={}
        for name,previous in weighted_names().items():
            if ('beta_zero' in name)!=(not bool(beta)):continue
            original=frozen_parent(old,previous,same);spec=oldspecs[previous]
            if spec['layers']!=layers or spec['obj']['active']!=original['active'] or not np.allclose(np.sqrt(original['weights']),spec['obj']['scales'],atol=1e-12,rtol=1e-12):raise ValueError('frozen parent support/weights differ')
            parents[name]=original
        result=continue_parents(engine,parents)
        for name,c in result.items():
            spec=copy.deepcopy(oldspecs[weighted_names()[name]]);spec['obj']['scales']=np.sqrt(c['weights']).tolist();specs[name]=spec
        fits.update(result)
    specs['asls_raw']=oldspecs['asls_raw'];atomic(out/'fit-stage.json',{'stage':'normal_fit_complete','updated_unix':time.time()})
    atomic(out/'normal-fit.json',jsonable(fits));atomic(out/'representation-specs.json',specs);atomic(out/'normal-role-split.json',split)
    atomic(out/'normal-source-sha256.json',hashes)
    atomic(out/'graph-reproduction.json',dict(graph=graphinfo,graph_weights_sha256=artifact_digest(graph),both_parent_beta_graphs_exact=True))
    atomic(out/'run_metadata.json',dict(evaluation_version=VERSION,fit_seed=seed,dataset=dataset,
        head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=project,text=True).strip(),code_fingerprint=code_fingerprint(project),
        parent_relative=parent.relative_to(project).as_posix(),cache_relative=cache.relative_to(project).as_posix(),
        normal_fit_seconds=time.perf_counter()-started,supports_asls_and_temperature_frozen=True,new_independent_normal_sampling_claimed=False,
        evaluation_before_global_gate=False,worker_sha256=sha256(__file__),settings_sha256=sha256(config)))
    for rel,r in hashes.items():
        if sha256(project/rel)!=r['sha256']:raise ValueError('immutable normal source changed')
    atomic(out/'normal-files-sha256.json',{p.name:{'sha256':sha256(p),'bytes':p.stat().st_size} for p in out.iterdir() if p.is_file()})
    atomic(out/'normal-complete.json',dict(evaluation_version=VERSION,fits=19,precision_pass_count=sum(c['diagnostics']['precision_pass'] for c in fits.values()),
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
    if gate['evaluation_version']!=VERSION or not gate['all_342_pass'] or gate['blocks']!=18 or gate['precision_pass_count']!=342:raise ValueError('full normal precision gate not passed')
    if len(gate['normal_fit_sha256'])!=90:raise ValueError('all 18 frozen normal artifact groups required')
    expected={(f'seed{s}',d,n) for s in [42,123,3407] for d in ['brain','liver','resc','oct2017','xray','camelyon16'] for n in ['normal-fit.json','representation-specs.json','normal-complete.json','normal-files-sha256.json','normal-role-split.json']}
    paths=[Path(rel) for rel in gate['normal_fit_sha256']]
    if {tuple(p.parts[-3:]) for p in paths}!=expected or len({p.parent.parent.parent.parent for p in paths})!=1:raise ValueError('all 18 artifact identities must match one batch')
    for rel,h in gate['normal_fit_sha256'].items():
        path=(project/rel).resolve()
        if not path.is_relative_to((project/'results').resolve()) or sha256(path)!=h:raise ValueError('globally frozen normal fit differs')
    for rel in gate['normal_fit_sha256']:
        path=project/rel
        if path.name=='normal-complete.json' and not strict_fit(path.parent)['all_precision_pass']:
            raise ValueError('globally frozen normal fits do not all pass')

def evaluate_block(out,gatepath,project):
    project=Path(project).resolve();out=project_path(out,project);gatepath=project_path(gatepath,project)
    gate=read(gatepath);validate_global_gate(gate,project);strict_fit(out)
    if gatepath.parent!=out.parents[2] or (out/'normal-fit.json').relative_to(project).as_posix() not in gate['normal_fit_sha256']:raise ValueError('normal gate belongs to another batch')
    if (out/'evaluation-complete.json').exists() or (out/'eval_progress.json').exists():raise ValueError('duplicate evaluation refused')
    meta=read(out/'run_metadata.json');parent=project/meta['parent_relative'];pm=read(parent/'run_metadata.json');cfg=pm['source_config'];seed=meta['fit_seed'];dataset=meta['dataset']
    from scripts.evaluate_relation_validation import strict_complete as parent_complete
    parent_complete(parent)
    specs=read(out/'representation-specs.json');old=read(parent/'selection/representation-specs.json')
    specs.update(previous_sparse_anchor=old['relation_sparse'],previous_dense_anchor=old['dense_weighted'])
    if set(specs)!=methods():raise ValueError('fixed 22 methods required')
    atomic(out/'evaluation-specs.json',specs);frozen=sha256(out/'evaluation-specs.json');started=time.perf_counter();set_seed(seed);torch.cuda.reset_peak_memory_stats()
    roots=discover_bmad_roots(cfg['paths']['processed_data']);train=scan_split(roots[dataset],'train',dataset)
    extractor=DINOv3Extractor(cfg['paths']['model_dir'],int(cfg['model']['input_size']),cfg['model']['amp'])
    split=read(out/'normal-role-split.json');manifest=read(project/read(parent/'run_metadata.json')['parent_relative']/'controls/fit_manifest.json');queries=[manifest['normal_train_paths'][i] for i in split['audit_image_positions']]
    full_normal_audit(extractor,train,specs,cfg,queries,out/'normal-full-audit',methods_selector=lambda ss:{k:v for k,v in ss.items() if k in weighted_names() or k=='asls_raw'})
    test=scan_split(roots[dataset],'test',dataset);original=read(project/read(parent/'run_metadata.json')['parent_relative']/'controls/run_metadata.json')
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
    atomic(out/'evaluation-complete.json',dict(evaluation_version=VERSION,methods=22,anchors_exact=True,global_gate_sha256=sha256(gatepath)))
    strict_evaluation(out)

def strict_evaluation(out):
    out=Path(out);strict_fit(out);done=read(out/'evaluation-complete.json');meta=read(out/'run_metadata.json');f=pd.read_csv(out/'metrics.csv')
    metrics=METRICS if meta['dataset'] in {'brain','liver','resc'} else METRICS[:2]
    if done['evaluation_version']!=VERSION or not done['anchors_exact'] or len(f)!=22 or set(f.method)!=methods() or not np.isfinite(f[metrics]).all().all() or not f[metrics].apply(lambda c:c.between(0,1)).all().all():raise ValueError('incomplete fixed detector evaluation')
    weighted=f[f.method.isin(weighted_names())];dense=f[f.method.isin(['precision_dense_weighted','precision_dense_beta_zero'])]
    if not weighted[~weighted.method.str.contains('dense_')].feature_dim.eq(256).all() or not dense.feature_dim.eq(int(f.loc[f.method.eq('asls_raw'),'feature_dim'].iloc[0])).all() or not f.memory_size.eq(20000).all():raise ValueError('fixed feature or memory budget differs')
    if sha256(out.parents[2]/'global-normal-gate.json')!=done['global_gate_sha256']:raise ValueError('global gate changed after evaluation')
    for rel,r in read(out/'evaluation-files-sha256.json').items():
        p=(out/rel).resolve()
        if not p.is_relative_to(out.resolve()) or sha256(p)!=r['sha256'] or p.stat().st_size!=r['bytes']:raise ValueError('evaluation artifact changed')
    return dict(rows=22,anchors_exact=True,fit_seed=meta['fit_seed'],dataset=meta['dataset'])

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',choices=['fit','evaluate'],required=True)
    for arg in ['parent','cache','config','out','gate']:p.add_argument('--'+arg,type=Path)
    a=p.parse_args();project=Path(__file__).resolve().parents[1]
    # The coordinator owns failure/exit logs. Never write to an input or a
    # duplicate output merely because argument/path validation failed.
    if a.phase=='fit':fit(a.parent,a.cache,a.config,a.out,project)
    else:evaluate_block(a.out,a.gate,project)
