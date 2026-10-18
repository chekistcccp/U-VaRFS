"""Fixed normal-only diagnostics of closed precision endpoints; never refit."""
from __future__ import annotations
import argparse,json,os,subprocess,time
from pathlib import Path
import numpy as np
import torch
from scripts.fixed_support_precision import read,atomic,sha256,strict_fit
from scripts.run_normal_tail_pilot import project_path,validate_normal_manifest
from uvarfs.endpoint_diagnostics import VERSION,SETTINGS,METHODS,diagnose_endpoint
from uvarfs.matching_uvarfs import MatchingObjective
from uvarfs.mechanism_validation import partition,artifact_digest
from uvarfs.protocol import code_fingerprint
from uvarfs.utils import load_config

def strict_complete(out):
    out=Path(out);done=read(out/'complete.json');meta=read(out/'run_metadata.json')
    if done['evaluation_version']!=VERSION or done['endpoints']!=5 or meta['source_seed']!=42 or done['optimizer_run'] or done['test_data_used']:raise ValueError('incomplete endpoint block')
    records=read(out/'files-sha256.json')
    expected={'run_metadata.json','source-sha256.json','stage.json'}|{n+s for n in METHODS for s in ['.json','.npz']}
    if set(records)!=expected:raise ValueError('endpoint artifact set differs')
    if set(p.name for p in out.iterdir())!=expected|{'files-sha256.json','complete.json'} or any(not p.is_file() or p.is_symlink() for p in out.iterdir()):raise ValueError('unlisted or linked endpoint artifact')
    for rel,r in records.items():
        p=(out/rel).resolve()
        if not p.is_relative_to(out.resolve()) or sha256(p)!=r['sha256'] or p.stat().st_size!=r['bytes']:raise ValueError('endpoint file changed')
    for name in METHODS:
        r=read(out/(name+'.json'))
        if r['version']!=VERSION or r['source_weights_modified'] or r['optimizer_run'] or r['test_data_used'] or r['source_method']!=name:raise ValueError('endpoint identity differs')
    return {'source_seed':42,'dataset':meta['dataset'],'endpoints':5,'optimizer_run':False,'test_data_used':False}

def run(source,out,config,project,device='cuda'):
    project=Path(project).resolve();source=project_path(source,project);config=project_path(config,project);out=project_path(out,project,output=True)
    if out.exists() or out.is_relative_to(source) or load_config(config)!=SETTINGS:raise ValueError('fresh endpoint output/fixed config required')
    if device=='cuda' and (not torch.cuda.is_available() or os.environ.get('CUDA_VISIBLE_DEVICES') not in {'0','1','2','3'}):raise ValueError('authorized free GPU required')
    checked=strict_fit(source);meta=read(source/'run_metadata.json')
    if checked['fit_seed']!=42:raise ValueError('predefined source seed42 required')
    parent=project/meta['parent_relative'];cache=project/meta['cache_relative'];oldmeta=read(parent/'run_metadata.json');cfg=oldmeta['source_config'];fits=read(source/'normal-fit.json')
    hashes=read(source/'normal-source-sha256.json')
    for path in [source/n for n in ['normal-fit.json','normal-role-split.json','normal-files-sha256.json','run_metadata.json']]+[config]:
        hashes[path.relative_to(project).as_posix()]={'sha256':sha256(path),'bytes':path.stat().st_size}
    for rel,r in hashes.items():
        p=(project/rel).resolve()
        if not p.is_relative_to((project/'results').resolve()) and p!=config:raise ValueError('normal source outside results')
        if sha256(p)!=r['sha256'] or p.stat().st_size!=r['bytes']:raise ValueError('immutable source changed')
    data=torch.load(cache/'normal-inputs.pt',map_location='cpu',weights_only=True)['l2']
    validate_normal_manifest(data['manifest'],read(parent/'controls/fit_manifest.json'));mask,_,split=partition(data,42)
    if split!=read(source/'normal-role-split.json'):raise ValueError('normal fit roles differ')
    layers=read(parent/'selection/asls.json')['selected_layers'];join=lambda tree:torch.cat([tree[l] for l in layers],-1).to(device)
    x=join({l:data['patches'][l][mask] for l in layers});groups=data['groups'][mask].to(device)
    out.mkdir(parents=True);started=time.perf_counter()
    for branch,beta in [('nuisance',.002),('beta_zero',0.)]:
        old=read(parent/'selection'/(branch+'-fit.json'))
        pairs=(join(data['paired_clean']),join(data['paired_perturbations'])) if beta else None
        u=cfg['uvarfs'];engine=MatchingObjective(x,x,groups,groups,old['temperature_fit_only'],pairs,beta,
            int(u.get('matching_chunk',256)),float(u.get('matching_score_weight',1.)),float(u.get('matching_distribution_weight',1.)),old['floor_mass'],old['cap_factor'])
        for name in METHODS:
            if ('beta_zero' in name)!=(beta==0):continue
            c=fits[name]
            if name=='precision_sparse' or name=='precision_beta_zero':previous=old['learned'][old['winner_index']]
            elif 'random_' in name:previous=old['random']['42']
            else:previous=old['dense']
            if c['active']!=previous['active'] or c['parent_weights_sha256']!=artifact_digest(previous['weights']):raise ValueError('parent support or initial weights differ')
            print(f'[endpoint] {checked["dataset"]}/{name} start',flush=True)
            atomic(out/'stage.json',{'stage':'normal_directional_probes','method':name,'updated_unix':time.time()})
            result,arrays=diagnose_endpoint(engine,c['active'],c['weights'],previous['weights'])
            if not np.isclose(result['loss'],c['training_objective'],rtol=1e-9,atol=1e-11) or not np.allclose(result['gradient_p'],c['final_gradient_p'],rtol=1e-8,atol=1e-10):raise ValueError('source endpoint objective/gradient does not reproduce')
            result.update(source_method=name,source_fit_sha256=sha256(source/'normal-fit.json'),beta=beta)
            atomic(out/(name+'.json'),result);np.savez_compressed(out/(name+'.npz'),**arrays)
            print(f'[endpoint] {name} done queries={result["normal_queries"]} ties={result["exact_tied_queries"]}',flush=True)
        del engine
        if device=='cuda':torch.cuda.empty_cache()
    for rel,r in hashes.items():
        if sha256(project/rel)!=r['sha256']:raise ValueError('normal diagnostic input changed')
    atomic(out/'source-sha256.json',hashes);atomic(out/'run_metadata.json',dict(evaluation_version=VERSION,source_seed=42,dataset=checked['dataset'],
      head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=project,text=True).strip(),code_fingerprint=code_fingerprint(project),
      source_relative=source.relative_to(project).as_posix(),worker_sha256=sha256(__file__),config_sha256=sha256(config),seconds=time.perf_counter()-started,
      supports_layers_temperature_and_weights_frozen=True,selection_and_audit_values_indexed=False,test_data_used=False,optimizer_run=False))
    atomic(out/'stage.json',{'stage':'complete','updated_unix':time.time()})
    atomic(out/'files-sha256.json',{p.name:{'sha256':sha256(p),'bytes':p.stat().st_size} for p in out.iterdir() if p.is_file()})
    atomic(out/'complete.json',{'evaluation_version':VERSION,'endpoints':5,'optimizer_run':False,'test_data_used':False});strict_complete(out)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['source','out','config']:p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();run(a.source,a.out,a.config,Path(__file__).resolve().parents[1])
