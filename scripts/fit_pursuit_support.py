"""Fit coordinate extensions using an immutable normal cache; no test table reads."""
from __future__ import annotations
import argparse,json,os,time
from pathlib import Path
import torch
from scripts.run_normal_tail_pilot import project_path,validate_normal_manifest,sha256
from uvarfs.pursuit_support import VERSION,fit_pursuit_extension
from uvarfs.utils import load_config,save_json

EXPECTED={'extension_version':VERSION,'feature_budget':256,'solver':{
    'chunk':256,'starts':2,'rounds':4,'breakpoint_quantiles':[0.,.25,.5,.75],
    'breakpoint_multiplier':1.05,'refit_max_iter':150,'tol':1e-5,'polish_max_iter':800,'polish_ftol':1e-12}}
METHODS={'pursuit_matching','pursuit_matching_uniform','pursuit_only','pursuit_only_uniform',
    'pursuit_beta_zero','pursuit_beta_zero_uniform','pursuit_only_beta_zero','pursuit_only_beta_zero_uniform',
    'v22_anchor','v22_beta_zero_anchor'}


def run(cache_path,solution_path,beta_zero_path,config_path,output,root,device='cpu'):
    root=Path(root).resolve()
    paths=[project_path(p,root) for p in [cache_path,solution_path,beta_zero_path,config_path]]
    cache_path,solution_path,beta_zero_path,config_path=paths
    output=project_path(output,root,output=True)
    if load_config(config_path)!=EXPECTED:raise ValueError('fixed pursuit configuration required')
    if device=='cuda':
        if os.environ.get('CUDA_VISIBLE_DEVICES') not in {'0','1','2','3'} or not torch.cuda.is_available():
            raise ValueError('one authorized physical GPU required')
    elif device!='cpu':raise ValueError('device must be cpu or cuda')
    solutions=[json.loads(p.read_text()) for p in [solution_path,beta_zero_path]]
    if solutions[0]['beta']<=0 or solutions[1]['beta']!=0:raise ValueError('paired nuisance and beta-zero required')
    keys=['layers','layer_normalization','temperature_fit_only','score_weight','distribution_weight',
        'weight_floor_mass','weight_cap_factor','fixed_feature_budget']
    if any(solutions[0][k]!=solutions[1][k] for k in keys):raise ValueError('paired layer/config mismatch')
    manifests=[p.parent/'fit_manifest.json' for p in [solution_path,beta_zero_path]]
    sources={str(p.relative_to(root)):sha256(p) for p in [*paths,*manifests]}
    cache=torch.load(cache_path,map_location='cpu',weights_only=True)
    mode=solutions[0]['layer_normalization'];source=cache[mode]
    for path in manifests:validate_normal_manifest(source['manifest'],json.loads(path.read_text()))
    layers=solutions[0]['layers']
    if not layers or len(layers)!=len(set(layers)):raise ValueError('unique ordered layers required')
    join=lambda field:torch.cat([source[field][l] for l in layers],-1).to(device)
    x,heldout=join('patches'),join('heldout')
    g,vg=source['groups'].to(device),source['heldout_groups'].to(device)
    split=source['manifest']['normal_selection_split']
    if g.cpu().tolist()!=split['fit_patch_groups'] or vg.cpu().tolist()!=split['holdout_patch_groups']:
        raise ValueError('normal cached group identities differ')
    if any(s['fixed_feature_budget']!=min(256,x.shape[1]) for s in solutions):raise ValueError('K256 source budget required')
    paired=(join('paired_clean'),join('paired_perturbations'))
    results={};specs={};started=time.perf_counter()
    partial=output.with_name(output.name+'.partial');partial.mkdir(parents=True)
    for label,solution in zip(['nuisance','beta_zero'],solutions):
        print(f'[pursuit] {label} start: old pool {len(solution["candidate_pool"])}; K={solution["fixed_feature_budget"]}',flush=True)
        result=fit_pursuit_extension(x,heldout,g,vg,paired if solution['beta'] else None,solution,EXPECTED['solver'],
            progress=lambda msg:print(msg,flush=True))
        save_json(result,partial/(label+'-fit.json'));results[label]={k:v for k,v in result.items() if k!='candidate_pool'}
        names=['pursuit_matching','pursuit_only'] if label=='nuisance' else ['pursuit_beta_zero','pursuit_only_beta_zero']
        for name,key in zip(names,['expanded','pursuit_only']):
            obj=result['representations'][key]
            specs[name]={'layers':layers,'layer_normalization':mode,'kind':'uvarfs','obj':obj}
            specs[name+'_uniform']={'layers':layers,'layer_normalization':mode,'kind':'selected_raw','obj':{'active':obj['active']}}
        anchor='v22_anchor' if label=='nuisance' else 'v22_beta_zero_anchor'
        specs[anchor]={'layers':layers,'layer_normalization':mode,'kind':'uvarfs',
            'obj':{'active':solution['active'],'scales':solution['scales']}}
    if set(specs)!=METHODS:raise ValueError('fixed ten representations required')
    for rel,h in sources.items():
        if sha256(root/rel)!=h:raise ValueError('original normal source changed')
    save_json(specs,partial/'representation-specs.json')
    report={'extension_version':VERSION,'stage':'normal_fitting_complete','new_fit_seconds':time.perf_counter()-started,
        'labels_or_masks_used':False,'registered_main_replaced':False,'source_sha256':sources,'results':results,
        'normal_fit_rows':len(x),'normal_holdout_rows':len(heldout),'new_independent_sampling_seed_generated':False}
    save_json(report,partial/'report.json')
    save_json({p.name:{'sha256':sha256(p),'bytes':p.stat().st_size} for p in partial.iterdir() if p.is_file()},partial/'files-sha256.json')
    partial.replace(output);return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['cache','solution','beta-zero','out']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--config',type=Path,default=Path('configs/pursuit_support.yaml'))
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu');a=p.parse_args()
    run(a.cache,a.solution,a.beta_zero,a.config,a.out,Path(__file__).resolve().parents[1],a.device)
