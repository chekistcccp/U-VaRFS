"""Fresh fitting on cached normal data, followed by frozen outer normal audit."""
from pathlib import Path
import argparse,json,os,time
import torch
import torch.nn.functional as F
from scripts.run_normal_tail_pilot import project_path,validate_normal_manifest,sha256
from uvarfs.mechanism_validation import VERSION,SETTINGS,partition,fit_design,jsonable,expected_methods
from uvarfs.pipeline_eval import transform
from uvarfs.utils import load_config,save_json


@torch.no_grad()
def audit_cached(source,training,specs,layers,temperature,device):
    """Outer features are accessed only after all representations are frozen."""
    audit={l:source['heldout'][l].to(device) for l in layers}
    reference={l:training[l] for l in layers}
    raw=dict(layers=layers,layer_normalization='l2',kind='raw')
    # The existing transform handles B,P,D; each row is a one-patch pseudo-batch.
    prepare=lambda f:{l:v[:,None,:] for l,v in f.items()}
    qa,ra=prepare(audit),prepare(reference)
    denseq=F.normalize(transform(qa,raw).squeeze(1).double(),dim=1);denser=F.normalize(transform(ra,raw).squeeze(1).double(),dim=1)
    dense=denseq@denser.T;logt=(dense/temperature).log_softmax(1);t=logt.exp();target=1-dense.max(1).values
    groups=source['heldout_groups'].to(device);records=[];images={}
    names={n for n in specs if n.startswith(('mechanism_','random_weighted_','random_uniform_','random_beta_zero_','dense_')) or n in {'asls_raw','asls_pca'}}
    for name in sorted(names):
        spec=specs[name];q=F.normalize(transform(qa,spec).squeeze(1).double(),dim=1);r=F.normalize(transform(ra,spec).squeeze(1).double(),dim=1)
        cosine=q@r.T;scores=1-cosine.max(1).values;kl=(t*(logt-(cosine/temperature).log_softmax(1))).sum(1).mean()
        perimage=[];rawimage=[]
        for g in groups.unique(sorted=True):
            perimage.append(float(scores[groups==g].max()));rawimage.append(float(target[groups==g].max()))
        records.append({'method':name,'normal_patch_score_mse':float((scores-target).square().mean()),
                        'normal_neighbor_kl':float(kl),'sampled_image_score_mse':sum((a-b)**2 for a,b in zip(perimage,rawimage))/len(perimage)})
        images[name]=perimage
    return dict(stage='frozen_representation_outer_audit',image_groups=groups.unique(sorted=True).tolist(),
                records=records,image_scores=images,audit_query_rows=len(denseq),fit_reference_rows=len(denser),
                sampled_patch_top1_percent_is_max=True,used_for_selection=False,patient_disjoint_claimed=False)


def run(cache_path,manifest_path,base_cfg,settings_path,output,root,device='cpu'):
    root=Path(root).resolve();output=project_path(output,root,output=True)
    cache_path,manifest_path,settings_path=[project_path(p,root) for p in [cache_path,manifest_path,settings_path]]
    if load_config(settings_path)!=SETTINGS:raise ValueError('fixed mechanism settings required')
    if device not in {'cpu','cuda'} or (device=='cuda' and (not torch.cuda.is_available() or os.environ.get('CUDA_VISIBLE_DEVICES') not in {'0','1','2','3'})):
        raise ValueError('one authorized device required')
    sources={p.relative_to(root).as_posix():sha256(p) for p in [cache_path,manifest_path,settings_path]}
    cached=torch.load(cache_path,map_location='cpu',weights_only=True)['l2']
    validate_normal_manifest(cached['manifest'],json.loads(manifest_path.read_text()))
    original=cached['manifest']['normal_selection_split']
    if cached['groups'].tolist()!=original['fit_patch_groups'] or cached['heldout_groups'].tolist()!=original['holdout_patch_groups']:
        raise ValueError('original cached image groups differ')
    fitmask,selectmask,split=partition(cached,int(base_cfg['seed']))
    layers=sorted(cached['patches'])
    for l in layers:
        if not torch.equal(cached['patches'][l][:len(cached['paired_clean'][l])],cached['paired_clean'][l]):
            raise ValueError('paired clean image/patch alignment differs')
    training={l:cached['patches'][l][fitmask].to(device) for l in layers}
    selection={l:cached['patches'][l][selectmask].to(device) for l in layers}
    paired=({l:cached['paired_clean'][l].to(device) for l in layers},
            {l:cached['paired_perturbations'][l].to(device) for l in layers})
    started=time.perf_counter();result=fit_design(training,selection,cached['groups'][fitmask].to(device),
        cached['groups'][selectmask].to(device),paired,base_cfg,SETTINGS,int(base_cfg['seed']))
    if set(result['specs'])!=expected_methods()-{'v22_anchor','v22_beta_zero_anchor','pursuit_anchor','pursuit_beta_zero_anchor'}:
        raise ValueError('fixed 31 fresh representations required')
    partial=output.with_name(output.name+'.partial');partial.mkdir(parents=True)
    save_json(result['specs'],partial/'representation-specs.json');spec_sha=sha256(partial/'representation-specs.json')
    save_json(result['asls'],partial/'asls.json');save_json(split,partial/'normal-role-split.json')
    for branch,data in result['branches'].items():save_json(data,partial/(branch+'-fit.json'))
    audit=audit_cached(cached,training,result['specs'],result['asls']['selected_layers'],
                       result['branches']['nuisance']['temperature_fit_only'],device)
    save_json(audit,partial/'normal-cache-audit.json')
    if sha256(partial/'representation-specs.json')!=spec_sha:raise ValueError('outer audit changed representations')
    report={'evaluation_version':VERSION,'stage':'fresh_fit_and_frozen_normal_audit_complete',
        'new_fit_seconds':result['new_fit_seconds'],'fit_and_cached_audit_seconds':time.perf_counter()-started,
        'normal_roles':split,'source_sha256':sources,'representations_frozen_before_audit_sha256':spec_sha,
        'labels_or_masks_used':False,'registered_main_replaced':False,'new_independent_normal_sampling_claimed':False}
    for rel,h in sources.items():
        if sha256(root/rel)!=h:raise ValueError('original normal source changed')
    save_json(report,partial/'report.json');save_json({p.name:{'sha256':sha256(p),'bytes':p.stat().st_size} for p in partial.iterdir()},partial/'files-sha256.json')
    partial.replace(output);return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['cache','manifest','base-config','config','out']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu');a=p.parse_args()
    run(a.cache,a.manifest,load_config(a.base_config),a.config,a.out,Path(__file__).resolve().parents[1],a.device)
