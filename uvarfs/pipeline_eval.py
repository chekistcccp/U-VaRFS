from __future__ import annotations
import json, math, re, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from .data import BMADDataset
from .memory import Reservoir, make_index
from .metrics import safe_auc, safe_ap, bootstrap_auc, PixelAccumulator
from .transforms import concat_layers, apply_pca
from .u_varfs import apply_uvarfs


def loader(samples,cfg,with_mask=False,indices=None,test=False):
    ds=BMADDataset(samples,int(cfg['model']['input_size']),with_mask=with_mask)
    if indices is not None:
        ds=Subset(ds,list(map(int,indices)))
    bs=int(cfg.get('data',{}).get('test_batch_size',cfg['model']['batch_size'])) if test else int(cfg['model']['batch_size'])
    return DataLoader(
        ds,batch_size=bs,shuffle=False,
        num_workers=int(cfg['model']['num_workers']),
        pin_memory=True,
        persistent_workers=int(cfg['model']['num_workers'])>0,
    )


def transform(feats,spec,concat_cache=None):
    key=tuple(spec['layers'])
    if concat_cache is not None and key in concat_cache:
        x=concat_cache[key]
    else:
        x=concat_layers(feats,spec['layers'])
        if concat_cache is not None:
            concat_cache[key]=x
    kind=spec['kind']
    if kind=='uvarfs':
        x=apply_uvarfs(x,spec['obj'])
    elif kind=='pca':
        x=apply_pca(x.reshape(-1,x.shape[-1]),spec['obj']).reshape(x.shape[0],x.shape[1],-1)
    elif kind=='random':
        idx=torch.as_tensor(spec['obj'],device=x.device,dtype=torch.long)
        x=x.index_select(-1,idx)
    return F.normalize(x.float(),dim=-1)


def _sample_indices(n,k,seed):
    if k <= 0 or k >= n:
        return np.arange(n,dtype=np.int64)
    rng=np.random.default_rng(seed)
    return np.sort(rng.choice(n,size=k,replace=False))


def _sample_patch_ids(p,k,seed):
    if k <= 0 or k >= p:
        return np.arange(p,dtype=np.int64)
    rng=np.random.default_rng(seed)
    return np.sort(rng.choice(p,size=k,replace=False))


def _heartbeat(prefix,batch_i,total,t0,cfg,extra=None,progress_path=None):
    elapsed=max(time.time()-t0,1e-6)
    rate=batch_i/elapsed
    eta=(total-batch_i)/rate if rate>0 else float('inf')
    gpu=''
    if torch.cuda.is_available():
        alloc=torch.cuda.memory_allocated()/1024**3
        reserved=torch.cuda.memory_reserved()/1024**3
        gpu=f' gpu={alloc:.2f}/{reserved:.2f}GB'
    msg=f'[{prefix}] batch {batch_i}/{total} ({100*batch_i/max(total,1):.1f}%) elapsed={elapsed/60:.1f}m ETA={eta/60:.1f}m{gpu}'
    if extra:
        msg+=' '+extra
    print(msg,flush=True)
    if progress_path is not None:
        Path(progress_path).write_text(json.dumps({
            'stage':prefix,'batch':batch_i,'total_batches':total,
            'percent':100*batch_i/max(total,1),'elapsed_seconds':elapsed,
            'eta_seconds':eta,'updated_unix':time.time(),
            'extra':extra or '',
        },indent=2),encoding='utf-8')


def build_memories(extractor,train_samples,specs,cfg,progress_path=None):
    memory_size=int(cfg['data']['memory_size'])
    memory_images=int(cfg['data'].get('memory_images',1024))
    patches_per_image=int(cfg['data'].get('memory_patches_per_image',32))
    seed=int(cfg['seed'])
    image_ids=_sample_indices(len(train_samples),memory_images,seed+1009)
    memories={k:Reservoir(memory_size,seed+i) for i,k in enumerate(specs)}
    dl=loader(train_samples,cfg,indices=image_ids,test=False)
    total=len(dl); t0=time.time()
    heartbeat_n=max(1,int(cfg['eval'].get('heartbeat_batches',20)))
    print(f'[memory] using {len(image_ids)}/{len(train_samples)} normal images, <= {patches_per_image} patches/image, cap={memory_size}',flush=True)

    for bi,batch in enumerate(tqdm(dl,desc='memory',leave=False,mininterval=10),start=1):
        feats=extractor(batch['image'])
        p=next(iter(feats.values())).shape[1]
        patch_ids=_sample_patch_ids(p,patches_per_image,seed+bi)
        patch_idx=torch.as_tensor(patch_ids,device=next(iter(feats.values())).device,dtype=torch.long)
        cache={}
        for name,spec in specs.items():
            z=transform(feats,spec,cache)
            z=z.index_select(1,patch_idx)
            memories[name].add(z.reshape(-1,z.shape[-1]))
        if bi==1 or bi%heartbeat_n==0 or bi==total:
            _heartbeat('memory',bi,total,t0,cfg,extra=f'methods={len(specs)}',progress_path=progress_path)

    result={k:v.value() for k,v in memories.items()}
    print('[memory] built: '+', '.join(f'{k}={len(v)}' for k,v in result.items()),flush=True)
    return result


def evaluate(extractor,test_samples,specs,memories,cfg,progress_path=None):
    eval_cfg=cfg['eval']
    indices={k:make_index(memories[k],eval_cfg) for k in specs}
    backend=type(next(iter(indices.values()))).__name__ if indices else 'none'
    print(f'[test] NN backend={backend}; methods={len(specs)}; samples={len(test_samples)}',flush=True)

    scores={k:[] for k in specs}
    labels=[]
    has_local=any(s.mask is not None for s in test_samples)
    pixels={k:PixelAccumulator() for k in specs} if has_local else {}
    dl=loader(test_samples,cfg,with_mask=has_local,test=True)
    grid=int(cfg['model']['input_size'])//int(extractor.patch_size)
    frac=float(eval_cfg['topk_fraction'])
    total=len(dl); t0=time.time()
    heartbeat_n=max(1,int(eval_cfg.get('heartbeat_batches',20)))

    for bi,batch in enumerate(tqdm(dl,desc='test',leave=False,mininterval=10),start=1):
        feats=extractor(batch['image'])
        b=batch['image'].shape[0]
        labels.extend(batch['label'].numpy().tolist())
        cache={}
        for name,spec in specs.items():
            z=transform(feats,spec,cache)
            sc=indices[name].score(z.reshape(-1,z.shape[-1])).reshape(b,-1)
            k=max(1,int(math.ceil(sc.shape[1]*frac)))
            img=np.mean(np.partition(sc,-k,axis=1)[:,-k:],axis=1)
            scores[name].extend(img.tolist())
            if has_local:
                if sc.shape[1] != grid*grid:
                    raise RuntimeError(
                        f'Unexpected DINO patch-token count {sc.shape[1]} for grid {grid}x{grid}; '
                        f'input={cfg["model"]["input_size"]}, patch={extractor.patch_size}'
                    )
                maps=torch.from_numpy(sc.reshape(b,grid,grid))[:,None]
                maps=F.interpolate(
                    maps,size=(int(cfg['model']['input_size']),)*2,
                    mode='bilinear',align_corners=False
                )[:,0].numpy()
                for i in range(b):
                    label_i=int(batch['label'][i]); has_i=bool(batch['has_mask'][i])
                    if label_i==0 or has_i:
                        pixels[name].update(batch['mask'][i].numpy(),maps[i])
        if bi==1 or bi%heartbeat_n==0 or bi==total:
            _heartbeat('test',bi,total,t0,cfg,extra=f'backend={backend}',progress_path=progress_path)

    rows=[]
    for name in specs:
        row={
            'method':name,
            'image_auroc':safe_auc(labels,scores[name]),
            'image_auprc':safe_ap(labels,scores[name]),
            'image_auroc_ci95':bootstrap_auc(
                labels,scores[name],
                int(eval_cfg['bootstrap_samples']),int(cfg['seed'])
            )
        }
        if has_local:
            row.update(pixels[name].finalize())
        rows.append(row)
    if progress_path is not None:
        Path(progress_path).write_text(json.dumps({
            'stage':'complete','samples':len(test_samples),'methods':len(specs),
            'backend':backend,'updated_unix':time.time(),
        },indent=2),encoding='utf-8')
    return rows


def family_name(name):
    return re.sub(r'_seed\d+$','',name)


def write_summaries(df:pd.DataFrame,results_dir):
    if df.empty:
        return
    d=df.copy()
    d['method_family']=d['method'].map(family_name)
    metrics=[c for c in [
        'image_auroc','image_auprc','pixel_auroc','pixel_auprc','aupro',
        'feature_dim','memory_size','compression_ratio','selected_layer_count'
    ] if c in d.columns]
    rows=[]
    for (dataset,family),g in d.groupby(['dataset','method_family'],dropna=False):
        row={'dataset':dataset,'method_family':family,'runs':len(g)}
        for c in metrics:
            vals=pd.to_numeric(g[c],errors='coerce')
            row[c+'_mean']=float(vals.mean())
            row[c+'_std']=float(vals.std(ddof=1)) if vals.notna().sum()>1 else 0.0
        rows.append(row)
    pd.DataFrame(rows).to_csv(Path(results_dir)/'summary_metrics.csv',index=False)
