from __future__ import annotations
import math, re
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from .data import BMADDataset
from .memory import Reservoir, CosineIndex
from .metrics import safe_auc, safe_ap, bootstrap_auc, PixelAccumulator
from .transforms import concat_layers, apply_pca
from .u_varfs import apply_uvarfs


def loader(samples,cfg,with_mask=False):
    ds=BMADDataset(samples,int(cfg['model']['input_size']),with_mask=with_mask)
    return DataLoader(ds,batch_size=int(cfg['model']['batch_size']),shuffle=False,
                      num_workers=int(cfg['model']['num_workers']),pin_memory=True,
                      persistent_workers=int(cfg['model']['num_workers'])>0)


def transform(feats,spec):
    x=concat_layers(feats,spec['layers']); kind=spec['kind']
    if kind=='uvarfs': x=apply_uvarfs(x,spec['obj'])
    elif kind=='pca': x=apply_pca(x.reshape(-1,x.shape[-1]),spec['obj']).reshape(x.shape[0],x.shape[1],-1)
    elif kind=='random':
        idx=torch.as_tensor(spec['obj'],device=x.device,dtype=torch.long); x=x.index_select(-1,idx)
    return F.normalize(x.float(),dim=-1)


def build_memories(extractor,train_samples,specs,cfg):
    memories={k:Reservoir(int(cfg['data']['memory_size']),int(cfg['seed'])+i) for i,k in enumerate(specs)}
    for batch in tqdm(loader(train_samples,cfg),desc='memory',leave=False):
        feats=extractor(batch['image'])
        for name,spec in specs.items():
            z=transform(feats,spec); memories[name].add(z.reshape(-1,z.shape[-1]))
    return {k:v.value() for k,v in memories.items()}


def evaluate(extractor,test_samples,specs,memories,cfg):
    indices={k:CosineIndex(memories[k],bool(cfg['eval'].get('faiss_gpu',True)),
                           bool(cfg['eval'].get('faiss_exact',False)),int(cfg['eval'].get('faiss_nprobe',16))) for k in specs}
    scores={k:[] for k in specs}; labels=[]; has_local=any(s.mask is not None for s in test_samples)
    pixels={k:PixelAccumulator() for k in specs} if has_local else {}
    dl=loader(test_samples,cfg,with_mask=has_local)
    grid=int(cfg['model']['input_size'])//int(extractor.patch_size); frac=float(cfg['eval']['topk_fraction'])
    for batch in tqdm(dl,desc='test',leave=False):
        feats=extractor(batch['image']); b=batch['image'].shape[0]; labels.extend(batch['label'].numpy().tolist())
        for name,spec in specs.items():
            z=transform(feats,spec); sc=indices[name].score(z.reshape(-1,z.shape[-1])).reshape(b,-1)
            k=max(1,int(math.ceil(sc.shape[1]*frac))); img=np.mean(np.partition(sc,-k,axis=1)[:,-k:],axis=1); scores[name].extend(img.tolist())
            if has_local:
                if sc.shape[1] != grid*grid:
                    raise RuntimeError(f'Unexpected DINO patch-token count {sc.shape[1]} for grid {grid}x{grid}; input={cfg["model"]["input_size"]}, patch={extractor.patch_size}')
                maps=torch.from_numpy(sc.reshape(b,grid,grid))[:,None]
                maps=F.interpolate(maps,size=(int(cfg['model']['input_size']),)*2,mode='bilinear',align_corners=False)[:,0].numpy()
                for i in range(b):
                    label_i=int(batch['label'][i]); has_i=bool(batch['has_mask'][i])
                    if label_i==0 or has_i: pixels[name].update(batch['mask'][i].numpy(),maps[i])
    rows=[]
    for name in specs:
        row={'method':name,'image_auroc':safe_auc(labels,scores[name]),'image_auprc':safe_ap(labels,scores[name]),
             'image_auroc_ci95':bootstrap_auc(labels,scores[name],int(cfg['eval']['bootstrap_samples']),int(cfg['seed']))}
        if has_local: row.update(pixels[name].finalize())
        rows.append(row)
    return rows


def family_name(name): return re.sub(r'_seed\d+$','',name)


def write_summaries(df:pd.DataFrame,results_dir):
    if df.empty: return
    d=df.copy(); d['method_family']=d['method'].map(family_name)
    metrics=[c for c in ['image_auroc','image_auprc','pixel_auroc','pixel_auprc','aupro','feature_dim','memory_size','compression_ratio','selected_layer_count'] if c in d.columns]
    rows=[]
    for (dataset,family),g in d.groupby(['dataset','method_family'],dropna=False):
        row={'dataset':dataset,'method_family':family,'runs':len(g)}
        for c in metrics:
            vals=pd.to_numeric(g[c],errors='coerce'); row[c+'_mean']=float(vals.mean()); row[c+'_std']=float(vals.std(ddof=1)) if vals.notna().sum()>1 else 0.0
        rows.append(row)
    pd.DataFrame(rows).to_csv(results_dir/'summary_metrics.csv',index=False)
