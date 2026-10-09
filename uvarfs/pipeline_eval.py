from __future__ import annotations
import json, math, re, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from .data import BMADDataset, validate_pixel_masks
from .memory import Reservoir, make_index
from .metrics import safe_auc, safe_ap, bootstrap_auc, PixelAccumulator, prepare_pixel_mask
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
        pin_memory=torch.cuda.is_available(),
        persistent_workers=int(cfg['model']['num_workers'])>0,
    )


def transform(feats,spec,concat_cache=None):
    mode=spec.get('layer_normalization','none')
    key=(mode,tuple(spec['layers']))
    if concat_cache is not None and key in concat_cache:
        x=concat_cache[key]
    else:
        x=concat_layers(feats,spec['layers'],mode)
        if concat_cache is not None:
            concat_cache[key]=x
    kind=spec['kind']
    if kind=='uvarfs':
        x=apply_uvarfs(x,spec['obj'])
    elif kind=='pca':
        x=apply_pca(x.reshape(-1,x.shape[-1]),spec['obj']).reshape(x.shape[0],x.shape[1],-1)
    elif kind in {'random','selected_raw'}:
        active=spec['obj']['active'] if kind=='selected_raw' else spec['obj']
        idx=torch.as_tensor(active,device=x.device,dtype=torch.long)
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


def _atomic_csv(frame,path):
    """Publish a complete diagnostic file even if final statistics fail later."""
    path=Path(path)
    temporary=path.with_name(path.name+'.tmp')
    frame.to_csv(temporary,index=False)
    temporary.replace(path)


def build_memories(extractor,train_samples,specs,cfg,progress_path=None):
    if not train_samples or any(s.label!=0 or s.split!='train' for s in train_samples):
        raise ValueError('normal memory requires the official normal-reference training split')
    memory_size=int(cfg['data']['memory_size'])
    memory_images=int(cfg['data'].get('memory_images',1024))
    patches_per_image=int(cfg['data'].get('memory_patches_per_image',32))
    seed=int(cfg['seed'])
    image_ids=_sample_indices(len(train_samples),memory_images,seed+1009)
    # Every method sees exactly the same sampled patch rows and reservoir draws.
    memories={k:Reservoir(memory_size,seed) for k in specs}
    dl=loader(train_samples,cfg,indices=image_ids,test=False)
    total=len(dl); t0=time.time()
    heartbeat_n=max(1,int(cfg['eval'].get('heartbeat_batches',20)))
    print(f'[memory] using {len(image_ids)}/{len(train_samples)} normal images, <= {patches_per_image} patches/image, cap={memory_size}',flush=True)
    patch_manifest=[]

    for bi,batch in enumerate(tqdm(dl,desc='memory',leave=False,mininterval=10),start=1):
        feats=extractor(batch['image'])
        p=next(iter(feats.values())).shape[1]
        patch_ids=_sample_patch_ids(p,patches_per_image,seed+bi)
        patch_manifest.append({'batch':bi,'patch_indices':patch_ids.tolist()})
        patch_idx=torch.as_tensor(patch_ids,device=next(iter(feats.values())).device,dtype=torch.long)
        cache={}
        for name,spec in specs.items():
            z=transform(feats,spec,cache)
            z=z.index_select(1,patch_idx)
            memories[name].add(z.reshape(-1,z.shape[-1]))
        if bi==1 or bi%heartbeat_n==0 or bi==total:
            _heartbeat('memory',bi,total,t0,cfg,extra=f'methods={len(specs)}',progress_path=progress_path)

    result={k:v.value() for k,v in memories.items()}
    if progress_path is not None:
        (Path(progress_path).parent/'memory_manifest.json').write_text(json.dumps({
            'normal_train_indices':image_ids.tolist(),
            'normal_train_paths':[str(train_samples[int(i)].image) for i in image_ids],
            'patch_batches':patch_manifest,'memory_capacity':memory_size,
            'reservoir_seed':seed,'shared_sampling_across_methods':True,
            'memory_rows':{k:len(v) for k,v in result.items()},
        },indent=2),encoding='utf-8')
    print('[memory] built: '+', '.join(f'{k}={len(v)}' for k,v in result.items()),flush=True)
    return result


def _save_heatmap(path, image, mask, maps):
    from PIL import Image, ImageDraw
    rgb=np.clip(image*np.array([.229,.224,.225])+np.array([.485,.456,.406]),0,1)
    height,width=rgb.shape[:2]
    panels=[('Input',Image.fromarray((rgb*255).astype(np.uint8)))]
    if mask is not None:
        panels.append(('Ground truth',Image.fromarray(np.repeat((mask>0)[...,None],3,axis=2).astype(np.uint8)*255)))
    for method,score in maps.items():
        # Cosine distance has a fixed 0..2 scale. No per-test tuning of scoring.
        value=np.clip(score/2.0,0,1)
        color=np.stack([np.clip(2*value,0,1),1-np.abs(2*value-1),np.clip(1-2*value,0,1)],axis=-1)
        panels.append((method,Image.fromarray(((.55*rgb+.45*color)*255).astype(np.uint8))))
    canvas=Image.new('RGB',(width*len(panels),height+24),'white')
    draw=ImageDraw.Draw(canvas)
    for j,(title,panel) in enumerate(panels):
        canvas.paste(panel,(j*width,24)); draw.text((j*width+5,5),title,fill='black')
    path.parent.mkdir(parents=True,exist_ok=True)
    canvas.save(path)


def evaluate(extractor,test_samples,specs,memories,cfg,progress_path=None):
    if not test_samples or not specs:
        raise ValueError('evaluation requires nonempty test samples and method specifications')
    eval_cfg=cfg['eval']
    dataset_name=test_samples[0].dataset if test_samples else ''
    coverage=validate_pixel_masks(test_samples,dataset_name)
    output=Path(progress_path).parent if progress_path is not None else None
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
    match_seconds={k:0.0 for k in specs}
    visual_methods=[k for k in eval_cfg.get('heatmap_methods',['all_raw','main']) if k in specs]
    visual_count=int(eval_cfg.get('max_heatmaps_per_dataset',12))
    # Deterministic example selection for FINAL visualization only.
    visual_ids=set()
    if output is not None and eval_cfg.get('save_heatmaps',False):
        abnormal=[i for i,s in enumerate(test_samples) if s.label==1]
        normal=[i for i,s in enumerate(test_samples) if s.label==0]
        visual_ids=set((abnormal[:(visual_count+1)//2]+normal)[:visual_count])
    position=0

    for bi,batch in enumerate(tqdm(dl,desc='test',leave=False,mininterval=10),start=1):
        feats=extractor(batch['image'])
        b=batch['image'].shape[0]
        labels.extend(batch['label'].numpy().tolist())
        cache={}
        visual_maps={i:{} for i in range(b) if position+i in visual_ids}
        prepared_masks=[prepare_pixel_mask(batch['mask'][i].numpy()) for i in range(b)] if has_local else []
        for name,spec in specs.items():
            z=transform(feats,spec,cache)
            flat=z.reshape(-1,z.shape[-1])
            index=indices[name]
            match_start=time.perf_counter()
            need_maps=has_local or (bool(visual_maps) and name in visual_methods)
            if hasattr(index,'score_tensor'):
                sc_t=index.score_tensor(flat).reshape(b,-1)
                k=max(1,int(math.ceil(sc_t.shape[1]*frac)))
                img=sc_t.topk(k,dim=1,largest=True,sorted=False).values.mean(dim=1).float().cpu().numpy()
                scores[name].extend(img.tolist())
                match_seconds[name]+=time.perf_counter()-match_start
                if need_maps:
                    if sc_t.shape[1] != grid*grid:
                        raise RuntimeError(
                            f'Unexpected DINO patch-token count {sc_t.shape[1]} for grid {grid}x{grid}; '
                            f'input={cfg["model"]["input_size"]}, patch={extractor.patch_size}'
                        )
                    maps=F.interpolate(
                        sc_t.reshape(b,1,grid,grid).float(),
                        size=(int(cfg['model']['input_size']),)*2,
                        mode='bilinear',align_corners=False
                    )[:,0].cpu().numpy()
            else:
                sc=index.score(flat).reshape(b,-1)
                k=max(1,int(math.ceil(sc.shape[1]*frac)))
                img=np.mean(np.partition(sc,-k,axis=1)[:,-k:],axis=1)
                scores[name].extend(img.tolist())
                match_seconds[name]+=time.perf_counter()-match_start
                if need_maps:
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

            if has_local:
                for i in range(b):
                    label_i=int(batch['label'][i]); has_i=bool(batch['has_mask'][i])
                    if label_i==0 or has_i:
                        pixels[name].update(batch['mask'][i].numpy(),maps[i],prepared=prepared_masks[i])
            if visual_maps and name in visual_methods:
                for i in visual_maps:
                    visual_maps[i][name]=maps[i]
        for i,examples in visual_maps.items():
            _save_heatmap(output/'heatmaps'/f'{position+i:06d}.png',
                          batch['image'][i].numpy(),
                          batch['mask'][i].numpy() if has_local else None,examples)
        position+=b
        if bi==1 or bi%heartbeat_n==0 or bi==total:
            _heartbeat('test',bi,total,t0,cfg,extra=f'backend={backend}',progress_path=progress_path)

    # The expensive DINO/NN pass is finished. Persist every prediction before
    # CI computation, so a statistics interruption does not hide scored images.
    if output is not None:
        predictions=pd.DataFrame({'image_path':[str(s.image) for s in test_samples],
                                  'label':labels,'mask_path':[str(s.mask) if s.mask else '' for s in test_samples],
                                  **scores})
        _atomic_csv(predictions,output/'image_predictions.csv')
        print(f'[test] predictions saved: {output/"image_predictions.csv"}; final statistics pending',flush=True)
    stats_start=time.time()
    _heartbeat('metrics',0,len(specs),stats_start,cfg,
               extra='point estimates and bootstrap pending',progress_path=progress_path)
    rows=[]
    for name in specs:
        row={
            'method':name,
            'image_auroc':safe_auc(labels,scores[name]),
            'image_auprc':safe_ap(labels,scores[name]),
            'matching_and_aggregation_seconds':match_seconds[name],
            'memory_vector_fp16_mib':memories[name].size*2/1024**2,
        }
        if has_local:
            row.update(pixels[name].finalize())
        rows.append(row)
    if output is not None:
        _atomic_csv(pd.DataFrame(rows),output/'evaluation_metrics.partial.csv')
    print(f'[metrics] point estimates ready; bootstrap draws={eval_cfg["bootstrap_samples"]} methods={len(specs)}',flush=True)
    for method_i,row in enumerate(rows,start=1):
        name=row['method']
        # Identify the current method BEFORE entering a potentially long call.
        _heartbeat('metrics',method_i-1,len(specs),stats_start,cfg,
                   extra=f'current={name} bootstrap',progress_path=progress_path)
        row['image_auroc_ci95']=bootstrap_auc(
            labels,scores[name],int(eval_cfg['bootstrap_samples']),int(cfg['seed']),
            batch_size=int(eval_cfg.get('bootstrap_batch_size',32)))
        if output is not None:
            _atomic_csv(pd.DataFrame(rows),output/'evaluation_metrics.partial.csv')
    _heartbeat('metrics',len(specs),len(specs),stats_start,cfg,
               extra='all final statistics ready',progress_path=progress_path)
    if progress_path is not None:
        Path(progress_path).write_text(json.dumps({
            'stage':'complete','samples':len(test_samples),'methods':len(specs),
            'backend':backend,'updated_unix':time.time(),
            'mask_coverage':coverage,
            'statistics_seconds':time.time()-stats_start,
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
        'feature_dim','memory_size','compression_ratio','selected_layer_count',
        'matching_and_aggregation_seconds','memory_vector_fp16_mib',
        'uvarfs_geometry_error'
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
