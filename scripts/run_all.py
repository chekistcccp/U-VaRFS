#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from uvarfs.utils import load_config, set_seed, ensure_dir
from uvarfs.data import discover_bmad_roots, scan_split
from uvarfs.dinov3 import DINOv3Extractor
from uvarfs.pipeline_fit import fit_method_specs
from uvarfs.pipeline_eval import build_memories, evaluate, write_summaries

EXPERIMENT_VERSION='gpu-eval-v2'


def _append_layer_rows(layer_rows,out,dname,num_layers):
    apath=out/'asls.json'
    if not apath.exists():
        return
    obj=json.loads(apath.read_text(encoding='utf-8'))
    for l in range(1,num_layers+1):
        layer_rows.append({
            'dataset':dname,'layer':l,
            'selected':int(l in obj.get('selected_layers',[])),
            'probability':obj.get('probabilities',{}).get(str(l),np.nan),
            'variability':obj.get('variability',{}).get(str(l),np.nan),
            'geometry_error':obj.get('geometry_error',np.nan),
        })


def _save_partial(all_rows,layer_rows,res):
    df=pd.DataFrame(all_rows)
    if not df.empty:
        df.to_csv(res/'all_metrics.partial.csv',index=False)
        write_summaries(df,res)
    if layer_rows:
        pd.DataFrame(layer_rows).to_csv(res/'layer_selection.csv',index=False)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',default='configs/default.yaml')
    ap.add_argument('--dataset',default='all',help='all or comma-separated processed dataset names')
    ap.add_argument('--force',action='store_true',help='rerun datasets even if results/<dataset>/metrics.csv already exists')
    args=ap.parse_args()

    cfg=load_config(args.config)
    set_seed(int(cfg['seed']))
    paths=cfg.get('paths',{})
    processed=Path(paths.get('processed_data','data/processed/BMAD'))
    model_dir=Path(paths.get('model_dir','models/dinov3_vitsplus'))
    res=ensure_dir(cfg['results_dir'])
    roots=discover_bmad_roots(processed)
    if not roots:
        raise SystemExit(f'No processed BMAD datasets found under {processed}. Run scripts/prepare_bmad.py first.')

    wanted=None if args.dataset=='all' else {x.strip().lower() for x in args.dataset.split(',') if x.strip()}
    extractor=DINOv3Extractor(model_dir,int(cfg['model']['input_size']),cfg['model']['amp'])
    all_rows,layer_rows=[],[]
    candidate_dim=extractor.num_layers*extractor.hidden_dim
    resume=bool(cfg.get('eval',{}).get('resume_completed',True)) and not args.force

    for dname,droot in sorted(roots.items()):
        if wanted and dname.lower() not in wanted:
            continue
        out=ensure_dir(res/dname)
        metrics_path=out/'metrics.csv'

        if resume and metrics_path.exists():
            try:
                existing=pd.read_csv(metrics_path)
            except Exception:
                existing=pd.DataFrame()
            compatible=(not existing.empty and 'method' in existing.columns and 'experiment_version' in existing.columns and existing['experiment_version'].astype(str).eq(EXPERIMENT_VERSION).all())
            if compatible:
                print(f'[resume] {dname}: found complete {metrics_path}; skipping dataset ({len(existing)} method rows)',flush=True)
                all_rows.extend(existing.to_dict('records'))
                _append_layer_rows(layer_rows,out,dname,extractor.num_layers)
                _save_partial(all_rows,layer_rows,res)
                continue

        train=scan_split(droot,'train',dname)
        test=scan_split(droot,'test',dname)
        if not train or not test:
            print(f'[skip] {dname}: incomplete split',flush=True)
            continue

        print(f'\n=== {dname}: train={len(train)} test={len(test)} ===',flush=True)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        t0=time.time()
        t=time.time()
        print(f'[{dname}] stage 1/3: fit ASLS/U-VaRFS',flush=True)
        specs=fit_method_specs(extractor,train,cfg,out)
        fit_s=time.time()-t
        print(f'[{dname}] fit complete in {fit_s/60:.1f} min; methods={len(specs)}',flush=True)

        t=time.time()
        print(f'[{dname}] stage 2/3: build sampled normal memories',flush=True)
        memories=build_memories(
            extractor,train,specs,cfg,
            progress_path=out/'memory_progress.json'
        )
        mem_s=time.time()-t
        print(f'[{dname}] memory complete in {mem_s/60:.1f} min',flush=True)

        t=time.time()
        print(f'[{dname}] stage 3/3: evaluate test set',flush=True)
        rows=evaluate(
            extractor,test,specs,memories,cfg,
            progress_path=out/'eval_progress.json'
        )
        eval_s=time.time()-t
        print(f'[{dname}] evaluation complete in {eval_s/60:.1f} min',flush=True)

        peak=torch.cuda.max_memory_allocated()/1024**3 if torch.cuda.is_available() else 0.0
        for r in rows:
            spec=specs[r['method']]
            dim=int(memories[r['method']].shape[1])
            r.update(
                dataset=dname,seconds=time.time()-t0,
                fit_seconds=fit_s,memory_seconds=mem_s,eval_seconds=eval_s,
                peak_gpu_gb=peak,
                selected_layers=';'.join(map(str,spec['layers'])),
                selected_layer_count=len(spec['layers']),
                feature_dim=dim,candidate_dim=candidate_dim,
                compression_ratio=1.0-dim/max(candidate_dim,1),
                memory_size=len(memories[r['method']]),experiment_version=EXPERIMENT_VERSION,
            )

        pd.DataFrame(rows).to_csv(metrics_path,index=False)
        all_rows.extend(rows)
        _append_layer_rows(layer_rows,out,dname,extractor.num_layers)
        _save_partial(all_rows,layer_rows,res)
        print(f'[{dname}] checkpoint saved: {metrics_path}',flush=True)

        # Release per-dataset GPU/CPU state before the next dataset.
        del memories,specs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df=pd.DataFrame(all_rows)
    df.to_csv(res/'all_metrics.csv',index=False)
    write_summaries(df,res)
    if layer_rows:
        pd.DataFrame(layer_rows).to_csv(res/'layer_selection.csv',index=False)
    partial=res/'all_metrics.partial.csv'
    if partial.exists() and not df.empty:
        partial.unlink()

    if not df.empty:
        cols=[c for c in [
            'dataset','method','image_auroc','image_auprc',
            'pixel_auroc','aupro','feature_dim','compression_ratio'
        ] if c in df.columns]
        print(df[cols].to_string(index=False),flush=True)
        print(f"\nSaved: {res/'all_metrics.csv'}",flush=True)
        print(f"Saved: {res/'summary_metrics.csv'}",flush=True)


if __name__=='__main__':
    main()
