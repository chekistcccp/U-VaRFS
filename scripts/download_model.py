#!/usr/bin/env python3
from __future__ import annotations
import argparse, os
from pathlib import Path


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--model-id',default=os.getenv('DINOV3_MODELSCOPE_ID','keras/dinov3_vit_small_plus_lvd1689m')); ap.add_argument('--out',default='models/dinov3_vitsplus')
    args=ap.parse_args(); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    if (out/'config.json').exists() and any(p.is_file() for p in out.rglob('*')):
        print(f'[model] already present: {out}'); return
    from modelscope import snapshot_download
    print(f'[model] downloading from ModelScope: {args.model_id} -> {out}')
    snapshot_download(args.model_id,local_dir=str(out))

if __name__=='__main__': main()
