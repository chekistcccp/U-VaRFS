"""Publish global results only after all six isolated dataset workers complete."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd

from uvarfs.protocol import (BMAD_DATASETS, EXPERIMENT_VERSION, code_fingerprint,
    config_fingerprint, completed_result_compatible)
from uvarfs.objectives import experiment_branches
from uvarfs.pipeline_eval import write_summaries
from uvarfs.utils import load_config
from scripts.run_all import _append_layer_rows


def merge_workers(root, cfg, code_hash):
    root=Path(root);frames=[];layers=[];revisions=set()
    for dataset in sorted(BMAD_DATASETS):
        folder=root/dataset
        meta=json.loads((folder/'run_metadata.json').read_text(encoding='utf-8'))
        progress=json.loads((folder/'eval_progress.json').read_text(encoding='utf-8'))
        frame=pd.read_csv(folder/'metrics.csv')
        if (meta.get('dataset')!=dataset or meta.get('experiment_version')!=EXPERIMENT_VERSION or
            meta.get('config')!=cfg or not meta.get('data_fingerprint') or
            meta.get('config_fingerprint')!=config_fingerprint(cfg) or
            not completed_result_compatible(frame,meta,progress,cfg,code_hash,
                dataset in {'brain','liver','resc'},meta.get('data_fingerprint')) or
            not frame.dataset.eq(dataset).all()):
            raise ValueError(f'{dataset}: incomplete or inconsistent worker result')
        for name in ['fit_manifest.json','memory_manifest.json','image_predictions.csv','representation_manifest.json']:
            if not (folder/name).is_file():
                raise ValueError(f'{dataset}: missing {name}')
        for branch in experiment_branches(cfg):
            for name in ['asls.json','uvarfs_main.json','normal_geometry_audit.json']:
                if not (folder/branch['folder']/name).is_file():
                    raise ValueError(f'{dataset}: incomplete branch artifact {name}')
        revisions.add(meta['git_head']);frames.append(frame)
        _append_layer_rows(layers,folder,dataset,int(meta['backbone']['num_layers']),cfg)
    if len(revisions)!=1:
        raise ValueError('worker source revisions differ')
    df=pd.concat(frames,ignore_index=True)
    # No global output is written until the entire all-six validation above passes.
    temp=root/'all_metrics.merge.tmp'
    df.to_csv(temp,index=False);temp.replace(root/'all_metrics.csv')
    write_summaries(df,root)
    pd.DataFrame(layers).to_csv(root/'layer_selection.csv',index=False)
    (root/'workers_complete.json').write_text(json.dumps({
        'experiment_version':EXPERIMENT_VERSION,'datasets':sorted(BMAD_DATASETS),
        'config_fingerprint':config_fingerprint(cfg),'code_fingerprint':code_hash,
        'git_head':next(iter(revisions)),'rows':len(df)},indent=2),encoding='utf-8')
    return df


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default='configs/default.yaml')
    parser.add_argument('--seed',type=int,required=True)
    args=parser.parse_args()
    cfg=load_config(args.config);cfg['seed']=args.seed
    cfg['results_dir']=str(Path(cfg['results_dir'])/f'seed{args.seed}')
    frame=merge_workers(cfg['results_dir'],cfg,code_fingerprint(Path(__file__).resolve().parents[1]))
    print(f'All six complete: seed={args.seed}, rows={len(frame)}',flush=True)


if __name__=='__main__':
    main()
