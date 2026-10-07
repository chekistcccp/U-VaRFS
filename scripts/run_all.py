#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, time, subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from uvarfs.utils import load_config, set_seed, ensure_dir, save_json
from uvarfs.data import discover_bmad_roots, scan_split, mask_coverage, validate_pixel_masks, PIXEL_DATASETS
from uvarfs.dinov3 import DINOv3Extractor
from uvarfs.pipeline_fit import fit_all_method_specs
from uvarfs.pipeline_eval import build_memories, evaluate, write_summaries
from uvarfs.representation import normalization_modes, normalization_prefix

from uvarfs.protocol import (EXPERIMENT_VERSION,BMAD_DATASETS,config_fingerprint,
                             code_fingerprint,data_fingerprint,expected_method_names,completed_result_compatible,validate_results_version)


def _append_layer_rows(layer_rows,out,dname,num_layers):
    paths=[out/'asls.json']+[out/name/'asls.json' for name in ['layer_l2','raw_input']]
    for apath in paths:
        if not apath.exists():
            continue
        obj=json.loads(apath.read_text(encoding='utf-8'))
        for l in range(1,num_layers+1):
            layer_rows.append({
                'dataset':dname,'layer':l,
                'layer_normalization':obj.get('layer_normalization','none'),
                'representation_role':'primary' if apath.parent==out else 'ablation',
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
    ap.add_argument('--check-data-only',action='store_true',help='validate normal train, all requested datasets and test masks without loading DINOv3')
    args=ap.parse_args()

    cfg=load_config(args.config)
    expected_method_names(cfg)  # Validate representation conventions before loading DINO.
    set_seed(int(cfg['seed']))
    paths=cfg.get('paths',{})
    processed=Path(paths.get('processed_data','data/processed/BMAD'))
    model_dir=Path(paths.get('model_dir','models/dinov3_vitsplus'))
    res=ensure_dir(cfg['results_dir'])
    validate_results_version(res)
    roots=discover_bmad_roots(processed)
    if not roots:
        raise SystemExit(f'No processed BMAD datasets found under {processed}. Run scripts/prepare_bmad.py first.')

    wanted=BMAD_DATASETS if args.dataset=='all' else {x.strip().lower() for x in args.dataset.split(',') if x.strip()}
    if not wanted:
        ap.error('--dataset must contain at least one dataset name')
    missing=wanted-set(roots)
    if missing:
        raise SystemExit(f'Missing requested BMAD datasets: {sorted(missing)}. The all-six protocol must not silently skip datasets.')
    if args.check_data_only:
        for dname,droot in sorted(roots.items()):
            if dname.lower() not in wanted:
                continue
            train=scan_split(droot,'train',dname)
            test=scan_split(droot,'test',dname)
            if not train or not test or any(s.label!=0 for s in train):
                raise ValueError(f'{dname}: invalid normal-reference train or empty test split')
            out=ensure_dir(res/dname)
            save_json(mask_coverage(test),out/'mask_coverage.json')
            coverage=validate_pixel_masks(test,dname)
            print(json.dumps({'dataset':dname,'normal_train':len(train),**coverage},ensure_ascii=False),flush=True)
        return
    repo=Path(__file__).resolve().parents[1]
    code_hash=code_fingerprint(repo)
    git_head=subprocess.run(['git','rev-parse','HEAD'],cwd=repo,capture_output=True,text=True).stdout.strip()
    extractor=DINOv3Extractor(model_dir,int(cfg['model']['input_size']),cfg['model']['amp'])
    all_rows,layer_rows=[],[]
    candidate_dim=extractor.num_layers*extractor.hidden_dim
    resume=bool(cfg.get('eval',{}).get('resume_completed',True)) and not args.force

    for dname,droot in sorted(roots.items()):
        if dname.lower() not in wanted:
            continue
        out=ensure_dir(res/dname)
        metrics_path=out/'metrics.csv'
        train=scan_split(droot,'train',dname)
        test=scan_split(droot,'test',dname)
        if not train or not test or any(s.label!=0 for s in train):
            raise ValueError(f'{dname}: train must be nonempty normal-reference data and test must be nonempty')
        save_json(mask_coverage(test),out/'mask_coverage.json')
        validate_pixel_masks(test,dname)
        data_hash=data_fingerprint(train,test)

        if resume and metrics_path.exists():
            try:
                existing=pd.read_csv(metrics_path)
            except Exception:
                existing=pd.DataFrame()
            try:
                metadata=json.loads((out/'run_metadata.json').read_text(encoding='utf-8'))
                progress=json.loads((out/'eval_progress.json').read_text(encoding='utf-8'))
            except (OSError,ValueError):
                metadata,progress={},{}
            compatible=completed_result_compatible(existing,metadata,progress,cfg,code_hash,dname in PIXEL_DATASETS,data_hash)
            compatible=compatible and all((out/name).is_file() for name in (
                'asls.json','fit_manifest.json','memory_manifest.json','image_predictions.csv',
                'normal_geometry_audit.json','representation_manifest.json'))
            _,modes=normalization_modes(cfg)
            compatible=compatible and all((out/normalization_prefix(mode).rstrip('_')/name).is_file()
                for mode in modes[1:] for name in ['asls.json','fit_manifest.json','uvarfs_main.json','normal_geometry_audit.json'])
            if compatible:
                print(f'[resume] {dname}: found complete {metrics_path}; skipping dataset ({len(existing)} method rows)',flush=True)
                all_rows.extend(existing.to_dict('records'))
                _append_layer_rows(layer_rows,out,dname,extractor.num_layers)
                _save_partial(all_rows,layer_rows,res)
                continue

        save_json({'experiment_version':EXPERIMENT_VERSION,'config':cfg,
                   'config_fingerprint':config_fingerprint(cfg),'code_fingerprint':code_hash,
                   'data_fingerprint':data_hash,
                   'git_head':git_head,'dataset':dname,'train_images':len(train),'test_images':len(test),
                   'expected_methods':expected_method_names(cfg),
                   'backbone':{'num_layers':extractor.num_layers,'hidden_dim':extractor.hidden_dim,
                               'patch_size':extractor.patch_size,'prefix_tokens':extractor.num_prefix},
                   'torch_version':torch.__version__,'cuda_version':torch.version.cuda,
                   'objective_matmul_precision':'highest',
                   'asls_geometry_representation':cfg['asls'].get('geometry_representation','patch'),
                   'asls_discrete_selection':cfg['asls'].get('discrete_selection','geometry_search'),
                   'uvarfs_sparsity_strategy':cfg['uvarfs'].get('sparsity_strategy','objective_forward_refit'),
                   'layer_normalization':cfg.get('representation',{}).get('layer_normalization','none'),
                   'timing_scope':'fit/memory/eval/peak are dataset-wide and shared across methods'},
                  out/'run_metadata.json')

        print(f'\n=== {dname}: train={len(train)} test={len(test)} ===',flush=True)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        t0=time.time()
        t=time.time()
        print(f'[{dname}] stage 1/3: fit ASLS/U-VaRFS',flush=True)
        specs=fit_all_method_specs(extractor,train,cfg,out)
        if sorted(specs)!=expected_method_names(cfg):
            raise ValueError(f'{dname}: fitted methods do not match the configured protocol')
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
                timing_scope='shared_dataset_all_methods',
                layer_normalization=spec.get('layer_normalization','none'),
                asls_geometry_feasible=spec.get('asls_geometry',{}).get('feasible'),
                asls_geometry_representation=spec.get('asls_geometry',{}).get('geometry_representation'),
                asls_geometry_error=spec.get('asls_geometry',{}).get('geometry_error'),
                asls_discrete_selection=spec.get('asls_geometry',{}).get('discrete_selection'),
            )
            if spec['kind']=='uvarfs':
                r.update(uvarfs_geometry_error=spec['obj']['geometry_error'],
                         uvarfs_geometry_feasible=spec['obj']['feasible'],
                         uvarfs_solver_converged=spec['obj']['solver_converged'],
                         uvarfs_selected_lambda=spec['obj']['lambda'],
                         uvarfs_objective=spec['obj']['objective'],
                         uvarfs_box_optimality_gap=spec['obj']['box_optimality_gap'],
                         uvarfs_objective_converged=spec['obj']['objective_converged'],
                         uvarfs_objective_certificate_scope=spec['obj']['objective_certificate_scope'],
                         uvarfs_budgeted_objective=spec['obj']['budgeted_objective'],
                         uvarfs_budgeted_box_optimality_gap=spec['obj']['budgeted_box_optimality_gap'],
                         uvarfs_budgeted_objective_converged=spec['obj']['budgeted_objective_converged'],
                         uvarfs_fixed_support_geometry_floor=spec['obj']['fixed_support_geometry_floor'],
                         uvarfs_fixed_support_can_meet_tolerance=spec['obj']['fixed_support_can_meet_tolerance'],
                         uvarfs_sparsity_strategy=spec['obj']['sparsity_strategy'],
                         uvarfs_budget_source=spec['obj'].get('budget_source','top_weights'),
                         uvarfs_objective_improvement=spec['obj'].get('objective_improvement',0.),
                         uvarfs_geometry_improvement=spec['obj'].get('geometry_improvement',0.),
                         uvarfs_fixed_support_box_gap=spec['obj'].get('fixed_support_box_gap'),
                         uvarfs_fixed_support_converged=spec['obj'].get('fixed_support_converged'),
                         uvarfs_exchange_source=spec['obj'].get('exchange_source'),
                         uvarfs_exchange_steps=spec['obj'].get('exchange_steps'),
                         uvarfs_exchange_objective_improvement=spec['obj'].get('exchange_objective_improvement'),
                         uvarfs_exchange_geometry_improvement=spec['obj'].get('exchange_geometry_improvement'))
                budget=spec['obj']['budget_geometry_audit']
                r.update(uvarfs_global_geometry_lower_bound=budget['geometry_error_lower_bound'],
                         uvarfs_necessary_features_lower_bound=budget['necessary_features_lower_bound'],
                         uvarfs_budget_ruled_out=budget['budget_ruled_out_at_working_precision'],
                         uvarfs_forward_source=spec['obj'].get('forward_source'),
                         uvarfs_forward_candidate_accepted=spec['obj'].get('forward_candidate_accepted'),
                         uvarfs_forward_objective_improvement=spec['obj'].get('forward_objective_improvement'))

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
