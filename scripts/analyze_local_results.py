#!/usr/bin/env python3
"""Read-only v20/v21 auditing; result files and reports remain outside Git."""
from __future__ import annotations
import argparse
import json
import itertools
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd

from uvarfs.protocol import BMAD_DATASETS, EXPERIMENT_VERSION, config_fingerprint, expected_method_names
from uvarfs.objectives import LOCAL, MATCHING, QUADRATIC, COSINE, PERFORMANCE, experiment_branches
from scripts.analyze_results import (source_hashes, validate_output, WeightedRanking,
    paired_bootstrap, verify_recorded_code, audit_sparse_path)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def audit_solver_result(path, cfg):
    result=read(path)
    objective=result.get('objective_definition')
    legacy_quadratic={'uvarfs_asls_top_weights.json','uvarfs_asls_prune_refit.json',
                      'uvarfs_asls_exchange_refit.json','uvarfs_legacy_main.json'}
    # These nested original-solver controls predate explicit objective identities.
    # Do not interpret a missing Main/local identity as a legacy control.
    if path.name in legacy_quadratic:
        if objective not in {None,QUADRATIC}:
            raise ValueError('legacy quadratic control has a different objective')
        objective=QUADRATIC
    if objective==LOCAL:
        return audit_local_solution(result,cfg)
    if objective==MATCHING:
        return audit_matching_solution(result,cfg)
    if objective in {QUADRATIC,COSINE,PERFORMANCE}:
        return audit_sparse_path(result,cfg)
    raise ValueError('missing or unknown solver objective identity')


def audit_local_solution(result, cfg):
    if result['objective_definition'] != LOCAL or result['global_optimality_claimed']:
        raise ValueError('wrong local objective/certificate')
    active=np.asarray(result['active'],int);weights=np.asarray(result['budgeted_weights'],float)
    k=min(int(cfg['max_features']),len(weights))
    if (len(active)!=k or len(set(active))!=k or (active<0).any() or (active>=len(weights)).any()
        or not np.isfinite(weights).all()):
        raise ValueError('invalid local fixed-K support')
    p=weights[active]
    cap=float(cfg.get('performance_weight_cap_factor',4.));floor=float(cfg.get('cosine_weight_floor_mass',1e-4))
    if not np.isclose(p.sum(),1,atol=2e-6) or p.min()<floor/k-1e-7 or p.max()>min(1,cap/k)+2e-6:
        raise ValueError('local simplex floor/cap violated')
    if np.count_nonzero(weights)!=k or not np.allclose(np.sqrt(p),result['scales'],rtol=1e-5,atol=1e-7):
        raise ValueError('local scales/support disagree')
    if not np.isclose(1/np.square(p).sum(),result['effective_weight_dimension'],rtol=1e-5):
        raise ValueError('wrong effective dimension')
    candidates=result['candidate_pool']
    for i,c in enumerate(candidates):
        cp=np.asarray(c['weights']);ca=np.asarray(c['active'],int)
        if (c['candidate_id']!=i or len(ca)!=k or len(set(ca))!=k or (ca<0).any() or (ca>=len(weights)).any()
            or not np.isfinite(cp).all() or not np.isclose(cp.sum(),1,atol=2e-6)
            or cp.min()<floor/k-1e-7 or cp.max()>min(1,cap/k)+2e-6):
            raise ValueError('invalid local candidate')
        train=c['training_terms'];val=c['validation_terms']
        training=train['local_representation_term']+result['rank_weight']*train['rank_term']+train['variability_term']
        validation=val['local_representation_term']+result['rank_weight']*val['rank_term']
        if not np.allclose([training,validation,validation+train['variability_term']],
                           [c['training_objective'],c['validation_objective'],c['selection_score']],rtol=1e-5,atol=1e-6):
            raise ValueError('local objective components disagree')
        if not np.isclose(train['variability_term'],result['beta']*train['relative_variability'],rtol=1e-5,atol=1e-7):
            raise ValueError('local variability coefficient mismatch')
    best=min(candidates,key=lambda c:(c['selection_score'],c['candidate_id']))
    if best['candidate_id']!=result['candidate_id'] or best['active']!=result['active'] or not np.allclose(best['weights'],p):
        raise ValueError('local holdout candidate selection mismatch')
    if not np.allclose([best['training_objective'],best['validation_objective'],best['selection_score']],
                      [result['objective'],result['validation_objective'],result['selection_score']],rtol=1e-5,atol=1e-6):
        raise ValueError('selected local summary mismatch')
    if result['feasible'] is not None or not result['global_geometry_is_diagnostic_only']:
        raise ValueError('local objective must not claim global Gram feasibility')
    return True


def audit_matching_solution(result, cfg):
    if (result['objective_definition']!=MATCHING or result['global_optimality_claimed'] or
            result['support_generation_uses_holdout'] or result['feasible'] is not None or
            result['objective_dtype']!='float64' or
            result['reference_policy']!='all_normal_fit_patches_excluding_same_image' or
            result['objective_certificate_scope']!='fixed_support_current_nearest_branch_projected_residual_only' or
            not np.isfinite(result['temperature_fit_only']) or result['temperature_fit_only']<.01):
        raise ValueError('wrong matching objective/protocol/certificate')
    weights=np.asarray(result['budgeted_weights'],float); active=np.asarray(result['active'],int)
    k=min(int(cfg['max_features']),len(weights));floor=float(cfg.get('cosine_weight_floor_mass',1e-4))
    cap=min(1.,float(cfg.get('performance_weight_cap_factor',4.))/k)
    def check_support(ca,cp):
        ca=np.asarray(ca,int);cp=np.asarray(cp,float)
        if (len(ca)!=k or cp.shape!=ca.shape or len(set(ca))!=k or (ca<0).any() or
            (ca>=len(weights)).any() or not np.isfinite(cp).all() or
            abs(cp.sum()-1)>1e-8 or cp.min()<floor/k-1e-10 or cp.max()>cap+1e-10):
            raise ValueError('matching fixed-K simplex bounds violated')
    check_support(active,weights[active])
    p=weights[active]
    if np.count_nonzero(weights)!=k or not np.allclose(np.sqrt(p),result['scales'],rtol=1e-9,atol=1e-10):
        raise ValueError('matching scales/support disagree')
    if not np.isclose(1/np.square(p).sum(),result['effective_weight_dimension'],rtol=1e-9):
        raise ValueError('matching effective dimension mismatch')
    pool=result['candidate_pool'];training_pool=[]
    maximum=3*(2+2*int(cfg.get('matching_exchange_max_steps',2))*int(cfg.get('matching_exchange_candidates',4)))
    if not 1<=len(pool)<=maximum or result['fixed_feature_budget']!=k:
        raise ValueError('matching candidate or feature budget mismatch')
    for key in ['fit_score_scale','fit_distribution_scale','fit_nuisance_scale',
                'validation_score_scale','validation_distribution_scale']:
        if not np.isfinite(result[key]) or result[key]<=0:
            raise ValueError('invalid matching normalization scale')
    remove={'validation_valid','validation_error','validation_objective','validation_terms','selection_score'}
    def total(terms):
        return result['score_weight']*terms['normal_match_score_relative_mse']+result['distribution_weight']*terms['normal_neighbor_distribution_kl_relative']+terms['variability_term']
    for i,c in enumerate(pool):
        check_support(c['active'],c['weights'])
        if c['candidate_id']!=i or not np.isclose(total(c['training_terms']),c['training_objective'],rtol=1e-8,atol=1e-9):
            raise ValueError('matching training candidate identity/terms disagree')
        if not np.isclose(c['training_terms']['variability_term'],result['beta']*c['training_terms']['relative_variability'],rtol=1e-8,atol=1e-9):
            raise ValueError('matching nuisance coefficient mismatch')
        if c['validation_valid']:
            val=total(c['validation_terms'])
            if c['validation_terms']['variability_term']!=0 or not np.allclose(
                [val,val+c['training_terms']['variability_term']],
                [c['validation_objective'],c['selection_score']],rtol=1e-8,atol=1e-9):
                raise ValueError('matching holdout score mismatch')
        elif 'zero sparse' not in c.get('validation_error',''):
            raise ValueError('unexplained invalid matching holdout candidate')
        if c['diagnostics']:
            hist=np.asarray(c['diagnostics']['training_history'])
            if not np.isfinite(hist).all() or (np.diff(hist)>1e-10).any() or not np.isclose(hist[-1],c['training_objective'],rtol=1e-8,atol=1e-9):
                raise ValueError('matching refit history fails strict descent')
        training_pool.append({key:value for key,value in c.items() if key not in remove})
    digest=hashlib.sha256(json.dumps(training_pool,sort_keys=True,allow_nan=False).encode()).hexdigest()
    if digest!=result['candidate_pool_fit_only_sha256']:
        raise ValueError('fit-only matching pool fingerprint mismatch')
    for exchange in result['accepted_exchange_trace']:
        c=pool[exchange['candidate_id']]
        if not exchange['after']<exchange['before']-1e-8 or not np.isclose(exchange['after'],c['training_objective']):
            raise ValueError('matching search continuation did not improve fit objective')
    best=min((c for c in pool if c['validation_valid']),key=lambda c:(c['selection_score'],c['candidate_id']))
    if best['candidate_id']!=result['candidate_id'] or best['active']!=result['active'] or not np.allclose(best['weights'],p,rtol=1e-9,atol=1e-10):
        raise ValueError('matching holdout winner mismatch')
    if not np.allclose([best['training_objective'],best['validation_objective'],best['selection_score']],
                      [result['objective'],result['validation_objective'],result['selection_score']],rtol=1e-8,atol=1e-9):
        raise ValueError('matching summary mismatch')
    residual=result['fixed_support_first_order_residual']
    if (not np.isfinite(residual) or residual<0 or residual!=best['fixed_support_first_order_residual'] or
        result['solver_converged']!=(residual<=float(cfg.get('tol',1e-5)))):
        raise ValueError('matching selected convergence diagnostic mismatch')
    return True


def audit(results, output, completed_only=False, draws=2000):
    results,output=Path(results),Path(output)
    validate_output(results,output);before=source_hashes(results)
    folders={name:results/name for name in BMAD_DATASETS if (results/name/'run_metadata.json').is_file()}
    rows,paired,records=[],[],[];identities=set();completed=[]
    for name,folder in sorted(folders.items()):
        meta=read(folder/'run_metadata.json');cfg=meta['config']
        version=meta['experiment_version']
        if version not in {EXPERIMENT_VERSION,'gpu-eval-v20-normal-local-selection'} or cfg['uvarfs']['objective'] not in {LOCAL,MATCHING} or (version=='gpu-eval-v20-normal-local-selection' and cfg['uvarfs']['objective']!=LOCAL):
            raise ValueError('mixed version or non-local results')
        if meta['config_fingerprint']!=config_fingerprint(cfg):
            raise ValueError('config fingerprint mismatch')
        identities.add((meta['config_fingerprint'],meta['code_fingerprint'],meta['git_head']))
        if not (folder/'metrics.csv').is_file():
            continue
        frame=pd.read_csv(folder/'metrics.csv');progress=read(folder/'eval_progress.json')
        if progress['stage']!='complete' or progress['methods']!=len(frame):
            raise ValueError('metrics present without complete evaluation')
        if sorted(frame.method)!=expected_method_names(cfg) or sorted(frame.method)!=sorted(meta['expected_methods']):
            raise ValueError('incomplete method set')
        if not frame.experiment_version.eq(version).all() or not frame.dataset.eq(name).all():
            raise ValueError('metric version/dataset mismatch')
        finite=['image_auroc','image_auprc']+(['pixel_auroc','pixel_auprc','aupro'] if name in {'brain','liver','resc'} else [])
        if not np.isfinite(frame[finite]).all().all() or not frame[finite].apply(lambda c:c.between(0,1)).all().all():
            raise ValueError('invalid or incomplete detection metrics')
        fit=read(folder/'fit_manifest.json');memory=read(folder/'memory_manifest.json')
        for manifest in [fit,memory]:
            paths=manifest['normal_train_paths'];ids=manifest['normal_train_indices']
            if len(paths)!=len(ids) or len(set(ids))!=len(ids) or len(set(paths))!=len(paths) or any(
                    '/train/good/' not in '/'+p.replace('\\','/').lower() for p in paths):
                raise ValueError('invalid normal-only provenance')
        split=fit['normal_selection_split'];fg=split['fit_patch_groups'];vg=split['holdout_patch_groups']
        fi=split['fit_image_positions'];vi=split['holdout_image_positions']
        if (set(fg)&set(vg) or len(fg)!=split['fit_patch_rows'] or len(vg)!=split['holdout_patch_rows']
            or set(fi)&set(vi) or set(fi)|set(vi)!=set(range(fit['fit_images']))
            or not set(fg)<=set(fi) or not set(vg)<=set(vi)
            or split['holdout_used_for_gradients'] or split['holdout_used_for_selection_reference']
            or split['paired_patch_rows']>len(fg) or fit['variability_images']>len(split['fit_image_positions'])):
            raise ValueError('normal fit/holdout leakage or row mismatch')
        expected_holdout=max(1,int(round(fit['fit_images']*cfg['data']['normal_holdout_fraction'])))
        if len(vi)!=expected_holdout or fit['variability_images']!=min(cfg['data']['variability_images'],fit['fit_images']):
            raise ValueError('normal split/variability budget mismatch')
        if (fit['fit_images']!=len(fit['normal_train_indices']) or fit['fit_images']>cfg['data']['fit_images']
            or fit['patches_per_image']!=cfg['data']['patches_per_image']
            or memory['memory_capacity']!=cfg['data']['memory_size']
            or len(memory['normal_train_indices'])>cfg['data']['memory_images']
            or not memory['shared_sampling_across_methods'] or set(memory['memory_rows'])!=set(frame.method)
            or len(set(memory['memory_rows'].values()))!=1
            or not frame.memory_size.eq(next(iter(memory['memory_rows'].values()))).all()):
            raise ValueError('normal data/memory budget mismatch')
        manifest=read(folder/'representation_manifest.json');indexed=frame.set_index('method')
        for branch in experiment_branches(cfg):
            base=folder/branch['folder'];prefix=branch['prefix'];bm=read(base/'fit_manifest.json')
            for key in ['normal_train_indices','normal_train_paths','sampled_patches','perturbation_rng','normal_selection_split']:
                if fit[key]!=bm[key]:
                    raise ValueError('branches do not share normal input/split')
            if bm['layer_normalization']!=branch['layer_normalization']:
                raise ValueError('branch input convention mismatch')
            asls=read(base/'asls.json')
            if asls['discrete_selection']!='normal_local':
                raise ValueError('wrong Main ASLS selection')
            layers=range(1,int(meta['backbone']['num_layers'])+1)
            expected={tuple(c) for k in range(int(cfg['asls']['min_layers']),min(int(cfg['asls']['max_layers']),len(layers))+1)
                      for c in itertools.combinations(layers,k)}
            actual=[tuple(c['layers']) for c in asls['selection_path']]
            # Zero-row subsets are deliberately invalid and may be absent;
            # normal BMAD nonzero features ordinarily produce the complete pool.
            if len(actual)!=len(set(actual)) or not set(actual)<=expected:
                raise ValueError('ASLS candidate budget/identity mismatch')
            for c in asls['selection_path']:
                score=c['local_representation_term']+cfg['asls'].get('local_rank_weight',1.)*c['rank_term']+cfg['asls']['variability_weight']*c['nuisance_cosine_distance']
                if not np.isfinite(score) or not np.isclose(score,c['selection_score'],rtol=1e-5,atol=1e-6):
                    raise ValueError('ASLS score components disagree')
            selected=min(asls['selection_path'],key=lambda c:(c['selection_score'],c['layers']))
            if selected['layers']!=asls['selected_layers'] or not np.isclose(selected['selection_score'],asls['selection_score']):
                raise ValueError('ASLS holdout selection mismatch')
            for path in base.glob('uvarfs_*.json'):
                audit_solver_result(path,cfg['uvarfs'])
            main=read(base/'uvarfs_main.json');main_name=prefix+'main';k=len(main['active'])
            if indexed.loc[main_name,'feature_dim']!=k or main['layers']!=asls['selected_layers']:
                raise ValueError('Main input/support dimensions disagree')
            if main['objective_definition']!=branch['objective'] or manifest['normalization_by_method'][main_name]!=branch['layer_normalization']:
                raise ValueError('Main branch identity mismatch')
            for control in ['asls_pca']+[f'asls_random_seed{s}' for s in cfg['random_baseline_seeds']]:
                if prefix+control in indexed.index and indexed.loc[prefix+control,'feature_dim']!=k:
                    raise ValueError('unequal feature baseline budget')
            if 'asls_selected_raw' in cfg['methods']:
                uniform=read(base/'selected_support_control.json')
                if uniform['active']!=main['active'] or uniform['layers']!=main['layers'] or uniform['source_main']!=main_name:
                    raise ValueError('same-support control mismatch')
            if branch['objective']==LOCAL:
                for method,key,expected in [('asls_no_variability','beta',0.),('asls_no_rank','rank_weight',0.)]:
                    if method+'_uvarfs' in cfg['methods']:
                        control=read(base/f'uvarfs_{method}.json')
                        if control[key]!=expected or control['layers']!=main['layers']:
                            raise ValueError('local causal control mismatch')
            if branch['objective']==MATCHING:
                for method,key in [('asls_no_variability','beta'),('asls_no_score','score_weight'),
                                   ('asls_no_distribution','distribution_weight')]:
                    if method+'_uvarfs' in cfg['methods']:
                        control=read(base/f'uvarfs_{method}.json')
                        if control[key]!=0. or control['layers']!=main['layers']:
                            raise ValueError('matching causal control mismatch')
        predictions=pd.read_csv(folder/'image_predictions.csv');y=predictions.label.to_numpy()
        if len(y)!=progress['samples'] or predictions.image_path.duplicated().any() or set(np.unique(y))!={0,1}:
            raise ValueError('prediction coverage/labels mismatch')
        coverage=read(folder/'mask_coverage.json')
        if coverage['samples']!=len(y) or coverage['abnormal']!=int((y==1).sum()):
            raise ValueError('mask coverage counts mismatch')
        if name in {'brain','liver','resc'} and (coverage['missing_abnormal_masks'] or coverage['matched_abnormal_masks']!=int((y==1).sum())):
            raise ValueError('missing pixel masks')
        for row in frame.itertuples():
            auc,ap=WeightedRanking(y,predictions[row.method]).metrics(np.ones(len(y)))
            if not np.allclose([auc[0],ap[0]],[row.image_auroc,row.image_auprc],rtol=1e-10,atol=1e-10):
                raise ValueError('image metrics do not match predictions')
        comparisons={m:[m] for m in ['asls_raw','asls_pca','asls_selected_raw','asls_no_variability_uvarfs',
            'asls_no_rank_uvarfs','asls_no_score_uvarfs','asls_no_distribution_uvarfs',
            'local_main','previous_main','fixed4_raw','all_raw'] if m in indexed.index}
        randoms=[f'asls_random_seed{s}' for s in cfg['random_baseline_seeds'] if f'asls_random_seed{s}' in indexed.index]
        if randoms:
            comparisons['asls_random_mean']=randoms
        ci=paired_bootstrap(predictions,comparisons,draws=draws)
        for method,names in comparisons.items():
            delta=indexed.loc['main','image_auroc']-indexed.loc[names,'image_auroc'].mean()
            paired.append({'dataset':name,'comparison':method,'delta_image_auroc':delta,
                           'ci95_lower':ci[method][0],'ci95_upper':ci[method][1]})
        records.append({'dataset':name,'git_blob_code_verified':verify_recorded_code(meta),
                        'fit_image_count':len(split['fit_image_positions']),'holdout_image_count':len(split['holdout_image_positions'])})
        rows.append(frame);completed.append(name)
    if len(identities)>1:
        raise ValueError('inconsistent code/config/revision across returned datasets')
    if not completed or (not completed_only and set(completed)!=BMAD_DATASETS):
        raise ValueError('full reporting requires all six complete datasets; use --completed-only for partial return')
    df=pd.concat(rows,ignore_index=True)
    for name in ['all_metrics.csv','all_metrics.partial.csv']:
        path=results/name
        if path.is_file():
            supplied=pd.read_csv(path);supplied=supplied[supplied.dataset.isin(completed)]
            pd.testing.assert_frame_equal(supplied.sort_values(['dataset','method']).reset_index(drop=True),
                df.sort_values(['dataset','method']).reset_index(drop=True),check_dtype=False,check_like=True)
    if before!=source_hashes(results):
        raise ValueError('original result artifacts changed during audit')
    output.mkdir(parents=True,exist_ok=True)
    df.to_csv(output/'verified_metrics.csv',index=False)
    pd.DataFrame(paired).to_csv(output/'paired_image_auroc.csv',index=False)
    df['method_family']=df.method.str.replace(r'_seed\d+$','',regex=True)
    metrics=['image_auroc','image_auprc','pixel_auroc','pixel_auprc','aupro']
    df.groupby(['dataset','method_family'])[metrics].agg(['mean','std','count']).to_csv(output/'family_mean_sd.csv')
    data={'experiment_version':version,'completed_datasets':completed,
          'all_six_complete':set(completed)==BMAD_DATASETS,'source_sha256':before,'diagnostics':records,
          'performance_proxy_is_not_detection_guarantee':True,'bootstrap_unit':'image; patient grouping unavailable'}
    (output/'analysis_data.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    (output/'analysis.md').write_text(
        '# Normal sparse selection result audit\n\n'
        f'Completed datasets: {", ".join(completed)}. All six complete: {data["all_six_complete"]}.\n\n'
        'See verified_metrics.csv for all image/pixel metrics, family_mean_sd.csv for all random seeds, '
        'and paired_image_auroc.csv for paired differences and image bootstrap intervals. '
        'Compare Main with Raw/PCA/Random and beta-zero before claiming a sparse-selection or variability benefit. '
        'Lower normal loss, compression and speed do not establish detection improvement. '
        'Image intervals do not account for correlated patients or repeated development on the same test set.\n',encoding='utf-8')
    return data


def audit_independent_seeds(roots, output, draws=2000):
    """Across-fit SD, distinct from within-image bootstrap or Random seeds."""
    roots=[Path(p).resolve() for p in roots];output=Path(output)
    if len(roots)<2 or len(set(roots))!=len(roots):
        raise ValueError('at least two distinct independent result directories required')
    seeds=set();settings=set();codes=set();frames=[]
    for root in roots:
        validate_output(root,output)
        metadata=[read(root/name/'run_metadata.json') for name in sorted(BMAD_DATASETS)]
        seed=int(metadata[0]['config']['seed'])
        if seed in seeds or any(int(m['config']['seed'])!=seed for m in metadata):
            raise ValueError('duplicate or mixed independent fitting seeds')
        seeds.add(seed)
        for m in metadata:
            config=dict(m['config']);config.pop('seed');config.pop('results_dir')
            settings.add(json.dumps(config,sort_keys=True));codes.add(m['code_fingerprint'])
        audit(root,output/f'seed{seed}',False,draws)
        frames.append(pd.read_csv(output/f'seed{seed}'/'verified_metrics.csv').assign(fit_seed=seed))
    if len(settings)!=1 or len(codes)!=1:
        raise ValueError('independent seeds must share method settings and source code')
    data=pd.concat(frames,ignore_index=True)
    metrics=['image_auroc','image_auprc','pixel_auroc','pixel_auprc','aupro']
    data.groupby(['dataset','method'])[metrics].agg(['mean','std','count']).to_csv(output/'independent_seed_mean_sd.csv')
    data.to_csv(output/'independent_seed_metrics.csv',index=False)
    macro=data[data.method=='main'].groupby('fit_seed')[['image_auroc','image_auprc']].mean()
    macro.to_csv(output/'main_all_six_macro_by_seed.csv')
    (output/'independent_seeds.json').write_text(json.dumps({'fit_seeds':sorted(seeds),
        'datasets_per_seed':6,'main_macro_mean':macro.mean().to_dict(),'main_macro_sd':macro.std(ddof=1).to_dict(),
        'random_baseline_seeds_are_not_main_repeats':True},indent=2),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser()
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--results',type=Path)
    group.add_argument('--seed-results',type=Path,nargs='+',help='two or more complete all-six independent fits')
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--completed-only',action='store_true')
    parser.add_argument('--bootstrap',type=int,default=2000)
    args=parser.parse_args()
    if args.bootstrap<100:
        parser.error('at least 100 bootstrap draws required')
    if args.seed_results is not None:
        if args.completed_only:
            parser.error('independent seed summaries require all six datasets per seed')
        audit_independent_seeds(args.seed_results,args.out,args.bootstrap)
    else:
        audit(args.results,args.out,args.completed_only,args.bootstrap)


if __name__=='__main__':
    main()
