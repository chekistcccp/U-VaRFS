#!/usr/bin/env python3
"""Normal-cache pilot only: no backbone forward, test scan, or evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path, PurePosixPath

import torch

from uvarfs.normal_tail_selection import PILOT_VERSION, select_normal_tail
from uvarfs.protocol import code_fingerprint
from uvarfs.utils import load_config, save_json


def project_path(path, root, *, output=False):
    raw = Path(path)
    resolved = (raw if raw.is_absolute() else root/raw).resolve()
    bound = root/'results' if output else root
    if not resolved.is_relative_to(bound) or resolved == bound:
        raise ValueError('pilot paths must resolve inside the current project results/root')
    if output and (resolved.exists() or resolved.with_name(resolved.name+'.partial').exists()):
        raise ValueError('fresh pilot output required; historical files are read-only')
    return resolved


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda: stream.read(1024*1024), b''):
            digest.update(data)
    return digest.hexdigest()


def validate_normal_manifest(cached, recorded):
    for key in ['normal_train_indices', 'normal_train_paths', 'sampled_patches',
                'normal_selection_split', 'perturbation_rng']:
        if key not in cached or cached[key] != recorded.get(key):
            raise ValueError('normal source manifest mismatch: '+key)
    paths = cached['normal_train_paths']
    if not paths or len(paths) != len(cached['normal_train_indices']):
        raise ValueError('normal training image roster is missing')
    for value in paths:
        parts = PurePosixPath(value.replace('\\', '/')).parts
        if not any(parts[i:i+2] == ('train', 'good') for i in range(len(parts)-1)):
            raise ValueError('only official normal train/good paths are allowed')
    split = cached['normal_selection_split']
    if (split.get('holdout_used_for_gradients') is not False or
            split.get('holdout_used_for_selection_reference') is not False):
        raise ValueError('normal holdout must not enter support generation')


def run(cache_path, solution_path, beta_zero_path, config_path, output, root, device='cpu'):
    root = root.resolve()
    paths = [project_path(p, root) for p in [cache_path, solution_path, beta_zero_path, config_path]]
    cache_path, solution_path, beta_zero_path, config_path = paths
    output = project_path(output, root, output=True)
    if subprocess.run(['git', 'check-ignore', '-q', '--', str(output.relative_to(root))],
                      cwd=root, capture_output=True).returncode != 0:
        raise ValueError('pilot output must be Git ignored')
    cfg = load_config(config_path)
    if set(cfg) != {'pilot_version', 'feature_budget', 'selector'} or cfg['pilot_version'] != PILOT_VERSION:
        raise ValueError('explicit normal-tail pilot configuration required')
    settings = cfg['selector']
    if set(settings) != {'top_fraction', 'tail_fraction', 'mean_se_multiplier', 'chunk'}:
        raise ValueError('unknown or incomplete normal-tail selector settings')
    if isinstance(cfg['feature_budget'], bool) or not isinstance(cfg['feature_budget'], int) or cfg['feature_budget'] < 1:
        raise ValueError('invalid pilot feature budget')
    if device == 'cuda':
        if os.environ.get('CUDA_VISIBLE_DEVICES') not in {'0', '1', '2', '3'} or not torch.cuda.is_available():
            raise ValueError('CUDA pilot needs one explicitly assigned allowed physical GPU')
    elif device != 'cpu':
        raise ValueError('pilot device must be cpu or explicitly assigned cuda')
    solutions = [json.loads(p.read_text(encoding='utf-8')) for p in [solution_path, beta_zero_path]]
    if solutions[0]['beta'] <= 0 or solutions[1]['beta'] != 0:
        raise ValueError('both nuisance and beta-zero controls are required')
    if any(s.get('layers') != solutions[0].get('layers') or
           s.get('layer_normalization') != solutions[0].get('layer_normalization') for s in solutions):
        raise ValueError('paired controls must use the exact same ordered layer input')
    for key in ['temperature_fit_only', 'score_weight', 'distribution_weight',
                'weight_floor_mass', 'weight_cap_factor', 'fixed_feature_budget']:
        if solutions[0][key] != solutions[1][key]:
            raise ValueError('paired controls differ beyond nuisance: '+key)
    manifests = [project_path(p.parent/'fit_manifest.json', root) for p in [solution_path, beta_zero_path]]
    sources = {str(p.relative_to(root)): sha256(p) for p in [*paths, *manifests]}
    # weights_only rejects arbitrary pickle objects. No test/prediction files are read.
    cache = torch.load(cache_path, map_location='cpu', weights_only=True)
    mode = solutions[0]['layer_normalization']
    source = cache[mode]
    for path in manifests:
        validate_normal_manifest(source['manifest'], json.loads(path.read_text(encoding='utf-8')))
    layers = solutions[0]['layers']
    if not layers or len(layers) != len(set(layers)):
        raise ValueError('nonempty unique layer sequence required')
    join = lambda field: torch.cat([source[field][layer] for layer in layers], -1).to(device)
    x, heldout = join('patches'), join('heldout')
    groups, heldout_groups = source['groups'].to(device), source['heldout_groups'].to(device)
    split = source['manifest']['normal_selection_split']
    if groups.cpu().tolist() != split['fit_patch_groups'] or heldout_groups.cpu().tolist() != split['holdout_patch_groups']:
        raise ValueError('normal tensor groups do not match source manifest')
    budget = min(cfg['feature_budget'], x.shape[1])
    if any(s['fixed_feature_budget'] != budget for s in solutions):
        raise ValueError('pilot must retain the original feature budget')
    results = {}
    for label, solution in zip(['nuisance', 'beta_zero'], solutions):
        print(f'[normal-tail] {label}: {len(solution["candidate_pool"])} frozen candidates; K={budget}', flush=True)
        progress = lambda done, total: print(f'[normal-tail] {label}: validated {done}/{total} candidates', flush=True)
        results[label] = select_normal_tail(solution, x, heldout, groups, heldout_groups, progress=progress, **settings)
    for path in [*paths, *manifests]:
        if sha256(path) != sources[str(path.relative_to(root))]:
            raise ValueError('normal source changed during pilot')
    partial = output.with_name(output.name+'.partial')
    partial.mkdir(parents=True)
    for label, result in results.items():
        save_json(result, partial/f'{label}-selection.json')
    specs = {}
    for label, solution in zip(['nuisance', 'beta_zero'], solutions):
        baseline = solution['candidate_pool'][solution['candidate_id']]
        controls = {'mean': {'active': baseline['active'], 'scales': [p**.5 for p in baseline['weights']]},
                    'tail': results[label]['representation']}
        for name, obj in controls.items():
            specs[f'{label}_{name}'] = {'layers': layers, 'layer_normalization': mode,
                'kind': 'uvarfs', 'obj': obj, 'pilot_only': True}
        specs[f'{label}_tail_uniform'] = {'layers': layers, 'layer_normalization': mode,
            'kind': 'selected_raw', 'obj': {'active': controls['tail']['active']}, 'pilot_only': True}
    save_json(specs, partial/'representation-specs.json')
    save_json(cfg, partial/'fixed-config.json')
    report = {'pilot_version': PILOT_VERSION, 'stage': 'normal_only_complete',
        'anomaly_performance_assessed': False, 'full_all_six_three_fit_complete': False,
        'default_main_replaced': False, 'source_sha256': sources,
        'code_fingerprint': code_fingerprint(root), 'feature_budget': budget,
        'normal_fit_rows': len(x), 'normal_holdout_rows': len(heldout),
        'labels_or_masks_used': False, 'new_backbone_forwards': 0,
        'nuisance_or_perturbation_tensors_read_by_selector': False,
        'results': {label: {k: result[k] for k in ['baseline_candidate_id', 'candidate_id', 'changed_candidate',
                    'baseline', 'selected', 'selection_seconds']} for label, result in results.items()}}
    save_json(report, partial/'report.json')
    save_json({p.name: {'bytes': p.stat().st_size, 'sha256': sha256(p)} for p in partial.iterdir() if p.is_file()}, partial/'files-sha256.json')
    partial.rename(output)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--normal-cache', type=Path, required=True)
    parser.add_argument('--solution', type=Path, required=True)
    parser.add_argument('--beta-zero-solution', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path('configs/normal_tail_pilot.yaml'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = parser.parse_args()
    run(args.normal_cache, args.solution, args.beta_zero_solution, args.config,
        args.out, Path(__file__).resolve().parents[1], args.device)


if __name__ == '__main__':
    main()
