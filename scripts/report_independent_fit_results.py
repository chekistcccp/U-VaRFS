#!/usr/bin/env python3
"""Read-only report from already audited, complete three-fit BMAD returns.

Reports stay in the project's ignored reports directory. This command never
selects parameters, methods or inputs.
Image bootstrap intervals condition on one fitted model and are not fit-seed CIs.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.analyze_results import WeightedRanking

DATASETS = ['brain', 'liver', 'resc', 'oct2017', 'xray', 'camelyon16']
PIXEL = ['brain', 'liver', 'resc']
METRICS = ['image_auroc', 'image_auprc', 'pixel_auroc', 'pixel_auprc', 'aupro']
SEEDS = {42, 123, 3407}
VERSIONS = {'gpu-eval-v20-normal-local-selection': 172,
            'gpu-eval-v21-normal-matching-distribution': 262}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def verify_file(root, relative, manifest):
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or relative not in manifest:
        raise ValueError('unrecorded or unsafe original input: ' + relative)
    if not path.is_file() or sha(path) != manifest[relative]:
        raise ValueError('original input hash mismatch: ' + relative)
    return path


def validate_frame(frame, version):
    expected = VERSIONS[version]
    if set(frame.fit_seed) != SEEDS or frame.duplicated(['fit_seed', 'dataset', 'method']).any():
        raise ValueError('exactly three distinct fit seeds and unique method rows required')
    methods = None
    for (seed, dataset), rows in frame.groupby(['fit_seed', 'dataset']):
        current = set(rows.method)
        if dataset not in DATASETS or len(current) != expected:
            raise ValueError('incomplete method/dataset coverage')
        if methods is not None and current != methods:
            raise ValueError('method sets differ between independent dataset runs')
        methods = current
        if not np.isfinite(rows[['image_auroc', 'image_auprc']]).all().all():
            raise ValueError('image metric missing')
        if dataset in PIXEL and not np.isfinite(rows[METRICS]).all().all():
            raise ValueError('pixel metric missing')
        if dataset not in PIXEL and not rows[METRICS[2:]].isna().all().all():
            raise ValueError('pixel metric fabricated for no-mask dataset')
    if set(map(tuple, frame[['fit_seed', 'dataset']].drop_duplicates().to_numpy())) != {
            (s, d) for s in SEEDS for d in DATASETS}:
        raise ValueError('full all-six reporting required')
    values = frame[METRICS].to_numpy(dtype=float)
    if np.any((values[np.isfinite(values)] < 0) | (values[np.isfinite(values)] > 1)):
        raise ValueError('metrics must be fractions in [0,1]')
    if version.endswith('v21-normal-matching-distribution'):
        needed = {'main', 'asls_no_score_uvarfs', 'asls_no_distribution_uvarfs',
                  'asls_no_variability_uvarfs', 'local_main'}
        if not needed <= methods:
            raise ValueError('v21 causal/previous-main controls missing')
    return methods


def families(methods, prefix, random_seeds):
    names = ['main', 'asls_raw', 'asls_pca', 'asls_selected_raw',
             'asls_no_variability_uvarfs', 'asls_no_score_uvarfs',
             'asls_no_distribution_uvarfs', 'asls_no_rank_uvarfs',
             'local_main', 'previous_main', 'fixed4_raw', 'all_raw', 'last_raw']
    result = {prefix + n: [prefix + n] for n in names if prefix + n in methods}
    for label, stem in [('Random256 mean', 'asls_random_seed'), ('Random-K layer mean', 'randomk_raw_seed')]:
        group = [prefix + stem + str(s) for s in random_seeds]
        if any(m not in methods for m in group):
            raise ValueError('complete predefined five-random family required')
        result[prefix + label] = group
    return result


def build_tables(frame, groups):
    rows = []
    random_rows = []
    for (seed, dataset), block in frame.groupby(['fit_seed', 'dataset']):
        indexed = block.set_index('method')
        for label, members in groups.items():
            value = indexed.loc[members, METRICS].mean()
            rows.append({'fit_seed': int(seed), 'dataset': dataset, 'family': label, **value.to_dict()})
            if len(members) > 1:
                random_rows.append({'fit_seed': int(seed), 'dataset': dataset, 'family': label,
                    'random_subset_count': len(members),
                    **{m + '_within_fit_sd': indexed.loc[members, m].std(ddof=1) for m in METRICS}})
    selected = pd.DataFrame(rows)
    macro = selected.groupby(['fit_seed', 'family'])[METRICS].mean().reset_index()
    deltas = []
    for (seed, dataset), block in selected.groupby(['fit_seed', 'dataset']):
        indexed = block.set_index('family')
        for family in indexed.index:
            prefix = 'raw_input_' if family.startswith('raw_input_') else ''
            target = prefix + 'main'
            if family == target:
                continue
            differences = indexed.loc[target, METRICS] - indexed.loc[family, METRICS]
            deltas.append({'fit_seed': int(seed), 'dataset': dataset, 'target': target,
                           'comparison': family, **differences.to_dict()})
    deltas = pd.DataFrame(deltas)
    macro_delta = deltas.groupby(['fit_seed', 'target', 'comparison'])[METRICS].mean().reset_index()
    return selected, macro, deltas, macro_delta, pd.DataFrame(random_rows)


def image_bootstrap(frame, groups, target, draws, rng_seed):
    y = frame.label.to_numpy(dtype=int)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError('two classes needed for stratified image bootstrap')
    classes = [np.flatnonzero(y == i) for i in (0, 1)]
    comparisons = {k: v for k, v in groups.items() if k != target}
    names = sorted({target} | {n for members in comparisons.values() for n in members})
    rankings = {n: WeightedRanking(y, frame[n]) for n in names}
    samples = {k: [[], []] for k in comparisons}
    rng = np.random.default_rng(rng_seed)
    for start in range(0, draws, 16):
        size = min(16, draws - start)
        weights = np.empty((size, len(y)), dtype=float)
        for positions in classes:
            weights[:, positions] = rng.multinomial(len(positions), np.full(len(positions), 1 / len(positions)), size)
        values = {n: ranking.metrics(weights) for n, ranking in rankings.items()}
        for family, members in comparisons.items():
            for m in (0, 1):
                baseline = np.mean([values[n][m] for n in members], axis=0)
                samples[family][m].extend((values[target][m] - baseline).tolist())
    rows = []
    for family, values in samples.items():
        for metric, samples_for_metric in zip(METRICS[:2], values):
            low, high = np.percentile(samples_for_metric, [2.5, 97.5])
            rows.append({'target': target, 'comparison': family, 'metric': metric,
                'ci95_lower': low, 'ci95_upper': high, 'draws': draws,
                'resampling_unit': 'stratified image, conditional on one fit'})
    return rows


def table(frame, columns):
    out = ['| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join(['---'] * len(columns)) + ' |']
    for _, row in frame.iterrows():
        out.append('| ' + ' | '.join(str(row[c]) for c in columns) + ' |')
    return '\n'.join(out)


def summary_table(by_seed, keys, signed=False):
    result = []
    for key, block in by_seed.groupby(keys, sort=False):
        key = (key,) if not isinstance(key, tuple) else key
        row = dict(zip(keys, key))
        for metric in METRICS:
            values = block[metric].dropna()
            if len(values) != 3:
                raise ValueError('summary requires three independent fit values for every applicable metric')
            mean, sd = 100 * values.mean(), 100 * values.std(ddof=1)
            row[metric] = f'{mean:+.3f} ± {sd:.3f}' if signed else f'{mean:.3f} ± {sd:.3f}'
        result.append(row)
    return pd.DataFrame(result)


def report(args):
    audit = args.audit_dir.resolve()
    roots = [r.resolve() for r in args.seed_results]
    out = args.out.resolve()
    if not out.is_relative_to(REPO/'reports') or any(out.is_relative_to(r) or r.is_relative_to(out) for r in roots + [audit]):
        raise ValueError('report output must be in project reports and separate from original/audit directories')
    if out.exists():
        raise ValueError('choose a new report directory; no overwrite')
    frame_path = audit / 'independent_seed_metrics.csv'
    frame = pd.read_csv(frame_path)
    versions = set(frame.experiment_version)
    if len(versions) != 1 or next(iter(versions)) not in VERSIONS:
        raise ValueError('mixed or unrecognized experiment version')
    version = next(iter(versions))
    methods = validate_frame(frame, version)
    tracked = {str(frame_path): sha(frame_path)}
    configurations, codes, root_by_seed, manifests = set(), set(), {}, {}
    for root in roots:
        meta = read(root / DATASETS[0] / 'run_metadata.json')
        seed = int(meta['config']['seed'])
        if seed in root_by_seed:
            raise ValueError('duplicate independent root seed')
        root_by_seed[seed] = root
        record = audit / f'seed{seed}' / 'analysis_data.json'
        data = read(record)
        if not data['all_six_complete'] or set(data['completed_datasets']) != set(DATASETS) or data['experiment_version'] != version:
            raise ValueError('complete matching protocol audit required')
        tracked[str(record)] = sha(record)
        manifest = data['source_sha256']; manifests[seed] = manifest
        for dataset in DATASETS:
            path = verify_file(root, dataset + '/run_metadata.json', manifest)
            meta = read(path); tracked[str(path)] = manifest[dataset + '/run_metadata.json']
            if int(meta['config']['seed']) != seed or meta['experiment_version'] != version:
                raise ValueError('seed/version metadata mismatch')
            cfg = dict(meta['config']); cfg.pop('seed'); cfg.pop('results_dir')
            configurations.add(json.dumps(cfg, sort_keys=True)); codes.add(meta['code_fingerprint'])
            metrics_path = verify_file(root, dataset + '/metrics.csv', manifest)
            tracked[str(metrics_path)] = manifest[dataset + '/metrics.csv']
            original = pd.read_csv(metrics_path).sort_values('method').reset_index(drop=True)
            missing = set(METRICS) - set(original.columns)
            if missing and (dataset in PIXEL or not missing <= set(METRICS[2:])):
                raise ValueError('original required metrics missing')
            for metric in missing:
                original[metric] = np.nan
            supplied = frame[(frame.fit_seed == seed) & (frame.dataset == dataset)].sort_values('method').reset_index(drop=True)
            if list(original.method) != list(supplied.method) or not np.allclose(original[METRICS], supplied[METRICS], atol=1e-12, rtol=1e-12, equal_nan=True):
                raise ValueError('aggregate metrics differ from audited original')
    if set(root_by_seed) != SEEDS or len(configurations) != 1 or len(codes) != 1:
        raise ValueError('exactly three frozen same-protocol fit roots required')
    random_seeds = json.loads(next(iter(configurations)))['random_baseline_seeds']
    if len(random_seeds) != 5 or len(set(random_seeds)) != 5:
        raise ValueError('predefined five random subset seeds required')
    groups = families(methods, '', random_seeds) | families(methods, 'raw_input_', random_seeds)
    selected, macro, delta, macro_delta, random_sd = build_tables(frame, groups)
    intervals = []
    if args.bootstrap:
        for seed in sorted(SEEDS):
            for index, dataset in enumerate(DATASETS):
                root = root_by_seed[seed]; relative = dataset + '/image_predictions.csv'
                path = verify_file(root, relative, manifests[seed]); tracked[str(path)] = manifests[seed][relative]
                wanted = {'label', 'image_path'} | {n for members in groups.values() for n in members}
                predictions = pd.read_csv(path, usecols=lambda c: c in wanted)
                if predictions.image_path.duplicated().any():
                    raise ValueError('duplicate evaluation images')
                block = frame[(frame.fit_seed == seed) & (frame.dataset == dataset)].set_index('method')
                for method in wanted - {'label', 'image_path'}:
                    auc, ap = WeightedRanking(predictions.label, predictions[method]).metrics(np.ones(len(predictions)))
                    if not np.allclose([auc[0], ap[0]], block.loc[method, METRICS[:2]].to_numpy(dtype=np.float64), atol=1e-10, rtol=1e-10):
                        raise ValueError('prediction point estimates disagree')
                for prefix in ('', 'raw_input_'):
                    branch = {k: v for k, v in groups.items() if k.startswith('raw_input_') == bool(prefix)}
                    for row in image_bootstrap(predictions, branch, prefix + 'main', args.bootstrap, seed + 1000003 * index):
                        intervals.append({'fit_seed': seed, 'dataset': dataset, **row})
                print(f'image intervals seed={seed} dataset={dataset} complete', flush=True)
    if any(sha(path) != digest for path, digest in tracked.items()):
        raise ValueError('read-only report inputs changed during analysis')
    out.mkdir(parents=True)
    for name, data in [('method_family_by_fit_seed', selected), ('macro_by_fit_seed', macro),
                       ('paired_delta_by_fit_seed', delta), ('paired_macro_by_fit_seed', macro_delta),
                       ('random_subset_within_fit_sd', random_sd), ('paired_image_intervals', pd.DataFrame(intervals))]:
        data.to_csv(out / (name + '.csv'), index=False)
    selected.groupby(['dataset', 'family'])[METRICS].agg(['mean', 'std', 'count']).to_csv(out / 'dataset_fit_mean_sd.csv')
    delta.groupby(['dataset', 'target', 'comparison'])[METRICS].agg(['mean', 'std', 'count']).to_csv(out / 'dataset_paired_delta_mean_sd.csv')
    shared = ['seconds', 'fit_seconds', 'memory_seconds', 'eval_seconds', 'peak_gpu_gb']
    costs = frame[frame.method == 'main'][['fit_seed', 'dataset', 'timing_scope', *shared,
        'selected_layer_count', 'feature_dim', 'candidate_dim', 'memory_size',
        'uvarfs_fit_seconds', 'matching_and_aggregation_seconds', 'memory_vector_fp16_mib']].copy()
    costs['cost_scope_note'] = 'dataset times/peak shared by all methods; U-VaRFS fit and matching fields separate'
    costs.to_csv(out / 'costs_with_scope.csv', index=False)
    macro_display = summary_table(macro, ['family'])
    paired_display = summary_table(macro_delta, ['target', 'comparison'], signed=True)
    lines = ['# 完整独立三种子性能报告', '', f'版本 `{version}`；独立 fit/memory seeds 42、123、3407，全六完整。', '',
        '以下为百分数或成对差值的百分点；mean/样本SD先按每个fit seed计算，再对三个独立fit汇总。六集检测等权，定位仅Brain/Liver/RESC等权。', '',
        '## 宏平均（百分数）', '', table(macro_display, ['family', *METRICS]), '',
        '## Main与预先指定对照的成对宏平均差（百分点）', '', table(paired_display, ['target', 'comparison', *METRICS]), '',
        '## 解释边界', '',
        '- 正常拟合目标、压缩率和提速不能补偿检测/定位下降；不依据本报告更改当前参数、输入或方法。',
        '- Random256/Random-K先在同一fit内平均五个预设子集；表内SD为跨fit SD，五Random子集的SD另见random_subset_within_fit_sd.csv。',
        '- raw_input是预设配对消融，不按test择优更名Main；fixed/all/last/Random-K层是层选择诊断，同层Raw/PCA/Random和β0才检验特征选择。',
        '- local_main是同轮完整v20目标控制，不冒充历史v20数值复现；完整原始方法表在已审计independent_seed_metrics.csv。',
        '- 图像区间只按同一拟合内分层成对重采样图像，并对五Random的指标取均值，未对其分数做集成。不是fit种子或患者置信区间；n=3与患者分组不可用限制推断。',
        '- 历史test已多次用于开发，不声称未经接触的最终测试；多方法、多指标图像区间未经多重检验校正。',
        '- 未保存完整像素预测，故不生成像素bootstrap或患者层级显著性结论。逐dataset/seed全部定位差异仍完整报告。',
        '- dataset总fit/memory/eval/peak成本含完整控制，不能归为Main独占开销；Main特征fit与matching成本分别标明。', '',
        f'图像bootstrap次数：{args.bootstrap}（0表示本次未附加区间；原严格审计已独立核验所有图像点估计）。', '',
        '本工具只生成证据表，不通过事后阈值宣布研究成功。实际跨模态收益、β稳定贡献和定位下降须共同审查。']
    (out / 'assessment.zh.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    (out / 'report-provenance.json').write_text(json.dumps({'experiment_version': version,
        'fit_seeds': sorted(SEEDS), 'datasets': DATASETS, 'original_and_audit_inputs_sha256': tracked,
        'code_fingerprint': next(iter(codes)), 'report_helper_sha256': sha(__file__),
        'original_inputs_unchanged': True, 'bootstrap': args.bootstrap,
        'source_training_parameters_changed': False, 'research_goal_completion_inferred': False}, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'report': str(out), 'complete_rows': len(frame), 'comparisons': len(groups),
                      'intervals': len(intervals), 'original_inputs_unchanged': True}, ensure_ascii=False))


def self_test():
    from sklearn.metrics import roc_auc_score, average_precision_score
    y = np.array([0, 1, 0, 1, 0, 1]); scores = np.array([.2, .2, .5, .5, .8, .8])
    weights = np.array([2, 1, 3, 4, 1, 2], dtype=float)
    auc, ap = WeightedRanking(y, scores).metrics(weights)
    assert np.allclose([auc[0], ap[0]], [roc_auc_score(y, scores, sample_weight=weights), average_precision_score(y, scores, sample_weight=weights)])
    predictions = pd.DataFrame({'label': y, 'main': scores, 'same1': scores, 'same2': scores})
    for row in image_bootstrap(predictions, {'main': ['main'], 'Random mean': ['same1', 'same2']}, 'main', 100, 91):
        assert row['ci95_lower'] == row['ci95_upper'] == 0
    tiny = pd.DataFrame([{'fit_seed': s, 'dataset': d, 'method': m,
        **{metric: (.4 + .1 * (m == 'main') + .001 * (s == 123)) if (metric.startswith('image') or d in PIXEL) else np.nan for metric in METRICS}}
        for s in SEEDS for d in DATASETS for m in ['main', 'asls_raw', 'random1', 'random2']])
    _, macro, _, paired, random_sd = build_tables(tiny, {'main': ['main'], 'asls_raw': ['asls_raw'], 'Random mean': ['random1', 'random2']})
    assert np.allclose(paired[METRICS], .1)
    assert len(macro) == 9 and len(random_sd) == 18
    assert summary_table(paired, ['target', 'comparison'], True).image_auroc.iloc[0] == '+10.000 ± 0.000'
    print('self-test passed: exact weighted ties/AP, identical paired intervals, fit-vs-random/macro aggregation')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audit-dir', type=Path)
    parser.add_argument('--seed-results', nargs=3, type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--bootstrap', type=int, default=2000)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test(); return
    if not args.audit_dir or not args.seed_results or not args.out or (args.bootstrap != 0 and args.bootstrap < 100):
        parser.error('audit-dir, three seed-results, new project reports out, and bootstrap=0 or >=100 required')
    report(args)

if __name__ == '__main__':
    main()
