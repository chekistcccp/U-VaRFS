#!/usr/bin/env python3
"""Review an explicit U-VaRFS objective proposal; NEVER fit/select features.

Only normal-training diagnostics are read from a returned experiment. Mathematical
examples and a candidate loss evaluator make the proposal reviewable. No selector,
optimizer, lambda selection, detector, or experiment configuration is changed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.analyze_results import DATASETS, source_hashes, validate_output
from uvarfs.representation import normalization_modes, normalization_prefix


def cosine_gram(x, weights=None):
    """Scale-invariant diagnostic, rejecting undefined zero-row cosines."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or not x.size or not np.isfinite(x).all():
        raise ValueError('features must be a finite, nonempty matrix')
    if weights is not None:
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape != (x.shape[1],) or not np.isfinite(weights).all() or (weights < 0).any() or weights.max() <= 0:
            raise ValueError('weights must be finite, nonnegative, and nonzero')
        # A common factor cancels exactly in cosine; avoid artificial epsilon
        # geometry changes when illustrating the all-weights-to-zero limit.
        x = x * np.sqrt(weights / weights.max())
    maximum = np.max(np.abs(x), axis=1, keepdims=True)
    if (maximum == 0).any():
        raise ValueError('cosine is undefined for a zero feature row')
    normalized = x / maximum
    normalized /= np.sqrt(np.square(normalized).sum(axis=1, keepdims=True))
    # Small mathematical review matrices do not need a BLAS runtime. This also
    # keeps review checks independent of the training runtime's thread libraries.
    return np.einsum('ij,kj->ik', normalized, normalized, optimize=False)


def proposed_loss_for_review(x, p, P, beta=.002, lam=.001):
    """Evaluate the proposed simplex + cardinality loss, without optimizing it.

    p>=0, sum(p)=1. Stability is relative to uniform full-input weights.
    Unlike an L1 penalty on a simplex, lambda*||p||_0 distinguishes supports.
    This is a mathematical review function, NOT the implemented main objective.
    """
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    if x.ndim != 2 or p.shape != (x.shape[1],) or not np.isfinite(p).all() or (p < 0).any() or not np.isclose(p.sum(), 1., rtol=0, atol=1e-12):
        raise ValueError('candidate relative weights must lie on the probability simplex')
    if P.ndim != 2 or P.shape[0] != len(p) or not np.isfinite(P).all() or (P < 0).any():
        raise ValueError('variability P must be a finite nonnegative M-by-V matrix')
    if not np.isfinite([beta, lam]).all() or beta < 0 or lam < 0:
        raise ValueError('beta and lambda must be finite and nonnegative')
    reference = cosine_gram(x)
    actual = cosine_gram(x, p)
    error = float(np.sqrt(np.square(reference - actual).sum() / np.square(reference).sum()))
    max_p = float(P.max()) if P.size else 0.
    if max_p > 0:
        scaled = P / max_p
        baseline = np.square(scaled.mean(axis=0)).sum()
        stability = float(np.square(np.einsum('ij,i->j', scaled, p, optimize=False)).sum() / baseline)
    else:
        stability = 0.
    representation = .5 * error**2
    variability = beta * stability
    sparsity = lam * np.count_nonzero(p)
    return {'evaluation_only': True, 'implemented_selector': False,
            'cosine_geometry_error': error, 'relative_variability': stability,
            'retained_dimensions': int(np.count_nonzero(p)),
            'representation_term': representation, 'variability_term': variability,
            'sparsity_term': float(sparsity), 'objective': float(representation + variability + sparsity)}


def mathematical_examples():
    """Two counterexamples and a geometry-damage control, not BMAD results."""
    base = np.array([[1., .5], [.5, 1.], [1., -1.], [-1., .3]])
    x = np.tile(base, (1, 16))
    x /= np.sqrt(np.square(x).sum(axis=1, keepdims=True))
    retained = np.zeros(x.shape[1]); retained[:2] = 1.
    reference = np.einsum('ij,kj->ik', x, x, optimize=False)
    weighted = np.einsum('ij,j,kj->ik', x, retained, x, optimize=False)
    p = retained / retained.sum()
    redundant = {'scope': 'constructed duplicate columns; not BMAD',
                 'candidate_dimensions': x.shape[1], 'retained_dimensions': 2,
                 'original_weighted_gram_error': float(np.sqrt(np.square(reference - weighted).sum() / np.square(reference).sum())),
                 **proposed_loss_for_review(x, p, np.ones((x.shape[1], 2)))}
    P = np.ones((base.shape[1], 2))
    sweep = []
    for alpha in [1., .1, .01, 1e-6]:
        weights = alpha * np.ones(base.shape[1])
        expected = cosine_gram(base)
        error = float(np.sqrt(np.square(expected - cosine_gram(base, weights)).sum() / np.square(expected).sum()))
        # A NAIVE replacement, not the current Gram objective. rscale=1 in
        # this constructed example; any fixed positive rscale has the same limit.
        naive = .5*error**2 + .002*np.square(np.einsum('ij,i->j', P, weights, optimize=False)).sum() + .001*weights.sum()
        review = proposed_loss_for_review(base, weights/weights.sum(), P)
        sweep.append({'common_weight_scale': alpha, 'cosine_error': error,
                      'naive_cosine_plus_original_penalties': float(naive),
                      'simplex_cardinality_proposal': review['objective']})
    damaged = np.array([[1., .1], [.1, 1.], [1., -1.]])
    return {'selection_candidate': False, 'scope': 'mathematical fixtures only',
            'duplicate_columns': redundant, 'naive_weight_collapse': sweep,
            'geometry_damage_control': proposed_loss_for_review(damaged, [1., 0.], np.ones((2, 2)))}


def returned_normal_diagnostics(results):
    """Summarize already-recorded NORMAL fit diagnostics, without test metrics."""
    rows = []
    for dataset in DATASETS:
        root = results / dataset
        metadata = json.loads((root / 'run_metadata.json').read_text(encoding='utf-8'))
        _, modes = normalization_modes(metadata['config'])
        for index, mode in enumerate(modes):
            folder = root if index == 0 else root / normalization_prefix(mode).rstrip('_')
            u = json.loads((folder / 'uvarfs_main.json').read_text(encoding='utf-8'))
            audit = json.loads((folder / 'normal_geometry_audit.json').read_text(encoding='utf-8'))
            normal = audit['methods']['main']
            if audit['selection_candidate'] is not False or audit['layer_feature_normalization'] != mode or u['layers'] != normal['layers']:
                raise ValueError('normal audit protocol or source layers disagree')
            if u['objective_certificate_scope'] != 'continuous_full_weights':
                raise ValueError('continuous certificate scope is missing or ambiguous')
            weights = np.asarray(u['budgeted_weights'], dtype=float)
            active = np.asarray(u['active'], dtype=int)
            if weights.ndim != 1 or not np.isfinite(weights).all() or (weights < 0).any() or (weights > 1).any():
                raise ValueError('returned box weights are invalid')
            if len(set(active.tolist())) != len(active) or (active < 0).any() or (active >= len(weights)).any():
                raise ValueError('returned support is invalid')
            np.testing.assert_allclose(np.square(u['scales']), weights[active], rtol=1e-5, atol=1e-8)
            inactive = np.ones(len(weights), dtype=bool); inactive[active] = False
            if (weights[inactive] != 0).any() or normal['feature_dim'] != len(active):
                raise ValueError('returned representation dimension disagrees')
            actual = normal.get('vs_selected_input_cosine', normal.get('vs_selected_raw_cosine'))
            rows.append({'dataset': dataset, 'layer_normalization': mode,
                         'source_experiment_version': metadata['experiment_version'],
                         'source_git_commit': metadata.get('git_head', metadata.get('git_commit')),
                         'normal_geometry_rows': audit['normal_geometry_rows'],
                         'candidate_dimensions': len(weights), 'retained_dimensions': len(active),
                         'original_weighted_gram_error': u['geometry_error'],
                         'original_geometry_feasible': u['feasible'],
                         'actual_cosine_gram_error': actual['cosine_gram_relative_error'],
                         'normal_fit_patch_1nn_agreement': actual['normal_patch_1nn_agreement'],
                         'continuous_full_box_gap': u['box_optimality_gap'],
                         'continuous_full_objective_converged': u['objective_converged'],
                         'budgeted_representation_term': u['budgeted_representation_term'],
                         'budgeted_variability_term': u['budgeted_variability_term'],
                         'budgeted_sparsity_term': u['budgeted_sparsity_term']})
    return rows


def review(results, output):
    validate_output(results, output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('choose an empty output directory; existing reports are preserved')
    before = source_hashes(results)
    rows = returned_normal_diagnostics(results)
    examples = mathematical_examples()
    if source_hashes(results) != before:
        raise ValueError('returned source files changed during review')
    payload = {'selection_candidate': False, 'implemented_selector': False,
               'reads_test_metrics_for_selection': False,
               'source_sha256': before, 'returned_normal_diagnostics': rows,
               'mathematical_examples': examples}
    lines = ['# U-VaRFS 目标升级的正常诊断评审（只读）', '',
             '当前仍是 Frozen DINOv3 + ASLS + U-VaRFS 的无异常标签医学异常检测研究。', '',
             '本脚本仅汇总已回传的正常训练诊断并检查数学例子，不读取 AUROC、异常标签或 masks 来选方法/参数，不拟合或输出新特征选择器。', '',
             '## 原目标与实际 detector 的正常几何', '',
             '| dataset | 输入 | 维数 | 原 Gram error | 实际 cosine error | normal fit patch 1NN agreement |',
             '| --- | --- | ---: | ---: | ---: | ---: |']
    for row in rows:
        lines.append(f"| {row['dataset']} | {row['layer_normalization']} | {row['retained_dimensions']}/{row['candidate_dimensions']} | "
                     f"{row['original_weighted_gram_error']:.6f} | {row['actual_cosine_gram_error']:.6f} | {row['normal_fit_patch_1nn_agreement']:.4f} |")
    lines += ['', '上表为服务器保存的正常拟合诊断，未回传原始 fit features，不能独立重算该正常 Gram。1NN 仅排除自身，包含同图其它 patch，不能当作官方 memory/test 性能。连续 box gap 的证书作用域是完整连续解，不能证明预算后支持集最优。', '',
              '## 两个数学问题', '',
              '32 维由两维重复 16 次构成，保留一组两维：原 weighted-Gram error=0.9375，而实际 cosine error≈0。该例仅证明目标错配可能存在，不代表真实医学数据有相同冗余。', '',
              '直接替换为 cosine loss 后沿用原惩罚：将所有权重乘以 a，cosine 不变，但 variability 和 L1 分别按 a²/a 缩小。a→0 导致趋零退化，a=0 的 cosine 未定义。', '',
              '| common scale | naive cosine + 原惩罚 | simplex + cardinality 评审目标 |',
              '| ---: | ---: | ---: |']
    for item in examples['naive_weight_collapse']:
        lines.append(f"| {item['common_weight_scale']:g} | {item['naive_cosine_plus_original_penalties']:.9g} | {item['simplex_cardinality_proposal']:.9g} |")
    lines += ['', '## 拟议改法', '',
              '在 sum(p)=1 的非负相对权重上保留实际 cosine geometry；按 full-input uniform weights 标定 variability；用 λ·非零维数表示稀疏性。提案历史记录见 UV_COSINE_OBJECTIVE_PROPOSAL.md，已批准的 v12 实现见 COSINE_UVARFS_UPGRADE.md。', '',
              '本脚本仅评审原 Gram 回传的正常诊断，不拟合或改变 selector；新主目标已按用户批准接入 v12，并保留原目标控制。数学例子的正向几何结果不能宣称真实 AUROC/AUPRO 改善，L2 主输入切换仍不在目标升级批准范围内。', '',
              f'回传源文件共 {len(before)} 个，读取前后 SHA256 一致。新报告仅留本地；不改写历史实验。', '']
    output.mkdir(parents=True, exist_ok=True)
    (output / 'review.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (output / 'review.md').write_text('\n'.join(lines), encoding='utf-8')
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    payload = review(args.results, args.out)
    print(f"Reviewed {len(payload['returned_normal_diagnostics'])} normal-input branches; "
          f"{len(payload['source_sha256'])} source hashes preserved; selector unchanged.")


if __name__ == '__main__':
    main()
