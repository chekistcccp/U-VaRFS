"""Opt-in normal-only tail selection from an already frozen matching pool.

This is a pilot selector, not a replacement for the registered Main. It never
generates supports, changes weights, or accesses anomaly labels/predictions.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from .matching_uvarfs import MATCHING, MatchingObjective, _features
from .numerics import precise_matmul

PILOT_VERSION = 'normal-tail-selection-pilot-v1'
VALIDATION_FIELDS = {'validation_valid', 'validation_error', 'validation_objective',
                     'validation_terms', 'selection_score'}


def image_top_scores(scores, groups, fraction):
    """Match detector ceil/Top-K, independently for each sampled image."""
    if not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError('invalid image top fraction')
    if scores.ndim != 1 or groups.shape != scores.shape or not torch.isfinite(scores).all():
        raise ValueError('invalid image scores/groups')
    ids = groups.unique(sorted=True)
    if not len(ids):
        raise ValueError('empty image groups')
    values, counts = [], []
    for group in ids:
        part = scores[groups == group]
        k = max(1, math.ceil(len(part)*fraction))
        values.append(part.topk(k).values.mean())
        counts.append(k)
    return torch.stack(values), counts


def fractional_cvar(values, fraction):
    """Exact empirical upper-tail mean, including fractional boundary mass."""
    if (values.ndim != 1 or not len(values) or not torch.isfinite(values).all()
            or not math.isfinite(fraction) or not 0 < fraction <= 1):
        raise ValueError('invalid tail risk')
    ordered = values.sort(descending=True).values
    mass = len(values)*fraction
    whole = int(math.floor(mass))
    total = ordered[:whole].sum()
    if whole < len(values):
        total = total+(mass-whole)*ordered[whole]
    return float(total/mass)


def cluster_mean_se(losses, groups):
    """Image-cluster sandwich SE of a patch mean; a selection heuristic, not CI."""
    ids = groups.unique()
    if losses.ndim != 1 or groups.shape != losses.shape or len(ids) < 2 or not torch.isfinite(losses).all():
        raise ValueError('mean SE needs at least two normal images')
    residuals = torch.stack([(losses[groups == g]-losses.mean()).sum() for g in ids])
    return float((len(ids)/(len(ids)-1)*residuals.square().sum()).sqrt()/len(losses))


@torch.no_grad()
@precise_matmul()
def dense_normal_scores(x, groups, chunk=256):
    if isinstance(chunk, bool) or not isinstance(chunk, int) or chunk < 1:
        raise ValueError('invalid dense normal chunk')
    _features(x, x, groups, groups)
    z = F.normalize(x.double(), dim=1)
    values = []
    for start in range(0, len(x), chunk):
        allowed = groups[start:start+chunk, None] != groups[None, :]
        if int(allowed.sum(1).min()) < 2:
            raise ValueError('two other-image reference patches required')
        cosine = (z[start:start+chunk] @ z.T).masked_fill(~allowed, -torch.inf)
        values.append((1-cosine.max(1).values).clamp_min(0))
    return torch.cat(values)


@torch.no_grad()
@precise_matmul()
def candidate_query_losses(engine, active, weights):
    """Independently recompute student nearest neighbors and per-query loss."""
    active = torch.as_tensor(active, device=engine.q.device, dtype=torch.long)
    p = torch.as_tensor(weights, device=engine.q.device, dtype=torch.float64)
    reference = engine.r[:, active]
    rb = reference.square() @ p
    if bool((rb <= 1e-20).any()):
        raise ValueError('zero sparse reference rows')
    losses, scores = [], []
    for block, (allowed, teacher, log_teacher) in enumerate(engine.teacher):
        start = block*engine.chunk
        query = engine.q[start:start+engine.chunk, active]
        qa = query.square() @ p
        if bool((qa <= 1e-20).any()):
            raise ValueError('zero sparse query rows')
        cosine = (query*p) @ reference.T/(qa.sqrt()[:, None]*rb.sqrt()[None, :])
        nearest = cosine.masked_fill(~allowed, -torch.inf).max(1).values
        log_student = F.log_softmax((cosine/engine.temperature).masked_fill(~allowed, -torch.inf), 1).masked_fill(~allowed, 0)
        w = engine.weights[start:start+len(query)]
        score_error = (nearest-engine.max_target[start:start+len(query)]).square()
        losses.append(w*(engine.score_weight*score_error/engine.score_scale +
                        engine.distribution_weight*(teacher*(log_teacher-log_student)).sum(1)/engine.scale))
        scores.append((1-nearest).clamp_min(0))
    return torch.cat(losses), torch.cat(scores)


def validate_frozen_pool(solution, width):
    if solution.get('objective_definition') != MATCHING or solution.get('support_generation_uses_holdout') is not False:
        raise ValueError('a fit-only matching solution is required')
    pool = solution['candidate_pool']
    fit_only = [{k: v for k, v in c.items() if k not in VALIDATION_FIELDS} for c in pool]
    digest = hashlib.sha256(json.dumps(fit_only, sort_keys=True, allow_nan=False).encode()).hexdigest()
    if digest != solution.get('candidate_pool_fit_only_sha256'):
        raise ValueError('frozen fit-only pool hash mismatch')
    k = solution['fixed_feature_budget']
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= width:
        raise ValueError('invalid fixed feature budget')
    floor, cap = solution['weight_floor_mass']/k, min(1., solution['weight_cap_factor']/k)
    for i, candidate in enumerate(pool):
        active, p = np.asarray(candidate['active']), np.asarray(candidate['weights'], dtype=float)
        if (candidate['candidate_id'] != i or len(active) != k or p.shape != (k,) or
                not np.issubdtype(active.dtype, np.integer) or len(np.unique(active)) != k or
                active.min() < 0 or active.max() >= width or not np.isfinite(p).all() or
                abs(p.sum()-1) > 1e-10 or p.min() < floor-1e-12 or p.max() > cap+1e-12):
            raise ValueError('invalid candidate support/weights')
        var = candidate['training_terms']['variability_term']
        if not math.isfinite(var) or (solution['beta'] == 0 and var != 0):
            raise ValueError('invalid nuisance term or beta-zero control')
        if candidate.get('validation_valid'):
            value, score = candidate['validation_objective'], candidate['selection_score']
            if not math.isfinite(value) or not math.isfinite(score) or not math.isclose(value+var, score, rel_tol=1e-10, abs_tol=1e-12):
                raise ValueError('invalid frozen validation selection score')
    valid = [c for c in pool if c.get('validation_valid')]
    if not valid or min(valid, key=lambda c: (c['selection_score'], c['candidate_id']))['candidate_id'] != solution['candidate_id']:
        raise ValueError('invalid baseline mean winner')
    selected = pool[solution['candidate_id']]
    if (list(solution['active']) != selected['active'] or
            not np.allclose(solution['scales'], np.sqrt(selected['weights']), atol=1e-12, rtol=1e-10) or
            not math.isclose(solution['selection_score'], selected['selection_score'], rel_tol=1e-10, abs_tol=1e-12)):
        raise ValueError('baseline representation differs from frozen winner')
    return digest


def choose_guarded_candidate(records, baseline_id, se, multiplier):
    baseline = next(c for c in records if c['candidate_id'] == baseline_id)
    threshold = baseline['selection_score']+multiplier*se
    for candidate in records:
        candidate['eligible'] = (candidate['selection_score'] <= threshold+1e-12 and
            candidate['image_score_mse_relative'] <= baseline['image_score_mse_relative']+1e-12 and
            candidate['image_score_cvar_relative'] <= baseline['image_score_cvar_relative']+1e-12)
    eligible = [c for c in records if c['eligible']]
    winner = min(eligible, key=lambda c: (c['image_score_cvar_relative'], c['selection_score'], c['candidate_id']))
    if winner['image_score_cvar_relative'] >= baseline['image_score_cvar_relative']-1e-12:
        winner = baseline
    return winner, threshold


@torch.no_grad()
@precise_matmul()
def select_normal_tail(solution, x, heldout, groups, heldout_groups, *,
                       top_fraction=.01, tail_fraction=.25, mean_se_multiplier=1., chunk=256,
                       progress=None):
    """Select, without refitting, a tail-safe candidate within one mean-loss SE.

    Guard both image-score MSE and its worst-image CVaR against the original
    mean winner. If no strict tail improvement exists, return that same winner.
    The mean-loss SE uses images as clusters and is not a performance interval.
    """
    if (isinstance(chunk, bool) or not isinstance(chunk, int) or chunk < 1 or
            not math.isfinite(mean_se_multiplier) or mean_se_multiplier < 0):
        raise ValueError('invalid tail selection configuration')
    # Validate fractions even when all candidates would fall back.
    for value in (top_fraction, tail_fraction):
        if not math.isfinite(value) or not 0 < value <= 1:
            raise ValueError('invalid tail fraction')
    groups = torch.as_tensor(groups, device=x.device)
    heldout_groups = torch.as_tensor(heldout_groups, device=x.device)
    if set(groups.cpu().tolist()) & set(heldout_groups.cpu().tolist()):
        raise ValueError('normal fit and holdout images must be disjoint')
    started = time.perf_counter()
    digest = validate_frozen_pool(solution, x.shape[1])
    engine = MatchingObjective(heldout, x, heldout_groups, groups, solution['temperature_fit_only'],
        beta=0., chunk=chunk, score_weight=solution['score_weight'],
        distribution_weight=solution['distribution_weight'])
    fit_image_scores, _ = image_top_scores(dense_normal_scores(x, groups, chunk), groups, top_fraction)
    tail_scale = max(float(fit_image_scores.square().mean()), 1e-4)
    dense_image_scores, counts = image_top_scores(1-engine.max_target, heldout_groups, top_fraction)
    records, baseline_losses = [], None
    for candidate in solution['candidate_pool']:
        if not candidate.get('validation_valid'):
            continue
        losses, scores = candidate_query_losses(engine, candidate['active'], candidate['weights'])
        value = float(losses.mean())
        if not math.isclose(value, candidate['validation_objective'], rel_tol=1e-8, abs_tol=1e-10):
            raise ValueError('normal cache and frozen validation loss disagree')
        images, _ = image_top_scores(scores, heldout_groups, top_fraction)
        error = (images-dense_image_scores).square()/tail_scale
        records.append({'candidate_id': candidate['candidate_id'],
            'selection_score': value+candidate['training_terms']['variability_term'],
            'image_score_mse_relative': float(error.mean()),
            'image_score_cvar_relative': fractional_cvar(error, tail_fraction),
            'image_score_errors_relative': error.cpu().tolist()})
        if candidate['candidate_id'] == solution['candidate_id']:
            baseline_losses = losses
        if progress is not None and (len(records) == 1 or len(records) % 10 == 0):
            progress(len(records), len(solution['candidate_pool']))
    if baseline_losses is None:
        raise ValueError('baseline is not valid on normal holdout')
    se = cluster_mean_se(baseline_losses, heldout_groups)
    baseline = next(c for c in records if c['candidate_id'] == solution['candidate_id'])
    winner, threshold = choose_guarded_candidate(records, solution['candidate_id'], se, mean_se_multiplier)
    selected = solution['candidate_pool'][winner['candidate_id']]
    representation = {'active': list(selected['active']), 'scales': np.sqrt(selected['weights']).tolist(),
                      'candidate_id': selected['candidate_id'], 'feature_dim': solution['fixed_feature_budget']}
    return {'pilot_version': PILOT_VERSION, 'selection_policy': 'normal_image_tail_with_one_cluster_se_mean_guard',
        'default_main_replaced': False, 'anomaly_performance_assessed': False,
        'sampled_patch_proxy_not_full_image_topk': True,
        'holdout_also_used_by_asls': True, 'patient_disjoint_claimed': False,
        'support_generation_uses_holdout': False, 'weights_refitted': False,
        'source_candidate_pool_fit_only_sha256': digest,
        'baseline_candidate_id': solution['candidate_id'], 'candidate_id': selected['candidate_id'],
        'changed_candidate': selected['candidate_id'] != solution['candidate_id'],
        'top_fraction': top_fraction, 'tail_fraction': tail_fraction,
        'mean_se_multiplier': mean_se_multiplier, 'mean_loss_cluster_se': se,
        'mean_loss_threshold': threshold, 'tail_scale_fit_only': tail_scale,
        'holdout_images': len(counts), 'top_patch_counts_per_image': counts,
        'baseline': copy.deepcopy(baseline), 'selected': copy.deepcopy(winner),
        'candidate_diagnostics': records, 'representation': representation,
        'selection_seconds': time.perf_counter()-started}
