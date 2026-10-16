import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
import yaml

from uvarfs.matching_uvarfs import MatchingObjective, fit_matching_uvarfs
from uvarfs.normal_tail_selection import (PILOT_VERSION, image_top_scores, fractional_cvar,
    cluster_mean_se, candidate_query_losses, choose_guarded_candidate, select_normal_tail)
from scripts.run_normal_tail_pilot import run, project_path, validate_normal_manifest
from test_v20_local_selection import case
from test_v21_matching_distribution import serialized, settings


def fitted(beta=.002):
    x, val, g, vg, paired = case()
    cfg = {**settings(), 'beta': beta, 'matching_exchange_max_steps': 0}
    return x, val, g, vg, serialized(fit_matching_uvarfs(x, val, g, vg, paired, cfg))


def test_topk_matches_detector_ceil_and_cvar_uses_fractional_boundary():
    scores = torch.tensor([.1, .9, .3, .4, .2], dtype=torch.float64)
    groups = torch.tensor([8, 8, 2, 2, 2])
    image, counts = image_top_scores(scores, groups, .5)
    torch.testing.assert_close(image, torch.tensor([.35, .9], dtype=torch.float64))
    assert counts == [2, 1]
    assert fractional_cvar(torch.tensor([1., 2., 7.]), .5) == pytest.approx((7+1)/1.5)
    assert fractional_cvar(torch.tensor([1., 2., 7.]), 1.) == pytest.approx(10/3)
    assert fractional_cvar(torch.tensor([1., 2., 7.]), .1) == pytest.approx(7)


def test_image_cluster_se_handles_unequal_patch_counts_without_pseudoreplication():
    values = torch.tensor([1., 2., 3., 7., 8., 9.], dtype=torch.float64)
    groups = torch.tensor([0, 0, 1, 2, 2, 2])
    residual = [sum(values[groups == g].numpy()-values.numpy().mean()) for g in [0, 1, 2]]
    expected = np.sqrt(3/2*np.square(residual).sum())/6
    assert cluster_mean_se(values, groups) == pytest.approx(expected)
    with pytest.raises(ValueError, match='two normal images'):
        cluster_mean_se(values, torch.zeros_like(groups))


def test_per_query_oracle_chunk_invariance_and_actual_student_neighbors():
    x, val, g, vg, result = fitted()
    p = torch.tensor(result['candidate_pool'][0]['weights'], dtype=torch.float64)
    active = result['candidate_pool'][0]['active']
    engine = MatchingObjective(val, x, vg, g, result['temperature_fit_only'], beta=0., chunk=3)
    loss, scores = candidate_query_losses(engine, active, p)
    value = engine.evaluate(active, p)[0]
    assert float(loss.mean()) == pytest.approx(value, abs=1e-12)
    q = F.normalize(val[:, active].double()*p.sqrt(), dim=1)
    r = F.normalize(x[:, active].double()*p.sqrt(), dim=1)
    expected = 1-(q @ r.T).max(1).values
    torch.testing.assert_close(scores, expected, atol=1e-12, rtol=1e-12)
    other = MatchingObjective(val, x, vg, g, result['temperature_fit_only'], beta=0., chunk=7)
    ll, ss = candidate_query_losses(other, active, p)
    torch.testing.assert_close(loss, ll, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(scores, ss, atol=1e-12, rtol=1e-12)
    dense = torch.arange(x.shape[1])
    ll, ss = candidate_query_losses(engine, dense, torch.full((x.shape[1],), 1/x.shape[1], dtype=torch.float64))
    assert float(ll.abs().max()) < 1e-12
    torch.testing.assert_close(ss, 1-engine.max_target, atol=1e-12, rtol=1e-12)


def test_guard_can_improve_tail_but_rejects_mean_or_image_mse_harm_and_ties():
    def record(i, mean, mse, tail):
        return dict(candidate_id=i, selection_score=mean, image_score_mse_relative=mse, image_score_cvar_relative=tail)
    records = [record(0, 1., 1., 2.), record(1, 1.05, .8, 1.),
               record(2, 1.2, .5, .5), record(3, 1.01, 1.1, .1)]
    winner, threshold = choose_guarded_candidate(records, 0, .1, 1.)
    assert winner['candidate_id'] == 1 and threshold == pytest.approx(1.1)
    assert [r['eligible'] for r in records] == [True, True, False, False]
    tied = [record(0, 1., 1., 2.), record(1, 1.01, .8, 2.)]
    assert choose_guarded_candidate(tied, 0, .1, 1.)[0]['candidate_id'] == 0


def test_frozen_support_pool_weights_and_budget_remain_read_only():
    x, val, g, vg, solution = fitted()
    before = copy.deepcopy(solution)
    result = select_normal_tail(solution, x, val, g, vg, chunk=3)
    assert solution == before
    assert result['source_candidate_pool_fit_only_sha256'] == solution['candidate_pool_fit_only_sha256']
    chosen = solution['candidate_pool'][result['candidate_id']]
    assert result['representation']['active'] == chosen['active']
    np.testing.assert_array_equal(result['representation']['scales'], np.sqrt(chosen['weights']))
    assert result['representation']['feature_dim'] == solution['fixed_feature_budget']
    assert result['selected']['image_score_cvar_relative'] <= result['baseline']['image_score_cvar_relative']+1e-12
    assert result['selected']['image_score_mse_relative'] <= result['baseline']['image_score_mse_relative']+1e-12
    assert result['selected']['selection_score'] <= result['mean_loss_threshold']+1e-12
    assert result['top_patch_counts_per_image'] == [1, 1]
    assert result['anomaly_performance_assessed'] is False and result['default_main_replaced'] is False
    changed = copy.deepcopy(solution); changed['candidate_pool'][0]['weights'][0] += .001
    with pytest.raises(ValueError, match='hash mismatch'):
        select_normal_tail(changed, x, val, g, vg)
    with pytest.raises(ValueError, match='validation loss disagree'):
        select_normal_tail(solution, x, val*torch.arange(1, 14), g, vg)
    with pytest.raises(ValueError, match='disjoint'):
        select_normal_tail(solution, x, val, g, g[:len(val)])


@pytest.mark.parametrize('kwargs', [{'top_fraction':0}, {'tail_fraction':float('nan')},
    {'mean_se_multiplier':-1}, {'chunk':True}])
def test_invalid_selection_settings_fail_before_work(kwargs):
    with pytest.raises(ValueError):
        select_normal_tail({}, torch.ones(2, 2), torch.ones(2, 2), torch.arange(2), torch.arange(2)+2, **kwargs)


def test_cli_normal_cache_pairing_safe_publication_and_original_sources(tmp_path):
    x, val, g, vg, mean = fitted()
    _, _, _, _, beta0 = fitted(0.)
    for s in [mean, beta0]: s.update(layers=[1], layer_normalization='l2')
    source = tmp_path/'results/source'; source.mkdir(parents=True)
    (tmp_path/'uvarfs').mkdir(); (tmp_path/'scripts').mkdir()
    (tmp_path/'scripts/run_all.py').write_text('# fixture source\n')
    manifest = {'normal_train_indices':list(range(8)),
        'normal_train_paths':[f'/project/data/Brain/train/good/{i}.png' for i in range(8)],
        'sampled_patches':[], 'perturbation_rng':{},
        'normal_selection_split':{'fit_patch_groups':g.tolist(), 'holdout_patch_groups':vg.tolist(),
            'holdout_used_for_gradients':False, 'holdout_used_for_selection_reference':False}}
    cache = {'l2':{'patches':{1:x}, 'heldout':{1:val}, 'groups':g, 'heldout_groups':vg, 'manifest':manifest}}
    torch.save(cache, source/'normal-inputs.pt')
    for name, obj in [('mean.json',mean), ('beta0.json',beta0), ('fit_manifest.json',manifest)]:
        (source/name).write_text(json.dumps(obj))
    cfg = {'pilot_version':PILOT_VERSION, 'feature_budget':4,
        'selector':dict(top_fraction=.01, tail_fraction=.25, mean_se_multiplier=1., chunk=3)}
    config = tmp_path/'config.yaml'; config.write_text(yaml.safe_dump(cfg))
    paths = [source/'normal-inputs.pt', source/'mean.json', source/'beta0.json', source/'fit_manifest.json']
    before = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    output = tmp_path/'results/pilot'
    report = run(paths[0], paths[1], paths[2], config, output, tmp_path)
    assert report['stage'] == 'normal_only_complete' and report['anomaly_performance_assessed'] is False
    assert report['results']['beta_zero']['baseline_candidate_id'] == beta0['candidate_id']
    specs = json.loads((output/'representation-specs.json').read_text())
    assert len(specs) == 6 and all(s['pilot_only'] for s in specs.values())
    from uvarfs.pipeline_eval import transform
    features = {1:val.reshape(2, 4, -1)}
    for spec in specs.values():
        transformed = transform(features, spec)
        assert transformed.shape == (2, 4, 4) and torch.isfinite(transformed).all()
    assert before == {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    assert not output.with_name(output.name+'.partial').exists()
    with pytest.raises(ValueError, match='fresh'):
        run(paths[0], paths[1], paths[2], config, output, tmp_path)
    with pytest.raises(ValueError, match='inside'):
        project_path(tmp_path.parent/'outside', tmp_path, output=True)
    bad = copy.deepcopy(manifest); bad['normal_train_paths'][0] = '/project/test/Ungood/a.png'
    with pytest.raises(ValueError, match='manifest mismatch'):
        validate_normal_manifest(bad, manifest)
    with pytest.raises(ValueError, match='train/good'):
        validate_normal_manifest(bad, bad)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable locally')
def test_cuda_normal_tail_metrics_match_cpu():
    x, val, g, vg, solution = fitted()
    cpu = select_normal_tail(solution, x, val, g, vg, chunk=3)
    gpu = select_normal_tail(solution, x.cuda(), val.cuda(), g.cuda(), vg.cuda(), chunk=3)
    assert cpu['candidate_id'] == gpu['candidate_id']
    assert cpu['selected']['image_score_cvar_relative'] == pytest.approx(gpu['selected']['image_score_cvar_relative'], abs=1e-10)
