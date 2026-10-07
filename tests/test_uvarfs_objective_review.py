import json
import math

import numpy as np
import pytest

from scripts.review_uvarfs_objective import (
    cosine_gram, mathematical_examples, proposed_loss_for_review,
    returned_normal_diagnostics, review,
)
from uvarfs.protocol import code_fingerprint


def test_candidate_representation_matches_independent_detector_and_stability_formula():
    rng = np.random.default_rng(57)
    x = rng.normal(size=(8, 5)); p = np.array([.1, .2, 0., .3, .4])
    P = rng.uniform(size=(5, 3)); beta = .007; lam = .02
    result = proposed_loss_for_review(x, p, P, beta, lam)
    # Independent scalar sample-space oracle, including diagonal and sign.
    expected = []; observed = []
    for a in x.tolist():
        for b in x.tolist():
            expected.append(math.fsum(v*w for v,w in zip(a,b))/math.sqrt(math.fsum(v*v for v in a)*math.fsum(v*v for v in b)))
            observed.append(math.fsum(v*w*q for v,w,q in zip(a,b,p))/math.sqrt(math.fsum(v*v*q for v,q in zip(a,p))*math.fsum(v*v*q for v,q in zip(b,p))))
    rep = .5*math.fsum((v-w)**2 for v,w in zip(expected,observed))/math.fsum(v*v for v in expected)
    selected_var = math.fsum(math.fsum(P[j,v]*p[j] for j in range(5))**2 for v in range(3))
    full_var = math.fsum(math.fsum(P[j,v]/5 for j in range(5))**2 for v in range(3))
    var = beta*selected_var/full_var
    assert result['representation_term'] == pytest.approx(rep, abs=1e-14)
    assert result['variability_term'] == pytest.approx(var, abs=1e-14)
    assert result['sparsity_term'] == pytest.approx(4*lam)
    assert result['objective'] == pytest.approx(rep+var+4*lam)
    assert result['evaluation_only'] and result['implemented_selector'] is False


def test_duplicate_geometry_and_naive_scaling_collapse_are_distinct_from_ad_evidence():
    examples = mathematical_examples()
    redundant = examples['duplicate_columns']
    assert redundant['original_weighted_gram_error'] == pytest.approx(15/16)
    assert redundant['cosine_geometry_error'] < 1e-14
    sweep = examples['naive_weight_collapse']
    assert np.diff([row['naive_cosine_plus_original_penalties'] for row in sweep]).max() < 0
    assert max(row['cosine_error'] for row in sweep) < 1e-14
    np.testing.assert_allclose([row['simplex_cardinality_proposal'] for row in sweep], sweep[0]['simplex_cardinality_proposal'])
    assert examples['geometry_damage_control']['representation_term'] > .01
    assert examples['selection_candidate'] is False


def test_geometry_is_scale_invariant_without_epsilon_and_zero_rows_are_invalid():
    x = np.array([[1., .2, -.1], [.3, 2., .5]])
    weights = np.array([.2, 0., .8])
    np.testing.assert_allclose(cosine_gram(x, weights), cosine_gram(x, weights*1e-200), atol=1e-14)
    with pytest.raises(ValueError, match='nonzero'):
        cosine_gram(x, np.zeros(3))
    with pytest.raises(ValueError, match='zero feature row'):
        cosine_gram([[0., 1.], [1., 0.]], [1., 0.])
    with pytest.raises(ValueError, match='simplex'):
        proposed_loss_for_review(x, weights*.01, np.ones((3, 2)))


@pytest.mark.parametrize('P',[np.zeros((3, 2)), np.zeros((3, 0))])
def test_zero_variability_is_well_defined(P):
    result = proposed_loss_for_review(np.ones((4, 3)), [.1, .4, .5], P)
    assert result['variability_term'] == 0 and np.isfinite(result['objective'])


def test_uniform_reference_variability_and_cardinality_reject_constant_l1_sparsity():
    rng = np.random.default_rng(61)
    x = rng.normal(size=(6, 5)); P = rng.uniform(size=(5, 2))
    dense = proposed_loss_for_review(x, np.ones(5)/5, P)
    sparse = proposed_loss_for_review(x, [.5, .5, 0., 0., 0.], P)
    assert dense['relative_variability'] == pytest.approx(1.)
    assert dense['cosine_geometry_error'] < 1e-14
    assert dense['sparsity_term']/sparse['sparsity_term'] == pytest.approx(5/2)
    rescaled = proposed_loss_for_review(x, [.5, .5, 0., 0., 0.], P*1e-100)
    assert rescaled['objective'] == pytest.approx(sparse['objective'])


@pytest.mark.parametrize('value',[-1., float('nan')])
def test_invalid_variability_is_not_silently_used(value):
    P = np.ones((2, 1)); P[0, 0] = value
    with pytest.raises(ValueError, match='variability'):
        proposed_loss_for_review(np.ones((3, 2)), [.5, .5], P)


def write_normal_fixture(root):
    # No predictions, labels, metrics.csv, masks, or raw feature matrices exist.
    from scripts.analyze_results import DATASETS
    for dataset in DATASETS:
        folder = root/dataset; folder.mkdir(parents=True)
        files = {
            'run_metadata.json': {'experiment_version':'fixture', 'git_commit':'test-source',
                                  'config':{'representation':{'layer_normalization':'none'}}},
            'uvarfs_main.json': {'layers':[2, 3], 'budgeted_weights':[1., 0., .25],
                                'active':[0, 2], 'scales':[1., .5], 'geometry_error':.2,
                                'feasible':False, 'box_optimality_gap':1e-6,
                                'objective_converged':True,
                                'objective_certificate_scope':'continuous_full_weights',
                                'budgeted_representation_term':.02,
                                'budgeted_variability_term':.001, 'budgeted_sparsity_term':.002},
            'normal_geometry_audit.json': {'selection_candidate':False,
                'layer_feature_normalization':'none', 'normal_geometry_rows':8,
                'methods':{'main':{'layers':[2, 3], 'feature_dim':2,
                    'vs_selected_input_cosine':{'cosine_gram_relative_error':.1,
                                               'normal_patch_1nn_agreement':.75}}}},
        }
        for name, data in files.items():
            (folder/name).write_text(json.dumps(data), encoding='utf-8')


def test_review_needs_only_normal_diagnostics_preserves_sources_and_rejects_overwrites(tmp_path):
    root = tmp_path/'source'; output = tmp_path/'output'
    write_normal_fixture(root)
    before = {str(p):p.read_bytes() for p in root.rglob('*.json')}
    payload = review(root, output)
    assert len(payload['returned_normal_diagnostics']) == 6
    assert payload['reads_test_metrics_for_selection'] is False
    assert {str(p):p.read_bytes() for p in root.rglob('*.json')} == before
    assert (output/'review.md').is_file()
    with pytest.raises(ValueError, match='empty output'):
        review(root, output)
    with pytest.raises(ValueError, match='outside'):
        review(root, root/'bad-output')
    path = root/'brain'/'uvarfs_main.json'
    broken = json.loads(path.read_text()); broken['scales'][1] = .8
    path.write_text(json.dumps(broken))
    with pytest.raises(AssertionError):
        returned_normal_diagnostics(root)


def test_review_script_is_outside_experiment_fingerprint(tmp_path):
    (tmp_path/'uvarfs').mkdir(); (tmp_path/'scripts').mkdir()
    (tmp_path/'uvarfs'/'fit.py').write_text('original loss')
    (tmp_path/'scripts'/'run_all.py').write_text('original runner')
    before = code_fingerprint(tmp_path)
    (tmp_path/'scripts'/'review_uvarfs_objective.py').write_text('review only')
    assert code_fingerprint(tmp_path) == before
