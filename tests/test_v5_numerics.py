from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

from scripts.analyze_results import WeightedRanking, paired_bootstrap
from uvarfs.asls import LayerGramGeometry, cosine_gram, fit_asls
from uvarfs.metrics import PixelAccumulator, prepare_pixel_mask
from uvarfs.numerics import precise_matmul
from uvarfs.protocol import validate_results_version, code_fingerprint
from uvarfs.u_varfs import _objective_certificate, _batched_fista, fit_uvarfs


def test_layer_inner_products_preserve_gram_error_and_gradient():
    torch.manual_seed(11)
    features={layer:torch.randn(32,5) for layer in range(1,5)}
    layers=list(features)
    geometry=LayerGramGeometry(features,layers)
    grams=torch.stack([cosine_gram(features[layer]) for layer in layers])
    consensus=grams.mean(0)
    probabilities=torch.tensor([.2,.4,.6,.8],requires_grad=True)
    direct=((probabilities[:,None,None]*grams).mean(0)-consensus).norm()/consensus.norm()
    equivalent=geometry.error((probabilities-1)/4)
    torch.testing.assert_close(equivalent.float(),direct,rtol=1e-6,atol=1e-6)
    expected=torch.autograd.grad(direct,probabilities,retain_graph=True)[0]
    actual=torch.autograd.grad(equivalent,probabilities)[0]
    torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
    selected=[0,2]
    error=(grams[selected].mean(0)-consensus).norm()/consensus.norm()
    assert float(geometry.subset_error(selected))==pytest.approx(float(error),rel=1e-6)
    off=~torch.eye(32,dtype=torch.bool)
    assert geometry.off_mean==pytest.approx(float(consensus[off].mean()),abs=1e-7)
    assert geometry.off_std==pytest.approx(float(consensus[off].std(unbiased=False)),abs=1e-7)


def test_pooled_geometry_can_hide_normal_patch_geometry():
    # All image means are the same. Local normal geometries are different.
    positions=torch.arange(48)*2*torch.pi/48
    patch={layer:torch.stack([torch.ones(48),2*torch.cos(layer*positions),2*torch.sin(layer*positions)],1)
           for layer in range(1,7)}
    pooled={layer:features.mean(0).expand(8,-1) for layer,features in patch.items()}
    cfg={'steps':15,'min_layers':2,'max_layers':6,'geometry_tolerance':.05}
    variability={layer:0 for layer in patch}
    old=fit_asls(pooled,variability,{**cfg,'geometry_representation':'pooled'})
    alternative=fit_asls(patch,variability,{**cfg,'geometry_representation':'patch'})
    assert len(old['selected_layers'])==2
    assert len(alternative['selected_layers'])==6
    assert alternative['selection_path'][0]['geometry_error']>.05
    assert alternative['feasible']


def test_precise_matmul_restores_policy_even_on_exception():
    previous=torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision('high')
    try:
        with pytest.raises(RuntimeError):
            with precise_matmul():
                assert torch.get_float32_matmul_precision()=='highest'
                raise RuntimeError('test')
        assert torch.get_float32_matmul_precision()=='high'
    finally:
        torch.set_float32_matmul_precision(previous)


def test_box_gap_bounds_original_quadratic_suboptimality():
    h=torch.diag(torch.tensor([.3,.6],dtype=torch.float64))
    p=torch.zeros(2,0,dtype=torch.float64)
    lambdas=[.02,.1]
    weights=torch.tensor([[.3,.8],[.6,.4]],dtype=torch.float64)
    certificate=_objective_certificate(weights,h,p,0,0,lambdas)
    optimal=torch.clamp(1-torch.tensor(lambdas)/h.diagonal()[:,None],0,1)
    reference=_objective_certificate(optimal,h,p,0,0,lambdas)
    assert torch.all(certificate['objective']-reference['objective']<=certificate['box_optimality_gap']+1e-12)
    torch.testing.assert_close(reference['box_optimality_gap'],torch.zeros(2,dtype=torch.float64),atol=1e-8,rtol=1e-8)


def test_restarted_fista_preserves_solution_and_reports_certificate():
    h=torch.diag(torch.tensor([.001,.1,1],dtype=torch.float64))
    p=torch.zeros(3,0,dtype=torch.float64)
    lambdas=[.00001,.0001]
    weights,_,_,diagnostics=_batched_fista(h,p,0,0,lambdas,3000,1e-9,10,3000)
    expected=torch.clamp(1-torch.tensor(lambdas)/h.diagonal()[:,None],0,1)
    torch.testing.assert_close(weights,expected,atol=2e-6,rtol=2e-6)
    assert sum(diagnostics['restart_counts'])>0
    assert all(diagnostics['objective_converged'])


def test_lambda_zero_is_diagnostic_and_never_enters_selection_path():
    cfg={'beta':.1,'lambda_grid':[.01,.001],'min_features':2,'max_features':4,
         'geometry_tolerance':.05,'max_iter':300,'tol':1e-6,'diagnose_infeasible':True}
    result=fit_uvarfs(torch.eye(4),torch.ones(2,4),cfg)
    assert not result['feasible']
    assert result['lambda'] in cfg['lambda_grid']
    assert [point['lambda'] for point in result['lambda_path']]==cfg['lambda_grid']
    assert result['zero_lambda_diagnostic']['lambda']==0
    assert not result['zero_lambda_diagnostic']['selection_candidate']
    assert result['zero_lambda_diagnostic']['geometry_error']>.05


def test_threshold_counting_is_exact_at_ties_and_endpoints():
    thresholds=np.array([2,1,.5,0],dtype=np.float64)
    values=np.array([-1,0,.25,.5,1,1,1.5,2,3],dtype=np.float32)
    reference=(values[:,None]>=thresholds).sum(0)
    np.testing.assert_array_equal(PixelAccumulator._counts_ge(values,thresholds),reference)
    np.testing.assert_array_equal(PixelAccumulator._counts_ge(values[:0],thresholds),np.zeros(4))


def test_shared_mask_connectivity_and_pro_match_direct_threshold_counts(monkeypatch):
    import uvarfs.metrics as metrics
    calls=[]; label=metrics.ndimage.label
    def tracked(mask):
        calls.append(True)
        return label(mask)
    monkeypatch.setattr(metrics.ndimage,'label',tracked)
    mask=np.zeros((16,16),np.uint8)
    mask[1:4,2:5]=1; mask[9:12,10:14]=1
    prepared=prepare_pixel_mask(mask)
    rng=np.random.default_rng(9)
    for _ in range(3):
        score=rng.uniform(0,1.5,size=mask.shape).astype(np.float32)
        accumulator=PixelAccumulator(pro_thresholds=20)
        accumulator.update(mask,score,prepared)
        expected_fp=(score[mask==0,None]>=accumulator.thresholds).sum(0)
        expected_pro=sum((score.ravel()[indices,None]>=accumulator.thresholds).mean(0)
                         for indices in prepared['regions'])
        np.testing.assert_array_equal(accumulator.fp,expected_fp)
        np.testing.assert_allclose(accumulator.pro,expected_pro)
    assert len(calls)==1


def test_shared_sort_histogram_preserves_every_edge_and_endpoint():
    accumulator=PixelAccumulator()
    edges=accumulator.edges.astype(np.float32)
    values=np.concatenate([edges,np.nextafter(edges,-np.inf),
                           np.nextafter(edges,np.inf),[-1.,3.]]).astype(np.float32)
    values=np.clip(values,accumulator.edges[0],accumulator.edges[-1]-1e-7)
    histogram,counts=accumulator._histogram_and_counts(values)
    np.testing.assert_array_equal(histogram,np.histogram(values,bins=accumulator.edges)[0])
    np.testing.assert_array_equal(counts,(values[:,None]>=accumulator.thresholds).sum(0))


def test_weighted_ranking_matches_independent_sklearn_with_ties():
    labels=np.array([0,1,0,1,0,1])
    scores=np.array([.3,.3,.5,.7,.1,.5])
    weights=np.array([[1,1,1,1,1,1],[2,1,3,2,1,4]])
    auc,ap=WeightedRanking(labels,scores).metrics(weights)
    for i,row in enumerate(weights):
        assert auc[i]==pytest.approx(roc_auc_score(labels,scores,sample_weight=row))
        assert ap[i]==pytest.approx(average_precision_score(labels,scores,sample_weight=row))


def test_paired_bootstrap_preserves_identical_method_and_mean_seed_auc():
    frame=pd.DataFrame({'label':[0,0,1,1],'main':[0,.2,.8,1],
                        'same':[0,.2,.8,1],'bad':[1,.8,.2,0]})
    result=paired_bootstrap(frame,{'same':['same'],'seed_mean':['same','bad']},draws=100)
    np.testing.assert_allclose(result['same'],[0,0])
    np.testing.assert_allclose(result['seed_mean'],[.5,.5])


@pytest.mark.parametrize('old_version',['gpu-eval-v4-protocol-fixes','gpu-eval-v5-normal-audit',
                                      'gpu-eval-v6-patch-asls','gpu-eval-v7-geometry-audit'])
def test_cross_version_results_cannot_be_overwritten(tmp_path,old_version):
    (tmp_path/'liver').mkdir()
    pd.DataFrame({'method':['main'],'experiment_version':[old_version]}).to_csv(tmp_path/'liver'/'metrics.csv',index=False)
    with pytest.raises(ValueError,match='Historical results must remain intact'):
        validate_results_version(tmp_path)


def test_code_fingerprint_is_independent_of_crlf(tmp_path):
    (tmp_path/'uvarfs').mkdir(); (tmp_path/'scripts').mkdir()
    (tmp_path/'uvarfs'/'a.py').write_bytes(b'a=1\r\nb=2\r\n')
    (tmp_path/'scripts'/'run_all.py').write_bytes(b'print(1)\r\n')
    before=code_fingerprint(tmp_path)
    (tmp_path/'uvarfs'/'a.py').write_bytes(b'a=1\nb=2\n')
    (tmp_path/'scripts'/'run_all.py').write_bytes(b'print(1)\n')
    assert code_fingerprint(tmp_path)==before
