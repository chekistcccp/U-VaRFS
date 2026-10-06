import copy
import json
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.optimize import minimize
from sklearn.metrics import average_precision_score, roc_auc_score

from uvarfs.data import Sample, MaskMatcher, scan_split, validate_pixel_masks
from uvarfs.metrics import PixelAccumulator
from uvarfs.pipeline_fit import fit_method_specs, fit_representation
from uvarfs.pipeline_eval import build_memories, evaluate, write_summaries
from uvarfs.protocol import (EXPERIMENT_VERSION, completed_result_compatible,
                             config_fingerprint, data_fingerprint, expected_method_names)
from uvarfs.u_varfs import _batched_fista, _select_solution
from uvarfs.utils import load_config
from uvarfs.variability import NormalVariability


def save_image(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(array, dtype=np.uint8)).save(path)


@pytest.mark.parametrize('batch_sizes', [[8], [2, 2, 2, 2], [1, 3, 4]])
def test_variability_matches_population_formula_across_batches(batch_sizes):
    # Batch means differ strongly: averaging within-batch ratios is incorrect.
    normal = torch.tensor([[i * 10.0, 4.0] for i in range(8)], dtype=torch.float64)
    perturbed = {'noise': normal + torch.tensor([2.0, .25]),
                 'gamma': normal * torch.tensor([1.01, 1.02])}
    accumulator = NormalVariability(2, perturbed, epsilon=1e-6)
    offset = 0
    for n in batch_sizes:
        x = normal[offset:offset+n]
        accumulator.add_normal(x)
        for kind, y in perturbed.items():
            accumulator.add_perturbation(kind, x, y[offset:offset+n])
        offset += n
    reference = torch.stack([(normal-y).square().mean(0) /
                              (normal.var(0, unbiased=False)+1e-6)
                              for y in perturbed.values()])
    torch.testing.assert_close(accumulator.vectors().double(), reference,
                               rtol=1e-7, atol=1e-7)


def test_variability_rejects_incomplete_perturbation_coverage():
    accumulator = NormalVariability(2, ['noise', 'blur'])
    normal = torch.ones(3, 2)
    accumulator.add_normal(normal)
    accumulator.add_perturbation('noise', normal, normal)
    with pytest.raises(ValueError, match='same normal-reference patches'):
        accumulator.vectors()


def test_batched_fista_matches_independent_box_constrained_quadratic():
    rng = np.random.default_rng(8)
    x = rng.normal(size=(12, 5))
    h = (x.T @ x)**2
    h /= h.sum()
    p = rng.uniform(.01, .4, size=(5, 2))
    beta, scale = .04, 1.3
    lambdas = [.003, .02, .15]
    actual, _, _, diagnostics = _batched_fista(
        torch.tensor(h), torch.tensor(p), beta, scale, lambdas,
        max_iter=4000, tol=1e-10, check_every=10, log_every=4000)
    for j, lam in enumerate(lambdas):
        def objective(w):
            difference = 1-w
            return .5*difference @ h @ difference + beta*scale*np.linalg.norm(p.T @ w)**2 + lam*w.sum()

        def gradient(w):
            return h @ (w-1) + 2*beta*scale*p @ (p.T @ w) + lam

        reference = minimize(objective, np.ones(5)*.5, jac=gradient,
                             bounds=[(0, 1)]*5, method='L-BFGS-B',
                             options={'ftol':1e-15, 'gtol':1e-11, 'maxiter':4000})
        assert reference.success
        np.testing.assert_allclose(actual[:,j].numpy(), reference.x, atol=2e-7, rtol=2e-7)
        assert diagnostics['converged'][j]


def test_geometry_is_checked_after_budgeting_each_lambda():
    # Both continuous candidates meet tolerance; keeping only two dimensions
    # invalidates the first, but preserves the second candidate's important axes.
    h = torch.diag(torch.tensor([.45, .45, .05, .05], dtype=torch.float64))
    w = torch.tensor([[.91, 1], [.91, 1], [1, 0], [1, 0]], dtype=torch.float64)
    diagnostics = {'relative_change':[0,0], 'projected_residual':[0,0], 'converged':[True,True]}
    chosen, active, _, path = _select_solution(w, h, [.01,.001],
        {'min_features':2, 'max_features':2, 'geometry_tolerance':.4}, diagnostics)
    assert path[0]['continuous_geometry_error'] < .4
    assert path[0]['geometry_error'] > .4
    assert not path[0]['feasible']
    assert chosen == 1 and path[1]['feasible']
    assert active.tolist() == [0,1]
    assert path[1]['geometry_error'] == pytest.approx(np.sqrt(.1))
    _, _, _, strict_path = _select_solution(w, h, [.01,.001],
        {'min_features':2, 'max_features':2, 'geometry_tolerance':.05}, diagnostics)
    assert not any(point['feasible'] for point in strict_path)


def test_mask_matching_uses_relative_subfolders_suffix_and_extension(tmp_path):
    root = tmp_path/'liver'
    save_image(root/'test'/'good'/'img'/'normal.jpg', np.zeros((8,8)))
    for patient in ['patient1','patient2']:
        save_image(root/'test'/'Ungood'/'img'/patient/'slice.jpg', np.zeros((8,8)))
        save_image(root/'test'/'Ungood'/'anomaly_mask'/patient/'slice_mask.png', np.ones((8,8))*255)
    samples = scan_split(root, 'test', 'liver')
    # On Windows Ungood/ungood must not count the same abnormal class twice.
    assert len(samples) == 3
    coverage = validate_pixel_masks(samples, 'liver')
    assert coverage['matched_abnormal_masks'] == 2
    for sample in samples:
        if sample.label:
            assert sample.image.parent.name == sample.mask.parent.name


def test_mask_matching_rejects_ambiguous_global_stems(tmp_path):
    for patient in ['a','b']:
        save_image(tmp_path/'masks'/patient/'slice_mask.png', np.ones((8,8)))
    matcher = MaskMatcher(tmp_path/'masks')
    with pytest.raises(ValueError, match='Ambiguous lesion mask'):
        matcher.match(tmp_path/'images'/'slice.jpg', tmp_path/'images')


def test_pixel_metrics_match_quantized_sklearn_and_perfect_map():
    rng = np.random.default_rng(1)
    mask = rng.integers(0,2,size=(16,16),dtype=np.uint8)
    score = rng.uniform(0,2,size=mask.shape).astype(np.float32)
    accumulator = PixelAccumulator(bins=64)
    accumulator.update(mask, score)
    metrics = accumulator.finalize()
    quantized = np.floor(score/(2/64))
    assert metrics['pixel_auroc'] == pytest.approx(roc_auc_score(mask.ravel(),quantized.ravel()))
    assert metrics['pixel_auprc'] == pytest.approx(average_precision_score(mask.ravel(),quantized.ravel()))
    perfect = PixelAccumulator(bins=64, pro_thresholds=20)
    mask = np.zeros((16,16),np.uint8)
    mask[4:8,4:8] = 1
    perfect.update(mask, mask.astype(np.float32))
    assert perfect.finalize() == pytest.approx({'pixel_auroc':1, 'pixel_auprc':1, 'aupro':1})


def test_aupro_interpolates_at_fpr_limit():
    accumulator = PixelAccumulator(pro_thresholds=3,max_fpr=.3)
    accumulator.regions = 1
    accumulator.normal_pixels = 100
    accumulator.fp[:] = [0,50,100]
    accumulator.pro[:] = [0,.5,1]
    # Integral of PRO=FPR over [0,.3], normalized by .3.
    assert accumulator.finalize()['aupro'] == pytest.approx(.15)


@pytest.mark.parametrize('split,label', [('train',1),('test',0),('valid',0)])
def test_normal_only_fit_and_memory_guards(split,label,tmp_path):
    sample = Sample(tmp_path/'unused.png',label,None,split,'liver')
    with pytest.raises(ValueError,match='normal-reference training split'):
        fit_representation(None,[sample],{},tmp_path)
    with pytest.raises(ValueError,match='normal-reference training split'):
        build_memories(None,[sample],{}, {})


def test_missing_pixel_mask_fails_before_index_build(monkeypatch,tmp_path):
    def forbidden(*args):
        pytest.fail('evaluation must validate masks before building an index')
    monkeypatch.setattr('uvarfs.pipeline_eval.make_index', forbidden)
    sample = Sample(tmp_path/'missing.png',1,None,'test','liver')
    with pytest.raises(ValueError,match='missing masks'):
        evaluate(None,[sample],{'main':{}},{'main':np.ones((1,2))},{'eval':{}})


def test_checkpoint_requires_version_exact_methods_fingerprint_and_pixel_metrics():
    cfg = {'seed':42,'methods':['main','randomk_raw'], 'random_baseline_seeds':[42,123]}
    frame = pd.DataFrame({'method':expected_method_names(cfg),
                          'experiment_version':EXPERIMENT_VERSION,
                          **{k:.8 for k in ['image_auroc','image_auprc','pixel_auroc','pixel_auprc','aupro']}})
    metadata = {'config_fingerprint':config_fingerprint(cfg),'code_fingerprint':'code','data_fingerprint':'data'}
    progress = {'stage':'complete','methods':3}
    assert completed_result_compatible(frame,metadata,progress,cfg,'code',True,'data')
    for invalid in [frame.iloc[:-1], frame.assign(experiment_version='gpu-eval-v3-batched-uvarfs'),
                    frame.drop(columns='aupro'), frame.assign(pixel_auprc=np.nan),
                    frame.assign(image_auroc=float('inf'))]:
        assert not completed_result_compatible(invalid,metadata,progress,cfg,'code',True,'data')
    assert not completed_result_compatible(frame,metadata,progress,cfg,'different-code',True,'data')
    assert not completed_result_compatible(frame,metadata,progress,cfg,'code',True,'changed-data')
    assert not completed_result_compatible(frame,[],progress,cfg,'code',True,'data')
    modified = copy.deepcopy(cfg)
    modified['seed'] = 123
    assert not completed_result_compatible(frame,metadata,progress,modified,'code',True,'data')


def test_data_fingerprint_detects_replaced_mask(tmp_path):
    image, mask = tmp_path/'image.png',tmp_path/'mask.png'
    save_image(image,np.zeros((8,8)))
    save_image(mask,np.zeros((8,8)))
    sample = Sample(image,1,mask,'test','liver')
    before = data_fingerprint([], [sample])
    save_image(mask,np.ones((9,9))*255)
    assert data_fingerprint([], [sample]) != before


def test_runner_preflight_checks_masks_and_all_six_without_loading_model(tmp_path,monkeypatch,capsys):
    import yaml
    repo = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('run_all_preflight',repo/'scripts'/'run_all.py')
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    def forbidden(*args):
        pytest.fail('data preflight must not load DINOv3')
    monkeypatch.setattr(runner,'DINOv3Extractor',forbidden)
    root = tmp_path/'BMAD'/'liver'
    save_image(root/'train'/'good'/'img'/'train.png',np.zeros((8,8)))
    save_image(root/'test'/'good'/'img'/'normal.png',np.zeros((8,8)))
    save_image(root/'test'/'Ungood'/'img'/'abnormal.jpg',np.zeros((8,8)))
    save_image(root/'test'/'Ungood'/'anomaly_mask'/'abnormal_mask.png',np.ones((8,8))*255)
    cfg = load_config(repo/'configs'/'default.yaml')
    cfg['paths']['processed_data'] = str(tmp_path/'BMAD')
    cfg['results_dir'] = str(tmp_path/'output')
    config_path = tmp_path/'config.yaml'
    config_path.write_text(yaml.safe_dump(cfg),encoding='utf-8')
    monkeypatch.setattr(sys,'argv',['run_all.py','--config',str(config_path),'--dataset','liver','--check-data-only'])
    runner.main()
    assert 'matched_abnormal_masks": 1' in capsys.readouterr().out
    monkeypatch.setattr(sys,'argv',['run_all.py','--config',str(config_path),'--check-data-only'])
    with pytest.raises(SystemExit,match='Missing requested BMAD datasets'):
        runner.main()
    for name in ['brain','resc','xray','oct2017','camelyon16']:
        dataset = tmp_path/'BMAD'/name
        save_image(dataset/'train'/'good'/'img'/'train.png',np.zeros((8,8)))
        save_image(dataset/'test'/'good'/'img'/'normal.png',np.zeros((8,8)))
        save_image(dataset/'test'/'Ungood'/'img'/'abnormal.jpg',np.zeros((8,8)))
        if name in ['brain','resc']:
            save_image(dataset/'test'/'Ungood'/'anomaly_mask'/'abnormal_mask.png',np.ones((8,8))*255)
    extra = tmp_path/'BMAD'/'experimental_other'
    (extra/'train').mkdir(parents=True)
    (extra/'test').mkdir(parents=True)
    runner.main()
    assert len(capsys.readouterr().out.strip().splitlines()) == 6


class TinyFrozenExtractor:
    """Test fixture only: exercises all layers and real image/mask I/O on CPU."""
    num_layers, hidden_dim, patch_size = 12, 4, 4

    @torch.inference_mode()
    def __call__(self, image):
        patches = F.avg_pool2d(image.permute(0,3,1,2),4).flatten(2).transpose(1,2)
        return {l:torch.cat([patches,torch.ones_like(patches[...,:1])*(1+l/12)],-1)
                for l in range(1,13)}

    def pooled(self, features):
        return {l:x.mean(1) for l,x in features.items()}


class CPUExactIndex:
    def __init__(self, memory):
        self.memory = F.normalize(torch.from_numpy(memory),dim=1)

    def score_tensor(self, query):
        return 1-(F.normalize(query,dim=1) @ self.memory.T).amax(1).clamp(-1,1)


@pytest.mark.parametrize('geometry_representation',['pooled','patch'])
def test_cpu_fixture_full_fit_memory_evaluation_and_randomk_fairness(tmp_path,monkeypatch,geometry_representation):
    monkeypatch.setattr('uvarfs.pipeline_eval.make_index',lambda memory,cfg:CPUExactIndex(memory))
    cfg = load_config(Path(__file__).resolve().parents[1]/'configs'/'default.yaml')
    cfg['model'].update(input_size=8,batch_size=2,num_workers=0)
    cfg['data'].update(fit_images=4,variability_images=4,patches_per_image=4,
                       memory_images=4,memory_patches_per_image=4,memory_size=5,test_batch_size=2)
    cfg['asls'].update(steps=10,geometry_tolerance=.8)
    # The patch case uses the production default; the pooled case is an explicit ablation.
    if geometry_representation=='pooled':
        cfg['asls']['geometry_representation']='pooled'
        cfg['methods']=[name.replace('asls_pooled_','asls_patch_') for name in cfg['methods']]
    cfg['uvarfs'].update(min_features=2,max_features=4,max_iter=400,
                         lambda_grid=[.01,.0001,.000001],geometry_tolerance=.8)
    cfg['eval'].update(bootstrap_samples=0,max_heatmaps_per_dataset=2)
    root = tmp_path/'liver'
    gradient = np.arange(64).reshape(8,8)
    for i in range(4):
        save_image(root/'train'/'good'/'img'/f'{i}.png', gradient+i*25)
    save_image(root/'test'/'good'/'img'/'normal.png',gradient)
    abnormal = gradient.copy()
    abnormal[:4,:4] = 255
    mask = np.zeros((8,8),np.uint8)
    mask[:4,:4] = 255
    save_image(root/'test'/'Ungood'/'img'/'lesion.jpg',abnormal)
    save_image(root/'test'/'Ungood'/'anomaly_mask'/'lesion_mask.png',mask)
    output = tmp_path/'output'
    output.mkdir()
    train, test = scan_split(root,'train','liver'), scan_split(root,'test','liver')
    extractor = TinyFrozenExtractor()
    specs = fit_method_specs(extractor,train,cfg,output)
    assert sorted(specs) == expected_method_names(cfg)
    assert len(specs) == 36
    fitted=json.loads((output/'asls.json').read_text())
    assert fitted['geometry_representation']==geometry_representation
    assert fitted['diagnostics']['normal_geometry_samples']==(16 if geometry_representation=='patch' else 4)
    assert specs['main']['layers']==fitted['selected_layers']
    assert fitted['discrete_selection']=='geometry_search'
    prefix=json.loads((output/'asls_gate_prefix.json').read_text())
    assert specs['legacy_main']['layers']==prefix['selected_layers']
    assert specs['legacy_main']['obj']['sparsity_strategy']=='top_weights'
    assert specs['asls_top_weights_uvarfs']['layers']==specs['main']['layers']
    assert specs['asls_top_weights_uvarfs']['obj']['sparsity_strategy']=='top_weights'
    assert specs['main']['obj']['sparsity_strategy']=='objective_prune_refit'
    search=json.loads((output/'asls_geometry_search.json').read_text())
    for kind in ['raw','uvarfs']:
        assert specs[f'asls_geometry_search_{kind}']['layers']==search['selected_layers']
        assert specs[f'asls_geometry_search_{kind}']['asls_geometry']['discrete_selection']=='geometry_search'
    uv=specs['main']['obj']
    assert uv['objective_certificate_scope']=='continuous_full_weights'
    assert 'budgeted_box_optimality_gap' in uv
    assert uv['fixed_support_geometry_floor']<=uv['geometry_error']+1e-6
    for point,old in zip(uv['lambda_path'],uv['legacy_top_weights']['lambda_path']):
        assert point['budgeted_objective']<=old['budgeted_objective']+1e-7
        assert point['geometry_error']<=old['geometry_error']+1e-7
    assert specs['main']['asls_geometry']['geometry_representation']==geometry_representation
    assert json.loads((output/'uvarfs_main.json').read_text())['layers']==fitted['selected_layers']
    comparison_mode='pooled' if geometry_representation=='patch' else 'patch'
    comparison=json.loads((output/f'asls_{comparison_mode}.json').read_text())
    assert comparison['diagnostics']['normal_geometry_samples']==(4 if comparison_mode=='pooled' else 16)
    for kind in ['raw','uvarfs']:
        spec=specs[f'asls_{comparison_mode}_{kind}']
        assert spec['layers']==comparison['selected_layers']
        assert spec['asls_geometry']['geometry_representation']==comparison_mode
    k = len(specs['main']['layers'])
    for seed in cfg['random_baseline_seeds']:
        assert len(specs[f'randomk_raw_seed{seed}']['layers']) == k
        assert specs[f'randomk_raw_seed{seed}']['layers'] == specs[f'randomk_uvarfs_seed{seed}']['layers']
    manifest=json.loads((output/'fit_manifest.json').read_text())
    assert manifest['asls_geometry_representation']==geometry_representation
    assert set(manifest['normal_train_paths']) == {str(s.image) for s in train}
    # Duplicate representations must yield exactly the same memory rows,
    # including reservoir replacement when sampled rows exceed capacity.
    memories = build_memories(extractor,train,{**specs,'duplicate':specs['all_raw']},cfg,
                               progress_path=output/'memory_progress.json')
    np.testing.assert_array_equal(memories['duplicate'],memories['all_raw'])
    del memories['duplicate']
    rows = evaluate(extractor,test,specs,memories,cfg,progress_path=output/'eval_progress.json')
    frame = pd.DataFrame(rows)
    assert frame[['image_auroc','image_auprc','pixel_auroc','pixel_auprc','aupro']].notna().all().all()
    predictions = pd.read_csv(output/'image_predictions.csv')
    assert len(predictions) == 2
    assert set(specs) <= set(predictions.columns)
    assert len(list((output/'heatmaps').glob('*.png'))) == 2
    assert json.loads((output/'eval_progress.json').read_text())['mask_coverage']['matched_abnormal_masks'] == 1
    for image in (output/'heatmaps').glob('*.png'):
        with Image.open(image) as preview:
            assert preview.size == (8*4,8+24)
    frame['dataset'] = 'liver'
    write_summaries(frame,output)
    summary = pd.read_csv(output/'summary_metrics.csv')
    random_summary = summary[summary['method_family']=='randomk_uvarfs'].iloc[0]
    reference = frame[frame['method'].str.startswith('randomk_uvarfs_')]['image_auroc']
    assert random_summary['runs'] == 5
    assert random_summary['image_auroc_mean'] == pytest.approx(reference.mean())
    assert random_summary['image_auroc_std'] == pytest.approx(reference.std(ddof=1))
