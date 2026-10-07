import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from uvarfs.metrics import bootstrap_auc
from uvarfs.pipeline_eval import build_memories, evaluate
from uvarfs.pipeline_fit import fit_all_method_specs
from uvarfs.protocol import config_fingerprint, expected_method_names, validate_results_version
from scripts.analyze_results import audit_fit_only, source_hashes
from test_v10_representation import make_fixture, CountingExtractor, CPUIndex
from test_v12_cosine_uvarfs import fixture


def independent_ci(labels,scores,draws,seed):
    generator=np.random.default_rng(seed); values=[]
    for _ in range(draws):
        indices=generator.integers(0,len(labels),len(labels))
        if len(np.unique(labels[indices]))==2:
            values.append(roc_auc_score(labels[indices],scores[indices]))
    return np.percentile(values,[2.5,97.5]) if values else [np.nan,np.nan]


@pytest.mark.parametrize('batch_size',[1,7,32,1000])
@pytest.mark.parametrize('tied',[False,True])
def test_bootstrap_preserves_original_draws_and_independent_sklearn_ci(batch_size,tied):
    rng=np.random.default_rng(38)
    labels=np.r_[np.zeros(17),np.ones(6)]
    scores=rng.integers(0,4,len(labels)).astype(float) if tied else rng.normal(size=len(labels))
    actual=bootstrap_auc(labels,scores,117,42,batch_size)
    np.testing.assert_allclose(actual,independent_ci(labels,scores,117,42),atol=2e-15,rtol=0)


def test_bootstrap_skips_single_class_draws_and_handles_degenerate_cases():
    labels=np.array([0,1]); scores=np.array([.7,.3])
    np.testing.assert_array_equal(bootstrap_auc(labels,scores,200),[0,0])
    assert np.isnan(bootstrap_auc(labels,scores,0)).all()
    assert np.isnan(bootstrap_auc(np.zeros(3),np.ones(3),100)).all()
    assert bootstrap_auc(np.array([0,1,0,1]),np.zeros(4),100)==[.5,.5]
    with pytest.raises(ValueError,match='aligned'):
        bootstrap_auc(labels,np.array([np.nan,1]),10)
    with pytest.raises(ValueError,match='binary'):
        bootstrap_auc(np.array([0,1,2]),np.ones(3),10)
    with pytest.raises(ValueError,match='positive'):
        bootstrap_auc(labels,scores,10,batch_size=0)


@pytest.mark.parametrize('interrupt',[False,True])
def test_predictions_and_pixel_point_estimates_survive_statistics_interruption(tmp_path,monkeypatch,interrupt):
    import uvarfs.pipeline_eval as pipeline
    cfg,train,test=make_fixture(tmp_path)
    cfg['eval']['bootstrap_samples']=43
    extractor=CountingExtractor(); out=tmp_path/'output'; out.mkdir()
    specs={'main':{'kind':'raw','layers':[2]},'control':{'kind':'raw','layers':[5]}}
    monkeypatch.setattr(pipeline,'make_index',lambda memory,cfg:CPUIndex(memory))
    memories=build_memories(extractor,train,specs,cfg,progress_path=out/'memory_progress.json')
    observed=[]
    def checked_bootstrap(labels,scores,n,seed,**kwargs):
        predictions=pd.read_csv(out/'image_predictions.csv')
        points=pd.read_csv(out/'evaluation_metrics.partial.csv')
        progress=json.loads((out/'eval_progress.json').read_text())
        assert len(predictions)==len(test) and set(specs)<=set(predictions.columns)
        assert points[['image_auroc','image_auprc','pixel_auroc','pixel_auprc','aupro']].notna().all().all()
        assert progress['stage']=='metrics' and progress['percent']<100
        name=list(specs)[len(observed)]
        assert f'current={name}' in progress['extra']
        observed.append(name)
        if interrupt:
            raise RuntimeError('simulated statistics interruption')
        return bootstrap_auc(labels,scores,n,seed,**kwargs)
    monkeypatch.setattr(pipeline,'bootstrap_auc',checked_bootstrap)
    if interrupt:
        with pytest.raises(RuntimeError,match='statistics interruption'):
            evaluate(extractor,test,specs,memories,cfg,progress_path=out/'eval_progress.json')
        assert json.loads((out/'eval_progress.json').read_text())['stage']=='metrics'
        assert not (out/'metrics.csv').exists()
    else:
        rows=evaluate(extractor,test,specs,memories,cfg,progress_path=out/'eval_progress.json')
        predictions=pd.read_csv(out/'image_predictions.csv')
        for row in rows:
            name=row['method']
            np.testing.assert_allclose(row['image_auroc_ci95'],independent_ci(predictions.label.to_numpy(),predictions[name].to_numpy(),43,cfg['seed']))
        assert observed==list(specs)
        assert json.loads((out/'eval_progress.json').read_text())['stage']=='complete'
    assert not list(out.glob('*.tmp'))


def test_partial_historical_metadata_blocks_cross_version_overwrite_without_csv(tmp_path):
    out=tmp_path/'brain'; out.mkdir()
    (out/'run_metadata.json').write_text(json.dumps({'experiment_version':'gpu-eval-v12-cosine-simplex'}))
    with pytest.raises(ValueError,match='Historical results must remain intact'):
        validate_results_version(tmp_path)
    (out/'run_metadata.json').write_text('{broken')
    with pytest.raises(ValueError,match='Cannot establish'):
        validate_results_version(tmp_path)


def test_fit_only_analysis_checks_shared_protocol_without_inventing_anomaly_metrics(tmp_path):
    cfg,train,_=fixture(tmp_path)
    cfg['methods']=['main','asls_raw','asls_pca','asls_random','all_raw']
    source=tmp_path/'returned'; out=source/'liver'
    fit_all_method_specs(CountingExtractor(),train,cfg,out)
    metadata={'dataset':'liver','config':cfg,'config_fingerprint':config_fingerprint(cfg),
              'expected_methods':expected_method_names(cfg),'git_head':'constructed-fixture',
              'code_fingerprint':'constructed-fixture','experiment_version':'constructed-fit-only'}
    (out/'run_metadata.json').write_text(json.dumps(metadata))
    before=source_hashes(source)
    result=audit_fit_only(source,tmp_path/'report')
    assert source_hashes(source)==before
    assert result['analysis_scope']=='normal_fit_only' and result['anomaly_performance_assessed'] is False
    assert result['datasets'][0]['final_metrics_available'] is False
    assert result['datasets'][0]['predictions_available'] is False
    assert result['datasets'][0]['git_blob_code_verified'] is False
    saved=pd.read_csv(tmp_path/'report'/'normal_fit_summary.csv')
    assert len(saved)==4 and 'image_auroc' not in saved and 'aupro' not in saved
    with pytest.raises(ValueError,match='outside'):
        audit_fit_only(source,out/'report')
    bad=out/'gram'/'asls.json'; value=json.loads(bad.read_text()); value['geometry_error']+=.1
    bad.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='changed ASLS'):
        audit_fit_only(source,tmp_path/'bad-report')
    bad=out/'fit_manifest.json'; value=json.loads(bad.read_text()); value['fit_images']+=1
    bad.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='normal fit budget'):
        audit_fit_only(source,tmp_path/'bad-budget-report')
