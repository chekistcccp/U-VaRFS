import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F

from uvarfs.objectives import COSINE, experiment_branches
from uvarfs.pipeline_eval import transform, build_memories, evaluate
from uvarfs.pipeline_fit import fit_all_method_specs
from uvarfs.protocol import expected_method_names, EXPERIMENT_VERSION
from uvarfs.selection_diagnostics import cosine_feasibility_diagnostics
from scripts.analyze_results import audit_representation_protocol, write_cosine_report, DATASETS
from test_v10_representation import CountingExtractor, CPUIndex
from test_v12_cosine_uvarfs import fixture


@pytest.mark.parametrize('mode',['none','l2'])
def test_selected_support_is_exact_uniform_weight_ablation(mode):
    features={1:torch.tensor([[[1.,2.,3.],[4.,1.,2.]]]),
              2:torch.tensor([[[10.,2.,1.],[1.,20.,3.]]])}
    spec={'layers':[2,1],'kind':'selected_raw','layer_normalization':mode,
          'obj':{'active':[0,3,5]}}
    x=torch.cat([F.normalize(features[l],dim=-1) if mode=='l2' else features[l] for l in [2,1]],-1)
    expected=F.normalize(x.index_select(-1,torch.tensor([0,3,5])),dim=-1)
    torch.testing.assert_close(transform(features,spec,{}),expected,rtol=0,atol=0)
    # On a fixed support, equal positive simplex weights cancel in cosine.
    uniform={**spec,'kind':'uvarfs','obj':{'active':[0,3,5],'scales':np.full(3,1/np.sqrt(3),dtype=np.float32)}}
    torch.testing.assert_close(transform(features,uniform),expected,atol=1e-7,rtol=1e-6)


@pytest.mark.parametrize('primary',['none','l2'])
def test_156_methods_preserve_all_old_fits_memories_and_predictions(tmp_path,monkeypatch,primary):
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    monkeypatch.setattr('uvarfs.pipeline_eval.make_index',lambda memory,cfg:CPUIndex(memory))
    cfg,train,test=fixture(tmp_path,primary)
    old_extractor=CountingExtractor(); before=tmp_path/'before'
    old=fit_all_method_specs(old_extractor,train,cfg,before)
    new_cfg=copy.deepcopy(cfg); new_cfg['methods'].append('asls_selected_raw')
    extractor=CountingExtractor(); after=tmp_path/'after'
    new=fit_all_method_specs(extractor,train,new_cfg,after)
    assert len(old)==152 and len(new)==156 and sorted(new)==expected_method_names(new_cfg)
    assert len(extractor.outputs)==len(old_extractor.outputs)==10
    assert json.loads((before/'fit_manifest.json').read_text())==json.loads((after/'fit_manifest.json').read_text())
    for branch in experiment_branches(cfg):
        prefix=branch['prefix']; folder=after/branch['folder']
        old_asls=json.loads((before/branch['folder']/'asls.json').read_text())
        new_asls=json.loads((folder/'asls.json').read_text())
        old_asls.pop('fit_seconds'); new_asls.pop('fit_seconds')
        assert old_asls==new_asls
        main=new[prefix+'main']; control=new[prefix+'asls_selected_raw']
        artifact=json.loads((folder/'selected_support_control.json').read_text())
        assert artifact==control['obj']
        assert artifact['source_main']==prefix+'main' and artifact['extra_fit'] is False
        assert artifact['layers']==main['layers'] and artifact['active']==main['obj']['active'].tolist()
        assert artifact['source_objective']==branch['objective']
    for name,spec in old.items():
        assert new[name]['layers']==spec['layers']
        if spec['kind']=='uvarfs':
            for key in ['active','scales','budgeted_weights']:
                np.testing.assert_array_equal(new[name]['obj'][key],spec['obj'][key])
            assert new[name]['obj']['lambda']==spec['obj']['lambda']
        torch.testing.assert_close(transform(extractor.outputs[0],new[name]),
                                   transform(old_extractor.outputs[0],spec),rtol=0,atol=0)
    old_memory=build_memories(old_extractor,train,old,cfg,progress_path=before/'memory_progress.json')
    new_memory=build_memories(extractor,train,new,new_cfg,progress_path=after/'memory_progress.json')
    for name in old:
        np.testing.assert_array_equal(new_memory[name],old_memory[name])
    old_manifest=json.loads((before/'memory_manifest.json').read_text())
    new_manifest=json.loads((after/'memory_manifest.json').read_text())
    for key in ['normal_train_indices','normal_train_paths','patch_batches','memory_capacity','reservoir_seed']:
        assert old_manifest[key]==new_manifest[key]
    old_rows=pd.DataFrame(evaluate(old_extractor,test,old,old_memory,cfg,progress_path=before/'eval_progress.json')).set_index('method')
    rows=pd.DataFrame(evaluate(extractor,test,new,new_memory,new_cfg,progress_path=after/'eval_progress.json'))
    for metric in ['image_auroc','image_auprc','pixel_auroc','pixel_auprc','aupro']:
        np.testing.assert_array_equal(rows.set_index('method').loc[old_rows.index,metric],old_rows[metric])
    old_predictions=pd.read_csv(before/'image_predictions.csv')
    predictions=pd.read_csv(after/'image_predictions.csv')
    pd.testing.assert_frame_equal(predictions[old_predictions.columns],old_predictions,check_exact=True)
    rows['layer_normalization']=[new[n]['layer_normalization'] for n in rows.method]
    rows['feature_dim']=[new_memory[n].shape[1] for n in rows.method]
    fit=json.loads((after/'fit_manifest.json').read_text())
    records=audit_representation_protocol(after,new_cfg,rows,fit)
    for record in records:
        diagnostic=record['normal_cosine_geometry']['methods']['asls_selected_raw']
        assert diagnostic['support_source_method']==record['method']
        assert diagnostic['vs_weighted_main_cosine']['zero_norm_rows']==0
    # A wrong support or another branch's Main must not pass the result audit.
    control_path=after/'selected_support_control.json'
    original=json.loads(control_path.read_text())
    for key,bad in [('active',[999]*original['feature_dim']),('source_main','gram_main'),
                    ('extra_fit',True),('layers',list(reversed(original['layers'])))]:
        changed={**original,key:bad}; control_path.write_text(json.dumps(changed))
        with pytest.raises(ValueError,match='selected-support control'):
            audit_representation_protocol(after,new_cfg,rows,fit)
    control_path.write_text(json.dumps(original))


@pytest.mark.parametrize('pool_error,path_error,reason,generated,path_count',[
    (.02,.06,'feasible_generated_candidates_outside_lambda_path',1,0),
    (.06,.06,'no_feasible_generated_candidate',0,0),
    (.02,.02,'feasible_lambda_path_point',1,1)])
def test_feasibility_scope_does_not_change_path_or_claim_global_infeasibility(pool_error,path_error,reason,generated,path_count):
    def point(error,k=2):
        return {'retained_features':k,'nonzero_features':k,'geometry_error':error,'zero_norm_rows':0}
    result={'objective_definition':COSINE,'weights':[0]*6,
            'candidate_pool':[point(pool_error),point(.08)],'lambda_path':[point(path_error)]}
    before=copy.deepcopy(result)
    diagnostics=cosine_feasibility_diagnostics(result,{'min_features':2,'max_features':4,'geometry_tolerance':.05})
    assert result==before and diagnostics['selection_candidate'] is False
    assert diagnostics['generated_feasible_candidates']==generated
    assert diagnostics['lambda_path_feasible_points']==path_count
    assert diagnostics['feasibility_diagnosis']==reason
    assert diagnostics['global_budget_infeasibility_claimed'] is False
    assert diagnostics['feasibility_scope']=='prescribed_lambda_path'
    assert diagnostics['generated_sparsest_feasible_dimension']==(2 if generated else None)


def test_runner_records_control_identity_and_requires_each_control_artifact(tmp_path,monkeypatch):
    import scripts.run_all as runner
    cfg,_,_=fixture(tmp_path)
    cfg['paths']['processed_data']=str(tmp_path); cfg['results_dir']=str(tmp_path/'output')
    cfg['methods']=['main','asls_raw','asls_selected_raw','asls_pca','asls_random','all_raw']
    extractor=CountingExtractor()
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    monkeypatch.setattr(runner,'load_config',lambda _:cfg)
    monkeypatch.setattr(runner,'DINOv3Extractor',lambda *args:extractor)
    monkeypatch.setattr(runner,'discover_bmad_roots',lambda _: {'liver':tmp_path/'liver'})
    monkeypatch.setattr('uvarfs.pipeline_eval.make_index',lambda memory,cfg:CPUIndex(memory))
    monkeypatch.setattr(sys,'argv',['run_all','--dataset','liver'])
    runner.main()
    frame=pd.read_csv(tmp_path/'output'/'all_metrics.csv'); indexed=frame.set_index('method')
    assert frame.experiment_version.eq(EXPERIMENT_VERSION).all()
    for branch in experiment_branches(cfg):
        prefix=branch['prefix']; own=indexed.loc[prefix+'main']; control=indexed.loc[prefix+'asls_selected_raw']
        assert control.support_source_method==prefix+'main'
        assert control.representation_weighting=='uniform_on_main_support'
        assert control.feature_dim==own.feature_dim
        assert pd.isna(control.uvarfs_objective_definition)
        if branch['objective']==COSINE:
            assert own.uvarfs_feasibility_scope=='prescribed_lambda_path'
            assert own.uvarfs_global_budget_infeasibility_claimed is False or own.uvarfs_global_budget_infeasibility_claimed==False
    folder=tmp_path/'output'/'liver'
    fit=json.loads((folder/'fit_manifest.json').read_text())
    records=audit_representation_protocol(folder,cfg,frame,fit)
    # Constructed fixture copies exercise the report, not actual BMAD results.
    all_rows=pd.concat([frame.assign(dataset=name) for name in DATASETS],ignore_index=True)
    all_rows['method_family']=all_rows.method.str.replace(r'_seed\d+$','',regex=True)
    families=[]
    for (dataset,family),group in all_rows.groupby(['dataset','method_family']):
        families.append({'dataset':dataset,'method_family':family,'runs':len(group),
                         'image_auroc_mean':group.image_auroc.mean(),
                         'image_auroc_std':group.image_auroc.std(ddof=1) if len(group)>1 else 0})
    report=tmp_path/'report'; report.mkdir()
    paired=pd.DataFrame({'dataset':DATASETS,'target_method':['main']*6,
                        'comparison':['asls_selected_raw']*6,'delta_auroc':[0.]*6,
                        'ci95_lower':[0.]*6,'ci95_upper':[0.]*6})
    data={'experiment_version':EXPERIMENT_VERSION,'git_revision':'constructed-fixture',
          'bootstrap_draws':100,'git_blob_code_verified':False,
          'diagnostics':[{'dataset':name,'representation_audit':records} for name in DATASETS]}
    write_cosine_report(report,all_rows,pd.DataFrame(families),paired,pd.DataFrame(),data)
    controls=pd.read_csv(report/'selected_support_geometry.csv')
    assert len(controls)==24 and set(controls.source_main)=={b['prefix']+'main' for b in experiment_branches(cfg)}
    scope=pd.read_csv(report/'cosine_feasibility_scope.csv')
    assert len(scope)==12 and not scope.global_budget_infeasibility_claimed.any()
    # Check CSV audit independently of JSON audit.
    bad=frame.copy(); bad.loc[bad.method=='asls_selected_raw','support_source_method']='gram_main'
    with pytest.raises(ValueError,match='CSV source'):
        audit_representation_protocol(folder,cfg,bad,fit)
    bad=frame.copy(); bad.loc[bad.method=='main','uvarfs_generated_feasible_candidates']=999
    with pytest.raises(ValueError,match='feasibility scope'):
        audit_representation_protocol(folder,cfg,bad,fit)
    before=len(extractor.outputs); runner.main(); assert len(extractor.outputs)==before
    for branch in experiment_branches(cfg):
        (folder/branch['folder']/'selected_support_control.json').unlink()
        runner.main()
        assert len(extractor.outputs)>before
        before=len(extractor.outputs)
