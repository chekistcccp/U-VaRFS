import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F

from uvarfs.local_uvarfs import LocalObjective, fit_local_uvarfs, fit_local_asls, neighbor_pairs
from uvarfs.objectives import LOCAL, QUADRATIC, experiment_branches
from uvarfs.pipeline_fit import collect_normal_inputs, fit_all_method_specs
from uvarfs.pipeline_eval import build_memories, evaluate, transform
from uvarfs.protocol import expected_method_names, EXPERIMENT_VERSION
from uvarfs.utils import load_config
from test_v10_representation import make_fixture, CountingExtractor, CPUIndex


def case():
    gen=torch.Generator().manual_seed(721)
    x=torch.randn(24,13,generator=gen)
    val=torch.randn(8,13,generator=gen)
    groups=torch.arange(6).repeat_interleave(4)
    vg=torch.arange(6,8).repeat_interleave(4)
    pert=torch.stack([x+.1*torch.randn(x.shape,generator=gen),x*.9])
    return x,val,groups,vg,(x,pert)


def test_pair_gradient_matches_independent_autograd_and_full_coordinate_oracle():
    x,val,g,vg,paired=case()
    engine=LocalObjective(x,x,g,g,paired,beta=.13)
    active=torch.tensor([0,2,3,7,10]); p=torch.tensor([.1,.2,.3,.15,.25],requires_grad=True)
    # Independent dense weighted cosine calculation, no engine.cosine reuse.
    def cosine(a,b,weights):
        y=a*weights.sqrt();z=b*weights.sqrt()
        return F.cosine_similarity(y,z,dim=1,eps=1e-20)
    c=cosine(engine.a[:,active],engine.b[:,active],p)
    n=len(x);d=c-engine.target;w=engine.weights
    rank=(engine.target[:n]-engine.target[n:]-c[:n]+c[n:]).clamp_min(0)
    var=(1-cosine(engine.va[:,active],engine.vb[:,active],p)).mean()/engine.var_scale
    loss=(w.repeat(2)*d.square()).mean()/engine.scale+(w*rank.square()).mean()/engine.scale+.13*var
    expected=torch.autograd.grad(loss,p)[0]
    value,grad,_=engine.evaluate(active,p.detach(),True)
    assert value==pytest.approx(float(loss.detach()),rel=2e-5)
    torch.testing.assert_close(grad,expected,atol=3e-5,rtol=2e-5)
    _,full,_=engine.evaluate(active,p.detach(),True,True)
    torch.testing.assert_close(full[active],grad,atol=3e-5,rtol=2e-5)
    # One-sided finite derivative for a previously zero coordinate.
    eps=.001;weights=torch.zeros(13);weights[active]=p.detach();weights[5]=eps
    y=x*weights.sqrt()
    cv=F.cosine_similarity(torch.cat([y,y]),torch.cat([y[engine.near],y[engine.far]]))
    viol=(engine.target[:n]-engine.target[n:]-cv[:n]+cv[n:]).clamp_min(0)
    vc=F.cosine_similarity(engine.va*weights.sqrt(),engine.vb*weights.sqrt())
    lv=(w.repeat(2)*(cv-engine.target).square()).mean()/engine.scale+(w*viol.square()).mean()/engine.scale+.13*(1-vc).mean()/engine.var_scale
    assert float((lv-loss.detach())/eps)==pytest.approx(float(full[5]),rel=.03,abs=.03)


def test_cross_image_neighbor_exclusion_and_normal_holdout_reference():
    x,val,g,vg,_=case()
    near,far=neighbor_pairs(x,x,g,g)
    assert bool((g[near]!=g).all()) and bool((g[far]!=g).all())
    near,far=neighbor_pairs(val,x,vg,g)
    assert max(int(near.max()),int(far.max()))<len(x)
    with pytest.raises(ValueError,match='other images'):
        neighbor_pairs(x,x,torch.zeros(24),torch.zeros(24))


def test_fixed_budget_selection_and_beta_zero_remove_all_nuisance_use():
    x,val,g,vg,paired=case()
    cfg={'max_features':6,'beta':.002,'local_refit_max_iter':20,'local_exchange_max_steps':1,
         'local_exchange_candidates':2,'performance_weight_cap_factor':2.}
    r=fit_local_uvarfs(x,val,g,vg,paired,cfg)
    p=r['budgeted_weights'][r['active']]
    assert len(p)==6 and p.sum()==pytest.approx(1,abs=1e-6)
    assert p.min()>=1e-4/6-1e-7 and p.max()<=2/6+1e-6
    assert r['effective_weight_dimension']>=3-1e-5 and r['feasible'] is None
    assert r['selection_score']==min(c['selection_score'] for c in r['candidate_pool'])
    assert r['geometry_metric']=='normal_local_pair_relative_rmse'
    a=fit_local_uvarfs(x,val,g,vg,paired,{**cfg,'beta':0.})
    b=fit_local_uvarfs(x,val,g,vg,(x,paired[1]*3+torch.randn_like(paired[1])),{**cfg,'beta':0.})
    np.testing.assert_array_equal(a['active'],b['active'])
    np.testing.assert_array_equal(a['budgeted_weights'],b['budgeted_weights'])
    assert a['variability_term']==0.


def test_holdout_changes_candidate_comparison_but_not_fitted_weights():
    x,val,g,vg,paired=case()
    cfg={'max_features':5,'beta':.002,'local_refit_max_iter':12,'local_exchange_max_steps':0}
    a=fit_local_uvarfs(x,val,g,vg,paired,cfg)
    b=fit_local_uvarfs(x,-val,g,vg,paired,cfg)
    assert len(a['candidate_pool'])==len(b['candidate_pool'])
    for first,second in zip(a['candidate_pool'],b['candidate_pool']):
        assert first['active']==second['active'] and first['weights']==second['weights']
        assert first['training_objective']==second['training_objective']
    assert a['candidate_pool'][0]['validation_objective']!=b['candidate_pool'][0]['validation_objective']


def local_fixture(tmp_path):
    cfg,train,test=make_fixture(tmp_path,'l2')
    cfg['asls'].update(discrete_selection='normal_local',max_layers=3)
    cfg['data'].update(variability_images=2,normal_holdout_fraction=.25)
    cfg['uvarfs'].update(objective=LOCAL,ablation_objective=QUADRATIC,local_refit_max_iter=8,
                         local_exchange_max_steps=0,cosine_refit_max_iter=8,cosine_polish_max_iter=0,
                         performance_exchange_max_steps=0,cosine_exchange_max_steps=0)
    cfg['methods']=['main','asls_raw','asls_pca','asls_random','asls_selected_raw','all_raw',
                    'fixed4_raw','asls_no_variability_uvarfs','asls_no_rank_uvarfs','previous_main',
                    'asls_fixed_budget_uvarfs','asls_compression_uvarfs','randomk_raw','randomk_uvarfs']
    return cfg,train,test


def test_shared_disjoint_collection_and_precise_asls_subset_scores(tmp_path):
    cfg,train,_=local_fixture(tmp_path);extractor=CountingExtractor()
    data=collect_normal_inputs(extractor,train,cfg)
    assert len(extractor.outputs)==6  # 2 clean batches + 4 existing nuisance passes.
    split=data['l2']['manifest']['normal_selection_split']
    assert set(split['fit_patch_groups']).isdisjoint(split['holdout_patch_groups'])
    assert split['fit_patch_rows']==12 and split['holdout_patch_rows']==4
    assert data['l2']['paired_perturbations'][3].shape==(4,8,4)
    assert split==data['none']['manifest']['normal_selection_split']
    d=data['l2'];layers=[1,2,3,4]
    patch={l:d['patches'][l] for l in layers};held={l:d['heldout'][l] for l in layers}
    result=fit_local_asls(patch,held,d['groups'],d['heldout_groups'],None,{**cfg['asls'],'min_layers':2,'max_layers':3})
    assert len(result['selection_path'])==10
    teacher=LocalObjective(torch.cat(list(held.values()),1),torch.cat(list(patch.values()),1),d['heldout_groups'],d['groups'],beta=0.)
    for candidate in result['selection_path']:
        active=torch.tensor([j for l in candidate['layers'] for j in range(layers.index(l)*4,layers.index(l)*4+4)])
        value,_,_=teacher.evaluate(active,torch.full((len(active),),1/len(active)))
        assert candidate['selection_score']==pytest.approx(value,abs=2e-4)


def test_complete_fit_memory_image_pixel_and_control_identities(tmp_path,monkeypatch):
    cfg,train,test=local_fixture(tmp_path);extractor=CountingExtractor();out=tmp_path/'fit'
    cfg['methods']=load_config(Path(__file__).resolve().parents[1]/'configs/default.yaml')['methods']
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    monkeypatch.setattr('uvarfs.pipeline_eval.make_index',lambda memory,cfg:CPUIndex(memory))
    specs=fit_all_method_specs(extractor,train,cfg,out)
    assert sorted(specs)==expected_method_names(cfg)
    assert len(specs)==172
    assert len(extractor.outputs)==6
    for branch in experiment_branches(cfg):
        prefix=branch['prefix'];main=specs[prefix+'main'];k=len(main['obj']['active'])
        assert specs[prefix+'asls_pca']['obj']['components'].shape[0]==k
        assert specs[prefix+'asls_selected_raw']['obj']['active']==main['obj']['active'].tolist()
        for seed in cfg['random_baseline_seeds']:
            assert len(specs[prefix+f'asls_random_seed{seed}']['obj'])==k
        if branch['objective']==LOCAL:
            assert specs[prefix+'asls_no_variability_uvarfs']['obj']['beta']==0.
            assert specs[prefix+'asls_no_rank_uvarfs']['obj']['rank_weight']==0.
        else:
            assert prefix+'asls_no_variability_uvarfs' not in specs
    memories=build_memories(extractor,train,specs,cfg,progress_path=out/'memory_progress.json')
    assert set(map(len,memories.values()))=={5}
    frame=pd.DataFrame(evaluate(extractor,test,specs,memories,cfg,progress_path=out/'eval_progress.json'))
    assert frame[['image_auroc','image_auprc','pixel_auroc','pixel_auprc','aupro']].notna().all().all()
    norm=json.loads((out/'normal_geometry_audit.json').read_text())
    assert norm['unit_layer_consensus_vs_full_input_cosine_error']<1e-6


def test_default_protocol_declares_real_method_change_and_original_feature_cap():
    cfg=load_config(Path(__file__).resolve().parents[1]/'configs/default.yaml')
    assert cfg['uvarfs']['objective']==LOCAL and cfg['asls']['discrete_selection']=='normal_local'
    assert cfg['representation']['layer_normalization']=='l2' and cfg['uvarfs']['max_features']==256
    assert len(expected_method_names(cfg))==172
    assert EXPERIMENT_VERSION in cfg['results_dir']


def test_runner_resume_local_artifacts_and_readonly_audit(tmp_path,monkeypatch):
    import scripts.run_all as runner
    import scripts.analyze_local_results as analyzer
    from scripts.analyze_results import source_hashes
    cfg,train,test=local_fixture(tmp_path)
    cfg['paths']['processed_data']=str(tmp_path);cfg['results_dir']=str(tmp_path/'output')
    extractor=CountingExtractor()
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    monkeypatch.setattr(runner,'load_config',lambda _:copy.deepcopy(cfg))
    monkeypatch.setattr(runner,'DINOv3Extractor',lambda *args:extractor)
    monkeypatch.setattr(runner,'discover_bmad_roots',lambda _: {'liver':tmp_path/'liver'})
    monkeypatch.setattr('uvarfs.pipeline_eval.make_index',lambda memory,cfg:CPUIndex(memory))
    monkeypatch.setattr(sys,'argv',['run_all','--dataset','liver'])
    runner.main();folder=tmp_path/'output'/'liver'
    before=len(extractor.outputs);runner.main();assert len(extractor.outputs)==before
    frame=pd.read_csv(folder/'metrics.csv');main=frame.set_index('method').loc['main']
    assert main.uvarfs_objective_definition==LOCAL and pd.isna(main.uvarfs_geometry_feasible)
    monkeypatch.setattr(analyzer,'verify_recorded_code',lambda _:False)  # Uncommitted constructed fixture.
    hashes=source_hashes(tmp_path/'output')
    with pytest.raises(ValueError,match='all six'):
        analyzer.audit(tmp_path/'output',tmp_path/'strict',draws=100)
    result=analyzer.audit(tmp_path/'output',tmp_path/'report',True,100)
    assert result['completed_datasets']==['liver'] and not result['all_six_complete']
    assert hashes==source_hashes(tmp_path/'output')
    # Both causal artifacts and the previous ASLS are required for resume.
    (folder/'uvarfs_asls_no_variability.json').unlink()
    runner.main();assert len(extractor.outputs)>before
    manifest=json.loads((folder/'fit_manifest.json').read_text())
    manifest['normal_selection_split']['holdout_used_for_gradients']=True
    (folder/'fit_manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='leakage'):
        analyzer.audit(tmp_path/'output',tmp_path/'tampered',True,100)


def test_local_audit_rejects_wrong_pool_selection_and_fake_feasibility():
    from scripts.analyze_local_results import audit_local_solution
    x,val,g,vg,paired=case();cfg={'max_features':4,'local_refit_max_iter':8,'local_exchange_max_steps':0}
    r=fit_local_uvarfs(x,val,g,vg,paired,cfg)
    r=json.loads(json.dumps(r,default=lambda v:v.tolist()))
    assert audit_local_solution(r,cfg)
    altered=copy.deepcopy(r);altered['feasible']=True
    with pytest.raises(ValueError,match='feasibility'):
        audit_local_solution(altered,cfg)
    altered=copy.deepcopy(r);altered['candidate_id']=(r['candidate_id']+1)%len(r['candidate_pool'])
    with pytest.raises(ValueError,match='selection'):
        audit_local_solution(altered,cfg)


def test_invalid_holdout_and_zero_teacher_fail_explicitly(tmp_path):
    cfg,train,_=local_fixture(tmp_path)
    cfg['data']['normal_holdout_fraction']=0.
    with pytest.raises(ValueError,match='fraction'):
        collect_normal_inputs(CountingExtractor(),train,cfg)
    x,val,g,vg,paired=case();x[0]=0
    with pytest.raises(ValueError,match='zero dense'):
        LocalObjective(x,x,g,g,paired)


def test_independent_seed_summary_requires_distinct_main_fits(tmp_path,monkeypatch):
    import scripts.analyze_local_results as analyzer
    from uvarfs.protocol import BMAD_DATASETS
    cfg=load_config(Path(__file__).resolve().parents[1]/'configs/default.yaml')
    roots=[]
    for seed in [42,123]:
        root=tmp_path/f'run{seed}';roots.append(root)
        for name in BMAD_DATASETS:
            folder=root/name;folder.mkdir(parents=True)
            meta={'config':{**cfg,'seed':seed,'results_dir':str(root)},'code_fingerprint':'constructed-fixture'}
            (folder/'run_metadata.json').write_text(json.dumps(meta))
    def constructed_audit(root,output,completed_only,draws):
        output.mkdir(parents=True)
        seed=json.loads((root/'liver'/'run_metadata.json').read_text())['config']['seed']
        pd.DataFrame([{'dataset':name,'method':'main','image_auroc':.7+seed/10000,
                      'image_auprc':.5,'pixel_auroc':np.nan,'pixel_auprc':np.nan,'aupro':np.nan}
                      for name in BMAD_DATASETS]).to_csv(output/'verified_metrics.csv',index=False)
    monkeypatch.setattr(analyzer,'audit',constructed_audit)
    analyzer.audit_independent_seeds(roots,tmp_path/'summary',100)
    saved=json.loads((tmp_path/'summary'/'independent_seeds.json').read_text())
    assert saved['fit_seeds']==[42,123] and saved['datasets_per_seed']==6
    assert saved['main_macro_sd']['image_auroc']>0
    for name in BMAD_DATASETS:
        path=roots[1]/name/'run_metadata.json';meta=json.loads(path.read_text())
        meta['config']['seed']=42;path.write_text(json.dumps(meta))
    with pytest.raises(ValueError,match='duplicate'):
        analyzer.audit_independent_seeds(roots,tmp_path/'duplicate',100)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA runtime unavailable locally')
def test_local_pair_gradient_cpu_cuda_agree():
    x,val,g,vg,paired=case();active=torch.arange(7);p=torch.full((7,),1/7)
    cpu=LocalObjective(x,x,g,g,paired).evaluate(active,p,True,True)
    gpu=LocalObjective(x.cuda(),x.cuda(),g.cuda(),g.cuda(),tuple(t.cuda() for t in paired)).evaluate(active.cuda(),p.cuda(),True,True)
    assert cpu[0]==pytest.approx(gpu[0],rel=1e-5)
    torch.testing.assert_close(cpu[1],gpu[1].cpu(),atol=3e-5,rtol=2e-5)
