import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F
from PIL import Image

from uvarfs.asls import LayerGramGeometry
from uvarfs.data import Sample, scan_split
from uvarfs.pipeline_eval import transform, build_memories, evaluate
from uvarfs.pipeline_fit import collect_normal_inputs, fit_all_method_specs, fit_method_specs
from uvarfs.protocol import expected_method_names
from uvarfs.representation import normalization_modes, normalize_layer
from uvarfs.transforms import concat_layers
from uvarfs.utils import load_config
from scripts.analyze_results import audit_representation_protocol


def test_unit_layer_concat_exactly_realizes_asls_subset_cosine_geometry():
    gen=torch.Generator().manual_seed(52)
    features={i:torch.randn(24,5,generator=gen)*scale for i,scale in enumerate([1.,3.,20.,100.],1)}
    all_grams=torch.stack([F.normalize(x,dim=1)@F.normalize(x,dim=1).T for x in features.values()])
    selected=[1,4]
    actual=F.normalize(concat_layers(features,selected,'l2'),dim=1)
    expected=all_grams[[0,3]].mean(0)
    torch.testing.assert_close(actual@actual.T,expected,rtol=1e-6,atol=2e-7)
    geometry=LayerGramGeometry(features,list(features))
    direct=float((actual@actual.T-all_grams.mean(0)).norm()/all_grams.mean(0).norm())
    assert direct==pytest.approx(float(geometry.subset_error([0,3])),abs=2e-7)
    scaled={i:x*(i*30) for i,x in features.items()}
    torch.testing.assert_close(concat_layers(features,selected,'l2'),concat_layers(scaled,selected,'l2'),atol=1e-7,rtol=1e-6)
    raw=F.normalize(concat_layers(features,selected),dim=1)
    assert float((raw@raw.T-expected).norm())>.1


def test_matching_cache_separates_input_normalizations_and_preserves_raw_path():
    features={1:torch.tensor([[[1.,0.],[1.,2.]]]),2:torch.tensor([[[20.,20.],[0.,30.]]])}
    cache={}
    raw={'layers':[1,2],'kind':'raw','layer_normalization':'none'}
    normalized={**raw,'layer_normalization':'l2'}
    before=transform(features,raw,cache)
    after=transform(features,normalized,cache)
    expected=F.normalize(torch.cat([F.normalize(x,dim=-1) for x in features.values()],dim=-1),dim=-1)
    torch.testing.assert_close(after,expected,rtol=0,atol=0)
    torch.testing.assert_close(before,F.normalize(torch.cat(list(features.values()),-1),dim=-1),rtol=0,atol=0)
    torch.testing.assert_close(transform(features,raw,cache),before,rtol=0,atol=0)
    assert len(cache)==2 and not torch.allclose(before,after)


@pytest.mark.parametrize('settings',[{'layer_normalization':'bad'},
    {'layer_normalization':'l2','ablation_layer_normalization':'l2'},
    {'layer_normalization':'none','ablation_layer_normalization':'bad'}])
def test_invalid_input_normalization_fails_early(settings):
    with pytest.raises(ValueError):
        normalization_modes({'representation':settings})


class CountingExtractor:
    num_layers=12; hidden_dim=4; patch_size=4; num_prefix=1

    def __init__(self):
        self.outputs=[]

    @torch.inference_mode()
    def __call__(self,image):
        image=image.cpu()
        x=F.avg_pool2d(image.permute(0,3,1,2),4).flatten(2).transpose(1,2)
        result={i:torch.cat([x,torch.ones_like(x[...,:1])*(1+i/12)],-1)*(2**i) for i in range(1,13)}
        self.outputs.append({i:v.clone() for i,v in result.items()})
        return result

    def pooled(self,features):
        return {i:F.normalize(x.mean(1),dim=1) for i,x in features.items()}


class CPUIndex:
    def __init__(self,memory):
        self.memory=F.normalize(torch.from_numpy(memory),dim=1)

    def score_tensor(self,query):
        return 1-(F.normalize(query.cpu(),dim=1)@self.memory.T).amax(1).clamp(-1,1)


def make_fixture(tmp_path,primary='none'):
    cfg=load_config(Path(__file__).resolve().parents[1]/'configs'/'default.yaml')
    cfg['methods']=[m for m in cfg['methods'] if m not in {'asls_exchange_refit_uvarfs','asls_selected_raw'}]
    cfg['uvarfs']['sparsity_strategy']='objective_exchange_refit'  # Historical paired 74-method regression.
    cfg['uvarfs'].update(objective='quadratic_gram',ablation_objective=None)
    cfg['representation']={'layer_normalization':primary,'ablation_layer_normalization':'l2' if primary=='none' else 'none'}
    cfg['model'].update(input_size=8,batch_size=2,num_workers=0)
    cfg['data'].update(fit_images=4,variability_images=4,patches_per_image=4,
        memory_images=4,memory_patches_per_image=4,memory_size=5,test_batch_size=2)
    cfg['asls'].update(steps=3,geometry_tolerance=.8)
    cfg['uvarfs'].update(min_features=2,max_features=4,max_iter=100,support_refit_max_iter=100,
        support_exchange_max_steps=2,lambda_grid=[.01,.0001],geometry_tolerance=.8)
    cfg['eval'].update(bootstrap_samples=0,save_heatmaps=False)
    root=tmp_path/'liver'
    for i in range(4):
        image=root/'train'/'good'/'img'/f'{i}.png'; image.parent.mkdir(parents=True,exist_ok=True)
        Image.fromarray((np.arange(64).reshape(8,8)+i*30).astype(np.uint8)).save(image)
    for category in ['good','Ungood']:
        image=root/'test'/category/'img'/'image.png'; image.parent.mkdir(parents=True,exist_ok=True)
        Image.fromarray(np.arange(64,dtype=np.uint8).reshape(8,8)).save(image)
    mask=root/'test'/'Ungood'/'anomaly_mask'/'image.png'; mask.parent.mkdir(parents=True)
    array=np.zeros((8,8),np.uint8); array[2:6,2:6]=255; Image.fromarray(array).save(mask)
    return cfg,scan_split(root,'train','liver'),scan_split(root,'test','liver')


def test_normalized_variability_uses_the_same_feature_space_as_fit(tmp_path):
    cfg,train,_=make_fixture(tmp_path)
    extractor=CountingExtractor()
    collected=collect_normal_inputs(extractor,train,cfg)
    assert len(extractor.outputs)==10  # Two batches, four perturbations; no second DINO pass.
    for mode in ['none','l2']:
        normal=[]; perturbed={i:[] for i in range(4)}
        for offset in [0,5]:
            normal.append(normalize_layer(extractor.outputs[offset][3],mode).reshape(-1,4).double())
            for i in range(4):
                perturbed[i].append(normalize_layer(extractor.outputs[offset+i+1][3],mode).reshape(-1,4).double())
        x=torch.cat(normal)
        expected=torch.stack([((x-torch.cat(perturbed[i])).square().mean(0))/(x.var(0,unbiased=False)+1e-6) for i in range(4)]).float()
        torch.testing.assert_close(collected[mode]['variabilities'][3].cpu(),expected,rtol=1e-6,atol=1e-7)
    torch.testing.assert_close(collected['l2']['patches'][3].norm(dim=1),torch.ones(16),rtol=1e-6,atol=1e-7)
    assert not torch.allclose(collected['none']['variabilities'][3],collected['l2']['variabilities'][3])


@pytest.mark.parametrize('primary',['none','l2'])
def test_paired_74_method_fit_memory_and_pixel_evaluation_share_protocol(tmp_path,monkeypatch,primary):
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    monkeypatch.setattr('uvarfs.pipeline_eval.make_index',lambda memory,cfg:CPUIndex(memory))
    cfg,train,test=make_fixture(tmp_path,primary)
    extractor=CountingExtractor(); output=tmp_path/'results'
    specs=fit_all_method_specs(extractor,train,cfg,output)
    assert sorted(specs)==expected_method_names(cfg) and len(specs)==74
    assert len(extractor.outputs)==10
    prefix='layer_l2_' if primary=='none' else 'raw_input_'
    for name in ['main','asls_raw','all_raw','fixed4_raw','asls_pca','asls_random_seed42']:
        assert specs[name]['layer_normalization']==primary
        assert specs[prefix+name]['layer_normalization']!=primary
    first=json.loads((output/'fit_manifest.json').read_text())
    second=json.loads((output/prefix.rstrip('_')/'fit_manifest.json').read_text())
    for key in ['normal_train_indices','normal_train_paths','sampled_patches','perturbation_rng','backbone_raw_patch_norms']:
        assert first[key]==second[key]
    assert first['variability_feature_space']==primary
    assert second['variability_feature_space']!=primary
    manifest=json.loads((output/'representation_manifest.json').read_text())
    assert manifest['shared_fit_images_and_patches'] and manifest['shared_backbone_forwards']
    # A raw primary must reproduce the independent original stream.
    original=copy.deepcopy(cfg); original['representation']['ablation_layer_normalization']=None
    single=fit_method_specs(CountingExtractor(),train,original,tmp_path/'single')
    for key in ['active','scales','budgeted_weights']:
        np.testing.assert_array_equal(specs['main']['obj'][key],single['main']['obj'][key])
    memories=build_memories(extractor,train,specs,cfg,progress_path=output/'memory_progress.json')
    assert set(map(len,memories.values()))=={5}
    rows=evaluate(extractor,test,specs,memories,cfg,progress_path=output/'eval_progress.json')
    frame=pd.DataFrame(rows)
    frame['layer_normalization']=[specs[name]['layer_normalization'] for name in frame.method]
    assert frame[['image_auroc','image_auprc','pixel_auroc','pixel_auprc','aupro']].notna().all().all()
    assert len(frame)==74
    assert json.loads((output/'eval_progress.json').read_text())['methods']==74
    assert len(audit_representation_protocol(output,cfg,frame,first))==2
    for branch in [output,output/prefix.rstrip('_')]:
        normal=json.loads((branch/'normal_geometry_audit.json').read_text())
        assert normal['methods']['main']['vs_full_input_hierarchy']==normal['methods']['main']['vs_raw_full_hierarchy']
        if normal['layer_feature_normalization']=='l2':
            assert normal['unit_layer_consensus_vs_full_input_cosine_error']<1e-6
    second['variability_feature_space']=primary
    (output/prefix.rstrip('_')/'fit_manifest.json').write_text(json.dumps(second))
    with pytest.raises(ValueError,match='variability'):
        audit_representation_protocol(output,cfg,frame,first)
