import json

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from uvarfs.geometry_audit import audit_normal_geometry, compare_gram
from uvarfs.pipeline_fit import perturbation_seed, fit_representation
from uvarfs.data import Sample
from uvarfs.perturb import perturb_batch
from uvarfs.sparse_support import objective_exchange
from uvarfs.u_varfs import _objective_certificate, fit_uvarfs
from scripts.analyze_results import audit_sparse_path
from uvarfs.asls import LayerGramGeometry, fit_asls


def test_uniform_gate_loss_slope_is_analytical_and_geometry_independent():
    gen=torch.Generator().manual_seed(5)
    features={i:torch.randn(8,3,generator=gen) for i in range(1,4)}
    geometry=LayerGramGeometry(features,list(features))
    rho=.02; gamma=.15
    variability=torch.tensor([.1,.5,.9],dtype=torch.float64)
    def objective(t):
        probabilities=torch.full((3,),t,dtype=torch.float64)
        return geometry.error((probabilities-1)/3)+gamma*(probabilities*variability).sum()/probabilities.sum()+rho*probabilities.mean()
    assert float((objective(.8)-objective(.2))/.6)==pytest.approx(rho-1,abs=1e-12)
    fitted=fit_asls(features,{i:0 for i in features},{'steps':2,'min_layers':1,'max_layers':3,'sparsity_weight':rho})
    assert fitted['diagnostics']['uniform_gate_loss_slope']==pytest.approx(rho-1)


def test_exchange_matches_exhaustive_exact_two_coordinate_changes():
    gen=torch.Generator().manual_seed(137)
    x=torch.randn(20,9,generator=gen,dtype=torch.float64)
    h=(x.T@x).square(); h/=h.sum()
    p=torch.rand(9,3,generator=gen,dtype=torch.float64)
    seed=torch.zeros(9,3,dtype=torch.float64)
    seed[[0,2,6],0]=torch.tensor([.5,.9,.6],dtype=torch.float64)
    seed[[1,7],1]=torch.tensor([.4,.8],dtype=torch.float64)
    lambdas=[.001,.0001,.01]; beta=.02; scale=.1
    a=h+2*beta*scale*(p@p.T)
    cfg={'support_exchange_max_steps':1,'support_exchange_chunk':2,'support_exchange_min_improvement':1e-12}
    actual,counts=objective_exchange(seed,h,p,beta,scale,lambdas,cfg,'test')
    expected=seed.clone()
    for c,lam in enumerate(lambdas):
        w=seed[:,c]; g=a@w-h.sum(1)+lam
        old=_objective_certificate(w[:,None],h,p,beta,scale,[lam])['objective'][0]
        old_rep=(1-w)@h@(1-w)
        choices=[]
        for i in torch.where(w>0)[0].tolist():
            for j in torch.where(w==0)[0].tolist():
                b=float(((a[j,i]*w[i]-g[j])/a[j,j]).clamp(0,1))
                if b<=1e-4:
                    continue
                candidate=w.clone(); candidate[i]=0; candidate[j]=b
                value=_objective_certificate(candidate[:,None],h,p,beta,scale,[lam])['objective'][0]
                representation=(1-candidate)@h@(1-candidate)
                if representation<=old_rep and value<old-1e-12:
                    choices.append((float(value),candidate))
        if choices:
            expected[:,c]=min(choices,key=lambda item:item[0])[1]
    torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
    assert counts==[int(not torch.equal(expected[:,i],seed[:,i])) for i in range(3)]
    assert torch.equal((actual>0).sum(0),(seed>0).sum(0))


def test_exchange_can_reintroduce_a_previously_omitted_dimension():
    h=torch.diag(torch.tensor([.9,.08,.02],dtype=torch.float64))
    seed=torch.tensor([[0.],[.9],[0.]],dtype=torch.float64)
    fitted,count=objective_exchange(seed,h,torch.zeros(3,0,dtype=torch.float64),0,0,[1e-5],
        {'support_exchange_max_steps':4,'support_exchange_chunk':1},'test')
    assert count==[1]
    assert torch.where(fitted[:,0]>0)[0].tolist()==[0]
    assert (1-fitted[:,0])@h@(1-fitted[:,0])<(1-seed[:,0])@h@(1-seed[:,0])


@pytest.mark.parametrize('beta',[0.,.002,.2])
def test_v9_path_protects_v8_and_retains_exact_v8_control(beta):
    gen=torch.Generator().manual_seed(23)
    x=torch.randn(48,12,generator=gen)
    v=torch.rand(3,12,generator=gen)
    cfg={'beta':beta,'lambda_grid':[.01,.001,.0001],'min_features':2,'max_features':4,
         'geometry_tolerance':.02,'max_iter':150,'support_refit_max_iter':200,
         'support_exchange_max_steps':3,'support_exchange_chunk':3,'tol':1e-6,'diagnose_infeasible':True}
    result=fit_uvarfs(x,v,cfg)
    old=fit_uvarfs(x,v,{**cfg,'sparsity_strategy':'objective_prune_refit'})
    assert result['sparsity_strategy']=='objective_exchange_refit'
    for key in ['lambda','lambda_path','geometry_error']:
        assert result['prune_refit'][key]==old[key]
    for key in ['active','scales','budgeted_weights']:
        np.testing.assert_array_equal(result['prune_refit'][key],old[key])
    for point,previous in zip(result['lambda_path'],old['lambda_path']):
        assert point['budgeted_objective']<=previous['budgeted_objective']+1e-7
        assert point['geometry_error']<=previous['geometry_error']+1e-7
        assert 2<=point['retained_features']<=4
        if point['budget_source'].startswith('objective_exchange'):
            assert point['fixed_support_box_gap'] is not None
    assert result['zero_lambda_diagnostic']['selection_candidate'] is False
    assert result['lambda'] in cfg['lambda_grid']
    assert audit_sparse_path(result,cfg)['selection_and_guards_verified']


def test_result_audit_rejects_a_tampered_refinement_or_label_inconsistent_selection():
    import copy
    gen=torch.Generator().manual_seed(9)
    x=torch.randn(16,6,generator=gen)
    v=torch.rand(2,6,generator=gen)
    cfg={'beta':.002,'lambda_grid':[.01,.001],'min_features':2,'max_features':3,
         'geometry_tolerance':.05,'max_iter':100,'support_refit_max_iter':100}
    result=fit_uvarfs(x,v,cfg)
    broken=copy.deepcopy(result)
    broken['lambda_path'][0]['budgeted_objective']=result['prune_refit']['lambda_path'][0]['budgeted_objective']+1
    with pytest.raises(ValueError,match='guard'):
        audit_sparse_path(broken,cfg)
    broken=copy.deepcopy(result)
    broken['lambda_path'][0]['feasible']=not broken['lambda_path'][0]['feasible']
    with pytest.raises(ValueError,match='feasibility'):
        audit_sparse_path(broken,cfg)


def test_noise_is_independent_of_global_rng_order_and_preserves_state():
    image=torch.zeros(2,8,8,3)
    seed=perturbation_seed(42,'liver',8)
    state=torch.random.get_rng_state().clone()
    first=perturb_batch(image,'noise',torch.Generator().manual_seed(seed))
    assert torch.equal(state,torch.random.get_rng_state())
    torch.rand(1000)
    second=perturb_batch(image,'noise',torch.Generator().manual_seed(seed))
    torch.testing.assert_close(first,second,rtol=0,atol=0)
    assert seed!=perturbation_seed(42,'brain',8)
    assert seed!=perturbation_seed(42,'liver',16)


def test_normal_variability_fit_is_reproducible_after_other_dataset_rng_consumption(tmp_path):
    from PIL import Image
    class Extractor:
        num_layers=3; hidden_dim=3

        def __call__(self,image):
            image=image.to('cuda' if torch.cuda.is_available() else 'cpu')
            x=F.avg_pool2d(image.permute(0,3,1,2),4).flatten(2).transpose(1,2)
            return {i:torch.tanh(x*i/3) for i in range(1,4)}

        def pooled(self,features):
            return {i:F.normalize(x.mean(1),dim=1) for i,x in features.items()}
    samples=[]
    for i in range(4):
        path=tmp_path/'train'/'good'/f'{i}.png'; path.parent.mkdir(parents=True,exist_ok=True)
        Image.fromarray((np.arange(64).reshape(8,8)+i*25).astype(np.uint8)).save(path)
        samples.append(Sample(path,0,None,'train','liver'))
    cfg={'seed':42,'model':{'input_size':8,'batch_size':2,'num_workers':0},
         'data':{'fit_images':4,'variability_images':4,'patches_per_image':4},
         'asls':{'steps':2,'min_layers':1,'max_layers':3,'geometry_representation':'patch'},'methods':[]}
    _,first,_=fit_representation(Extractor(),samples,cfg,tmp_path/'first')
    torch.rand(1000)
    if torch.cuda.is_available():
        torch.rand(1000,device='cuda')
    _,second,_=fit_representation(Extractor(),samples,cfg,tmp_path/'second')
    for layer in first:
        torch.testing.assert_close(first[layer],second[layer],rtol=0,atol=0)
    left=json.loads((tmp_path/'first'/'fit_manifest.json').read_text())
    right=json.loads((tmp_path/'second'/'fit_manifest.json').read_text())
    assert left['perturbation_rng']==right['perturbation_rng']


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA runtime unavailable locally')
def test_exchange_cuda_matches_independent_cpu_path():
    gen=torch.Generator().manual_seed(21)
    x=torch.randn(20,11,generator=gen,dtype=torch.float64)
    h=(x.T@x).square(); h/=h.sum()
    p=torch.rand(11,2,generator=gen,dtype=torch.float64)
    seed=torch.zeros(11,2,dtype=torch.float64)
    seed[:4]=torch.rand(4,2,generator=gen,dtype=torch.float64)
    cfg={'support_exchange_max_steps':4,'support_exchange_chunk':3}
    cpu,count=objective_exchange(seed,h,p,.01,.1,[.001,.0001],cfg,'cpu')
    cuda,cuda_count=objective_exchange(seed.cuda(),h.cuda(),p.cuda(),.01,.1,[.001,.0001],cfg,'cuda')
    torch.testing.assert_close(cuda.cpu(),cpu,rtol=1e-6,atol=1e-7)
    assert cuda_count==count


def test_normal_audit_exposes_layer_norm_and_cosine_renormalization_mismatch():
    # Identical unit-layer geometry, unequal raw norms: selection geometry is
    # invariant to scaling, but the actual concatenated cosine geometry is not.
    f1=torch.tensor([[1.,0.],[0.,1.],[1.,1.]])
    f2=torch.tensor([[1.,1.],[1.,0.],[0.,1.]])*20
    patches={1:f1,2:f2}
    obj={'active':np.array([0,1]),'scales':np.array([1.,1.]),'geometry_error':.25}
    specs={'asls_raw':{'layers':[1],'kind':'raw'},'main':{'layers':[1,2],'kind':'uvarfs','obj':obj}}
    audit=audit_normal_geometry(patches,specs)
    assert audit['selection_candidate'] is False
    assert audit['unit_layer_consensus_vs_raw_full_cosine_error']>.1
    assert audit['layer_patch_norms']['2']['mean']>audit['layer_patch_norms']['1']['mean']*10
    assert audit['methods']['main']['weighted_normal_row_norms']['mean']<.1
    expected=F.normalize(torch.cat([f1,f2],1),dim=1)
    actual=F.normalize(f1,dim=1)
    gram=expected@expected.T
    direct=float((actual@actual.T-gram).norm()/gram.norm())
    assert audit['methods']['main']['vs_raw_full_hierarchy']['cosine_gram_relative_error']==pytest.approx(direct,abs=1e-7)
    assert audit['methods']['main']['vs_selected_raw_cosine']['cosine_gram_relative_error']==pytest.approx(direct,abs=1e-7)
    json.dumps(audit,allow_nan=False)


def test_gram_audit_matches_direct_gram_and_excludes_self():
    gen=torch.Generator().manual_seed(4)
    a=F.normalize(torch.randn(7,5,generator=gen),dim=1)
    b=torch.randn(7,3,generator=gen)
    reference=a@a.T
    report=compare_gram(reference,b,chunk=2)
    actual=F.normalize(b,dim=1)@F.normalize(b,dim=1).T
    assert report['cosine_gram_relative_error']==pytest.approx(float((reference-actual).norm()/reference.norm()),abs=1e-7)
    reference.fill_diagonal_(-torch.inf); actual.fill_diagonal_(-torch.inf)
    assert report['normal_patch_1nn_agreement']==pytest.approx(float((reference.argmax(1)==actual.argmax(1)).float().mean()))
