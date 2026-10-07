"""Read-only normal patch geometry diagnostics; never used for selection."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .numerics import precise_matmul
from .u_varfs import apply_uvarfs


def describe_norms(features):
    norms=features.float().norm(dim=1)
    return {'min':float(norms.min()),'max':float(norms.max()),
            'mean':float(norms.mean()),'std':float(norms.std(unbiased=False))}


def compare_gram(reference, features, chunk=512):
    """Exact cosine Gram error, with bounded temporary row blocks."""
    normalized=F.normalize(features.float(),dim=1)
    n=normalized.shape[0]
    squared=torch.zeros((),device=features.device,dtype=torch.float64)
    denominator=torch.zeros_like(squared)
    agreement=torch.zeros((),device=features.device,dtype=torch.float64)
    regret=torch.zeros_like(agreement)
    for start in range(0,n,chunk):
        end=min(start+chunk,n)
        actual=normalized[start:end]@normalized.T
        expected=reference[start:end]
        denominator+=expected.double().square().sum()
        squared+=(actual-expected).double().square().sum()
        if n>1:
            rows=torch.arange(end-start,device=features.device)
            own=torch.arange(start,end,device=features.device)
            actual[rows,own]=-torch.inf
            expected=expected.clone(); expected[rows,own]=-torch.inf
            found=actual.argmax(1); best=expected.argmax(1)
            agreement+=(found==best).sum()
            regret+=(expected[rows,best]-expected[rows,found]).double().sum()
    return {'cosine_gram_relative_error':float(torch.sqrt(squared/denominator.clamp_min(1e-24))),
            'normal_patch_1nn_agreement':float(agreement/n) if n>1 else None,
            'normal_patch_1nn_reference_cosine_regret':float(regret/n) if n>1 else None,
            'zero_norm_rows':int((features.norm(dim=1)==0).sum()),
            'neighbor_scope':'sampled normal fit patches; self excluded; ties may change identity'}


@torch.inference_mode()
@precise_matmul()
def audit_normal_geometry(patches, specs):
    """Compare the actual concat/cosine geometry without changing its scaling.

    ASLS uses equal-weight unit-layer Grams; downstream uses raw layer concat.
    U-VaRFS preserves an unrenormalized weighted Gram; the detector renormalizes
    rows. Record these different quantities instead of treating their tolerances
    as a guarantee of normal nearest-neighbor preservation.
    """
    layers=sorted(patches)
    joined=torch.cat([patches[l] for l in layers],dim=1).float()
    reference=F.normalize(joined,dim=1)
    raw_gram=reference@reference.T
    del joined,reference
    consensus=torch.zeros_like(raw_gram)
    for layer in layers:
        x=F.normalize(patches[layer].float(),dim=1)
        consensus.add_(x@x.T,alpha=1/len(layers))
    numerator=torch.zeros((),device=raw_gram.device,dtype=torch.float64)
    denominator=torch.zeros_like(numerator)
    for start in range(0,len(raw_gram),512):
        reference=consensus[start:start+512]
        numerator+=(raw_gram[start:start+512]-reference).double().square().sum()
        denominator+=reference.double().square().sum()
    mismatch=float(torch.sqrt(numerator/denominator.clamp_min(1e-24)))
    result={'selection_candidate':False,'normal_geometry_rows':len(raw_gram),
            'layer_feature_normalization':'raw features unchanged',
            'unit_layer_consensus_vs_raw_full_cosine_error':mismatch,
            'layer_patch_norms':{str(l):describe_norms(patches[l]) for l in layers},
            'methods':{}}
    targets=['main','asls_raw','asls_prune_refit_uvarfs','asls_top_weights_uvarfs',
             'asls_gate_prefix_raw','asls_gate_prefix_uvarfs','legacy_main']
    for name in targets:
        if name not in specs:
            continue
        spec=specs[name]
        x=torch.cat([patches[l] for l in spec['layers']],dim=1).float()
        representation=apply_uvarfs(x,spec['obj']) if spec['kind']=='uvarfs' else x
        diagnostic={'layers':spec['layers'],'feature_dim':representation.shape[1],
                    'vs_raw_full_hierarchy':compare_gram(raw_gram,representation),
                    'vs_unit_layer_consensus':compare_gram(consensus,representation)}
        if spec['kind']=='uvarfs':
            selected=F.normalize(x,dim=1)
            diagnostic['vs_selected_raw_cosine']=compare_gram(selected@selected.T,representation)
            weighted=apply_uvarfs(selected,spec['obj'])
            diagnostic['weighted_normal_row_norms']=describe_norms(weighted)
            diagnostic['unrenormalized_weighted_gram_error']=spec['obj']['geometry_error']
        result['methods'][name]=diagnostic
    return result
