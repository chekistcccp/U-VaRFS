"""Normal-only, rank-revealing coordinate supports, not a PCA representation.

Column projections are used only to choose real latent coordinates. The detector
still receives the original features on those coordinates with U-VaRFS weights.
No holdout argument or layer quota is used.
"""
from __future__ import annotations
import math
import torch
import torch.nn.functional as F
from .local_uvarfs import neighbor_pairs


@torch.no_grad()
def pivot_coordinates(matrix, k):
    if (matrix.ndim!=2 or not 1<=k<=matrix.shape[1] or not torch.isfinite(matrix).all()):
        raise ValueError('invalid normal diversity matrix or coordinate budget')
    residual=matrix.double().clone()
    initial=residual.square().sum(0)
    used=torch.zeros(matrix.shape[1],device=matrix.device,dtype=torch.bool)
    selected=[];pivot_energy=[];rank_steps=0
    cutoff=max(1.,float(initial.max()))*1e-24
    for _ in range(k):
        scores=residual.square().sum(0).masked_fill(used,-torch.inf)
        index=int(scores.argmax());energy=float(scores[index])
        if energy<=cutoff:
            remaining=torch.arange(matrix.shape[1],device=matrix.device)[~used]
            order=initial[~used].argsort(descending=True,stable=True)
            selected.extend(remaining[order[:k-len(selected)]].tolist())
            break
        selected.append(index);used[index]=True;pivot_energy.append(energy);rank_steps+=1
        q=residual[:,index]/math.sqrt(energy)
        residual-=q[:,None]*(q@residual)[None,:]
    support=torch.tensor(selected,device=matrix.device,dtype=torch.long).sort().values
    if len(support)!=k or len(support.unique())!=k:
        raise ValueError('normal diversity failed to fill exact coordinate budget')
    return support,{'rank_steps':rank_steps,'pivot_residual_energy':pivot_energy,
        'rank_deficient_fill':'descending_original_column_energy',
        'coordinate_selection_only':True}


@torch.no_grad()
def diverse_starts(x, groups, paired, beta, k, rank):
    near,far=neighbor_pairs(x,x,groups,groups,rank=rank)
    distance=(1-F.cosine_similarity(x,x[near])).clamp_min(0)
    weight=(1+distance/distance.mean().clamp_min(1e-4)).clamp_max(4)
    weight/=weight.mean()
    xx=x.double();centered=xx-xx.mean(0)
    contrast=torch.cat([xx-xx[near],xx-xx[far]],dim=0)*weight.sqrt().repeat(2)[:,None]/math.sqrt(2)
    stability=torch.zeros(x.shape[1],device=x.device,dtype=torch.float64)
    # Entire nuisance access is removed for the feature beta-zero control.
    if beta and paired is not None:
        clean,perturbed=paired
        stability=(perturbed.double()-clean.double()[None]).square().mean((0,1))/(xx.square().mean(0)+1e-6)
    variants=[('normal_covariance_rrqr',centered),('normal_neighbor_rrqr',contrast),
              ('stable_normal_neighbor_rrqr',contrast/(1+stability).sqrt())]
    starts=[];diagnostics={}
    for name,matrix in variants:
        active,info=pivot_coordinates(matrix,k)
        starts.append((name,active));diagnostics[name]=info
    return starts,diagnostics
