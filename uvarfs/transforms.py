from __future__ import annotations
import numpy as np
import torch
from sklearn.decomposition import PCA


def concat_layers(feats: dict[int, torch.Tensor], layers: list[int]) -> torch.Tensor:
    return torch.cat([feats[l] for l in layers], dim=-1)


def fit_pca(x: torch.Tensor, n_components: int, seed: int = 42):
    arr=x.detach().float().cpu().numpy()
    n_components=min(n_components, arr.shape[0]-1, arr.shape[1])
    p=PCA(n_components=n_components, svd_solver="randomized", random_state=seed)
    p.fit(arr)
    return p


def apply_pca(x: torch.Tensor, pca) -> torch.Tensor:
    """Apply fitted sklearn PCA with torch GEMM on the current device.

    sklearn is retained only for fitting. Inference no longer moves every patch
    tensor to CPU, which was a major evaluation bottleneck.
    """
    xf=x.float()
    mean=torch.as_tensor(pca.mean_,device=xf.device,dtype=xf.dtype)
    components=torch.as_tensor(pca.components_,device=xf.device,dtype=xf.dtype)
    return (xf-mean) @ components.T
