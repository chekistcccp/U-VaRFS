from __future__ import annotations
import time
import numpy as np
import torch
from sklearn.decomposition import PCA
from .representation import normalize_layer


def concat_layers(feats: dict[int, torch.Tensor], layers: list[int], layer_normalization='none') -> torch.Tensor:
    return torch.cat([normalize_layer(feats[l],layer_normalization) for l in layers], dim=-1)


def fit_pca(x: torch.Tensor, n_components: int, seed: int = 42):
    n_components=min(int(n_components),x.shape[0]-1,x.shape[1])
    if n_components < 1:
        raise ValueError("PCA needs at least one component")

    t0=time.time()
    print(
        f'[pca] start: samples={x.shape[0]} features={x.shape[1]} '
        f'components={n_components} device={x.device}',
        flush=True
    )

    if x.is_cuda:
        # Randomized CUDA PCA. q adds a small oversampling margin while keeping
        # this baseline much cheaper than a full SVD.
        torch.cuda.manual_seed_all(int(seed))
        xf=x.float()
        mean=xf.mean(dim=0)
        xc=xf-mean
        q=min(min(xc.shape),max(n_components,min(n_components+16,320)))
        _,_,V=torch.pca_lowrank(xc,q=q,center=False,niter=2)
        components=V[:,:n_components].T.contiguous()
        result={
            'mean':mean.detach().cpu().numpy().astype(np.float32),
            'components':components.detach().cpu().numpy().astype(np.float32),
            'backend':'torch_cuda_pca_lowrank',
        }
    else:
        arr=x.detach().float().cpu().numpy()
        p=PCA(n_components=n_components,svd_solver='randomized',random_state=seed)
        p.fit(arr)
        result={
            'mean':p.mean_.astype(np.float32,copy=False),
            'components':p.components_.astype(np.float32,copy=False),
            'backend':'sklearn_randomized',
        }

    print(f"[pca] done: backend={result['backend']} elapsed={time.time()-t0:.1f}s",flush=True)
    return result


def apply_pca(x: torch.Tensor, pca) -> torch.Tensor:
    """Apply PCA with GEMM on the current device."""
    xf=x.float()
    if isinstance(pca,dict):
        mean_arr=pca['mean']
        comp_arr=pca['components']
    else:
        mean_arr=pca.mean_
        comp_arr=pca.components_
    mean=torch.as_tensor(mean_arr,device=xf.device,dtype=xf.dtype)
    components=torch.as_tensor(comp_arr,device=xf.device,dtype=xf.dtype)
    return (xf-mean) @ components.T
