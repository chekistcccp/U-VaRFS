from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
import time
from scipy.stats import rankdata


def cosine_gram(x: torch.Tensor) -> torch.Tensor:
    x = F.normalize(x.float(), dim=-1)
    return x @ x.T


def fit_asls(layer_features: dict[int, torch.Tensor], layer_variability: dict[int, float], cfg: dict) -> dict:
    layers = sorted(layer_features)
    steps=int(cfg.get('steps',300))
    if not layers or steps<1 or not 1 <= int(cfg.get('min_layers',2)) <= min(int(cfg.get('max_layers',6)),len(layers)):
        raise ValueError('invalid ASLS layer budget or optimizer steps')
    grams = torch.stack([cosine_gram(layer_features[l]) for l in layers], dim=0)
    consensus = grams.mean(dim=0)
    denom = consensus.norm().clamp_min(1e-8)
    v = torch.tensor([layer_variability.get(l, 0.0) for l in layers], device=grams.device, dtype=torch.float32)
    if v.numel() and v.max() > v.min():
        v = (v - v.min()) / (v.max() - v.min() + 1e-8)
    logits = torch.zeros(len(layers), device=grams.device, requires_grad=True)
    opt = torch.optim.Adam([logits], lr=float(cfg.get("lr", 0.05)))
    gamma = float(cfg.get("variability_weight", 0.15))
    rho = float(cfg.get("sparsity_weight", 0.02))
    history=[]
    t0=time.perf_counter()
    log_every=max(1,int(cfg.get('log_every',50)))
    for step in range(1,steps+1):
        p = torch.sigmoid(logits)
        weighted = (p[:, None, None] * grams).sum(0) / max(len(layers), 1)
        geom = (weighted - consensus).norm() / denom
        var = (p * v).sum() / p.sum().clamp_min(1e-6)
        sparse = p.mean()
        loss = geom + gamma * var + rho * sparse
        opt.zero_grad(set_to_none=True); loss.backward()
        if step==1 or step%log_every==0 or step==steps:
            history.append({'step':step,'loss':float(loss.detach()),'geometry':float(geom.detach()),
                            'variability':float(var.detach()),'sparsity':float(sparse.detach()),
                            'gradient_norm':float(logits.grad.norm().detach())})
            print(f'[asls] step={step}/{steps} loss={history[-1]["loss"]:.5f} '
                  f'geometry={history[-1]["geometry"]:.5f} probability_mean={history[-1]["sparsity"]:.5f}',flush=True)
        opt.step()
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    order = np.argsort(-probs)
    tol = float(cfg.get("geometry_tolerance", 0.05))
    min_layers = int(cfg.get("min_layers", 2)); max_layers = int(cfg.get("max_layers", 6))
    best = list(order[:min_layers])
    cons = consensus.detach()
    selection_path=[]
    for k in range(min_layers, min(max_layers, len(layers)) + 1):
        ids = order[:k]
        kg = grams[ids].mean(dim=0)
        err = float(((kg - cons).norm() / denom).cpu())
        selection_path.append({'layer_count':k,'layers':[layers[i] for i in ids],
                               'geometry_error':err,'feasible':err<=tol})
        best = list(ids)
        if err <= tol:
            break
    selected = [layers[i] for i in best]
    final_err = float(((grams[best].mean(dim=0) - consensus).norm() / denom).cpu())
    off_diagonal=~torch.eye(cons.shape[0],device=cons.device,dtype=torch.bool)
    raw_v=np.array([layer_variability.get(l,0.0) for l in layers])
    ranks_p=rankdata(probs)
    ranks_v=rankdata(raw_v)
    correlation=float(np.corrcoef(ranks_p,ranks_v)[0,1]) if np.ptp(raw_v)>0 and np.ptp(probs)>0 else None
    saturated=bool(np.min(probs)>0.98)
    if saturated:
        print('[asls] diagnostic: gates near all-open; preserve objective and inspect normal geometry/gradient trace',flush=True)
    return {
        "selected_layers": selected,
        "probabilities": {str(l): float(probs[i]) for i,l in enumerate(layers)},
        "variability": {str(l): float(layer_variability.get(l, 0.0)) for l in layers},
        "geometry_error": final_err,
        "feasible":final_err<=tol,
        "selection_path":selection_path,
        "optimizer_trace":history,
        "fit_seconds":time.perf_counter()-t0,
        "diagnostics":{'gates_near_all_open':saturated,'probability_span':float(np.ptp(probs)),
                       'gate_variability_rank_correlation':correlation,
                       'consensus_off_diagonal_mean':float(cons[off_diagonal].mean()) if off_diagonal.any() else None,
                       'consensus_off_diagonal_std':float(cons[off_diagonal].std(unbiased=False)) if off_diagonal.any() else None,
                       'layer_geometry_errors':{str(l):float((grams[i]-cons).norm()/denom) for i,l in enumerate(layers)},
                       'representation':'normal_mean_pooled_cosine_gram'},
    }
