from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F


def cosine_gram(x: torch.Tensor) -> torch.Tensor:
    x = F.normalize(x.float(), dim=-1)
    return x @ x.T


def fit_asls(layer_features: dict[int, torch.Tensor], layer_variability: dict[int, float], cfg: dict) -> dict:
    layers = sorted(layer_features)
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
    for _ in range(int(cfg.get("steps", 300))):
        p = torch.sigmoid(logits)
        weighted = (p[:, None, None] * grams).sum(0) / max(len(layers), 1)
        geom = (weighted - consensus).norm() / denom
        var = (p * v).sum() / p.sum().clamp_min(1e-6)
        sparse = p.mean()
        loss = geom + gamma * var + rho * sparse
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    order = np.argsort(-probs)
    tol = float(cfg.get("geometry_tolerance", 0.05))
    min_layers = int(cfg.get("min_layers", 2)); max_layers = int(cfg.get("max_layers", 6))
    best = list(order[:min_layers])
    cons = consensus.detach()
    for k in range(min_layers, min(max_layers, len(layers)) + 1):
        ids = order[:k]
        kg = grams[ids].mean(dim=0)
        err = float(((kg - cons).norm() / denom).cpu())
        best = list(ids)
        if err <= tol:
            break
    selected = [layers[i] for i in best]
    final_err = float(((grams[best].mean(dim=0) - consensus).norm() / denom).cpu())
    return {
        "selected_layers": selected,
        "probabilities": {str(l): float(probs[i]) for i,l in enumerate(layers)},
        "variability": {str(l): float(layer_variability.get(l, 0.0)) for l in layers},
        "geometry_error": final_err,
    }
