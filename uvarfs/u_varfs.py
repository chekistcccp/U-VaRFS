from __future__ import annotations
import math
import numpy as np
import torch
import torch.nn.functional as F


def _power_lipschitz(a: torch.Tensor, iters: int = 50) -> float:
    x = torch.randn(a.shape[0], device=a.device, dtype=a.dtype)
    x = x / x.norm().clamp_min(1e-8)
    for _ in range(iters):
        x = a @ x
        x = x / x.norm().clamp_min(1e-8)
    return float((x @ (a @ x)).abs().item()) + 1e-6


def _fista(H, R, beta, lam, max_iter, tol, w0=None):
    m = H.shape[0]
    A = H + 2.0 * beta * R
    L = _power_lipschitz(A)
    w = torch.ones(m, device=H.device, dtype=H.dtype) if w0 is None else w0.clone()
    y = w.clone(); t = 1.0
    ones = torch.ones_like(w)
    for _ in range(max_iter):
        grad = H @ (y - ones) + 2.0 * beta * (R @ y)
        z = y - grad / L
        wn = torch.clamp(torch.sign(z) * torch.relu(z.abs() - lam / L), 0.0, 1.0)
        if torch.norm(wn - w) / torch.norm(w).clamp_min(1e-8) < tol:
            w = wn; break
        tn = 0.5 * (1 + math.sqrt(1 + 4*t*t))
        y = wn + ((t - 1) / tn) * (wn - w)
        w, t = wn, tn
    return w


def fit_uvarfs(X: torch.Tensor, variability_vectors: torch.Tensor, cfg: dict) -> dict:
    X = F.normalize(X.float(), dim=1)
    n = max(X.shape[0], 1)
    G = (X.T @ X) / n
    H = G * G
    ones0 = torch.ones(H.shape[0], device=H.device, dtype=H.dtype)
    hscale = (ones0 @ H @ ones0).clamp_min(1e-12)
    H = H / hscale
    P = variability_vectors.float().T  # M x V
    if P.numel() == 0:
        R = torch.zeros_like(H)
    else:
        P = P / (P.norm(dim=0, keepdim=True) + 1e-8)
        R = P @ P.T
        R = R * (H.norm() / R.norm().clamp_min(1e-12))
    beta = float(cfg.get("beta", 0.05))
    lambdas = [float(x) for x in cfg.get("lambda_grid", [0.1,0.05,0.01,0.005,0.001])]
    tol_geom = float(cfg.get("geometry_tolerance", 0.05))
    active_thr = float(cfg.get("active_threshold", 1e-4))
    minf = int(cfg.get("min_features", 32)); maxf = int(cfg.get("max_features", 256))
    base = float(torch.ones(H.shape[0], device=H.device) @ H @ torch.ones(H.shape[0], device=H.device))
    base = max(base, 1e-12)
    candidates = []
    warm = None
    for lam in sorted(lambdas, reverse=True):
        w = _fista(H, R, beta, lam, int(cfg.get("max_iter", 400)), float(cfg.get("tol",1e-5)), warm)
        warm = w.detach()
        active = torch.where(w > active_thr)[0]
        diff = 1.0 - w
        geom = math.sqrt(max(float(diff @ H @ diff), 0.0) / base)
        candidates.append((lam, w.detach().clone(), active.detach().clone(), geom))
    feasible = [c for c in candidates if c[3] <= tol_geom and len(c[2]) >= minf]
    if feasible:
        chosen = min(feasible, key=lambda c: len(c[2]))
    else:
        chosen = min(candidates, key=lambda c: c[3])
    lam, w, active, _ = chosen
    if len(active) > maxf:
        active = torch.topk(w, k=maxf).indices.sort().values
    if len(active) < minf:
        active = torch.topk(w, k=min(minf, len(w))).indices.sort().values

    # Metrics must reflect the representation that is actually used at inference.
    # Features removed by the hard active-set truncation have zero effective weight.
    final_w = torch.zeros_like(w)
    final_w[active] = w[active]
    diff = 1.0 - final_w
    geom = math.sqrt(max(float(diff @ H @ diff), 0.0) / base)
    scales = torch.sqrt(w[active].clamp_min(1e-8))
    return {
        "lambda": lam,
        "weights": w.cpu().numpy(),
        "active": active.cpu().numpy(),
        "scales": scales.cpu().numpy(),
        "geometry_error": geom,
    }


def apply_uvarfs(x: torch.Tensor, result: dict) -> torch.Tensor:
    idx = torch.as_tensor(result["active"], device=x.device, dtype=torch.long)
    scales = torch.as_tensor(result["scales"], device=x.device, dtype=x.dtype)
    return x.index_select(-1, idx) * scales
