from __future__ import annotations
import math, time
import numpy as np
import torch
import torch.nn.functional as F


def _r_scale(H: torch.Tensor, P: torch.Tensor) -> float:
    """Return scale matching ||P P^T||_F to ||H||_F without materializing P P^T."""
    if P.numel() == 0:
        return 0.0
    # ||P P^T||_F == ||P^T P||_F for the Frobenius norm.
    small=P.T @ P
    return float((H.norm()/small.norm().clamp_min(1e-12)).item())


def _apply_A(x: torch.Tensor, H: torch.Tensor, P: torch.Tensor, beta: float, rscale: float) -> torch.Tensor:
    out=H @ x
    if P.numel():
        out=out + (2.0*beta*rscale) * (P @ (P.T @ x))
    return out


def _power_lipschitz(H: torch.Tensor, P: torch.Tensor, beta: float, rscale: float, iters: int = 30) -> float:
    x=torch.randn(H.shape[0],device=H.device,dtype=H.dtype)
    x=x/x.norm().clamp_min(1e-8)
    for _ in range(iters):
        x=_apply_A(x,H,P,beta,rscale)
        x=x/x.norm().clamp_min(1e-8)
    ax=_apply_A(x,H,P,beta,rscale)
    return float((x @ ax).abs().item())+1e-6


@torch.inference_mode(False)
def _batched_fista(H, P, beta, rscale, lambdas, max_iter, tol, check_every=25, log_every=50, label='uvarfs'):
    """Solve all lambda candidates in parallel.

    Columns correspond to lambda values. This is mathematically the same
    proximal-gradient objective as the old per-lambda FISTA path, but converts
    thousands of synchronized GEMVs into GPU-friendly GEMMs.
    """
    m=H.shape[0]
    lams=torch.as_tensor(lambdas,device=H.device,dtype=H.dtype).reshape(1,-1)
    L=_power_lipschitz(H,P,beta,rscale)
    W=torch.ones((m,len(lambdas)),device=H.device,dtype=H.dtype)
    Y=W.clone()
    t=1.0
    ones=torch.ones_like(W)
    t0=time.time()

    for it in range(1,max_iter+1):
        grad=H @ (Y-ones)
        if P.numel():
            grad=grad + (2.0*beta*rscale) * (P @ (P.T @ Y))
        Z=Y-grad/L
        Wn=torch.clamp(torch.sign(Z)*torch.relu(Z.abs()-lams/L),0.0,1.0)
        tn=0.5*(1.0+math.sqrt(1.0+4.0*t*t))
        Y=Wn+((t-1.0)/tn)*(Wn-W)

        need_check=(it % check_every == 0) or it==max_iter
        need_log=(it==1) or (it % log_every == 0) or it==max_iter
        stop=False
        rel_val=None
        if need_check or need_log:
            # One synchronization every check_every iterations instead of every iteration.
            rel=(torch.linalg.vector_norm(Wn-W,dim=0)/
                 torch.linalg.vector_norm(W,dim=0).clamp_min(1e-8))
            rel_val=float(rel.max().item())
            if need_check and rel_val < tol:
                stop=True
        W,t=Wn,tn

        if need_log:
            elapsed=time.time()-t0
            active=(W > 1e-4).sum(dim=0)
            amin=int(active.min().item()); amax=int(active.max().item())
            print(
                f'[{label}] iter={it}/{max_iter} max_rel={rel_val:.3e} '
                f'active={amin}-{amax} elapsed={elapsed:.1f}s',
                flush=True
            )
        if stop:
            print(f'[{label}] converged at iter={it}, max_rel={rel_val:.3e}',flush=True)
            break
    return W,L,it


def fit_uvarfs(X: torch.Tensor, variability_vectors: torch.Tensor, cfg: dict, label: str='uvarfs') -> dict:
    t0=time.time()
    X=F.normalize(X.float(),dim=1)
    n=max(X.shape[0],1)
    m=X.shape[1]
    print(f'[{label}] start: patches={X.shape[0]} features={m} lambdas={len(cfg.get("lambda_grid",[]))} device={X.device}',flush=True)

    # H is the exact feature-space representation-preservation matrix used by U-VaRFS.
    G=(X.T @ X)/n
    H=G*G
    ones0=torch.ones(H.shape[0],device=H.device,dtype=H.dtype)
    hscale=(ones0 @ H @ ones0).clamp_min(1e-12)
    H=H/hscale
    del G

    P=variability_vectors.float().T  # M x V
    if P.numel():
        P=P/(P.norm(dim=0,keepdim=True)+1e-8)

    beta=float(cfg.get('beta',0.05))
    rscale=_r_scale(H,P)
    lambdas=[float(x) for x in cfg.get('lambda_grid',[0.1,0.05,0.01,0.005,0.001])]
    tol_geom=float(cfg.get('geometry_tolerance',0.05))
    active_thr=float(cfg.get('active_threshold',1e-4))
    minf=int(cfg.get('min_features',32))
    maxf=int(cfg.get('max_features',256))
    max_iter=int(cfg.get('max_iter',400))
    tol=float(cfg.get('tol',1e-5))
    check_every=int(cfg.get('check_every',25))
    log_every=int(cfg.get('log_every',50))

    base=float(torch.ones(H.shape[0],device=H.device) @ H @ torch.ones(H.shape[0],device=H.device))
    base=max(base,1e-12)

    W,L,used_iter=_batched_fista(
        H,P,beta,rscale,lambdas,max_iter,tol,
        check_every=check_every,log_every=log_every,label=label
    )

    # Evaluate all lambda candidates in one matrix operation.
    D=1.0-W
    HD=H @ D
    geom=torch.sqrt(torch.clamp((D*HD).sum(dim=0)/base,min=0.0))
    active_counts=(W>active_thr).sum(dim=0)

    candidates=[]
    for j,lam in enumerate(lambdas):
        active=torch.where(W[:,j]>active_thr)[0]
        candidates.append((lam,W[:,j].detach().clone(),active.detach().clone(),float(geom[j].item())))

    feasible=[c for c in candidates if c[3] <= tol_geom and len(c[2]) >= minf]
    chosen=min(feasible,key=lambda c:len(c[2])) if feasible else min(candidates,key=lambda c:c[3])
    lam,w,active,_=chosen

    if len(active)>maxf:
        active=torch.topk(w,k=maxf).indices.sort().values
    if len(active)<minf:
        active=torch.topk(w,k=min(minf,len(w))).indices.sort().values

    final_w=torch.zeros_like(w)
    final_w[active]=w[active]
    diff=1.0-final_w
    geom_final=math.sqrt(max(float(diff @ H @ diff),0.0)/base)
    scales=torch.sqrt(w[active].clamp_min(1e-8))

    elapsed=time.time()-t0
    print(
        f'[{label}] done: lambda={lam:g} active={len(active)}/{m} '
        f'geometry_error={geom_final:.4f} iterations={used_iter} elapsed={elapsed:.1f}s',
        flush=True
    )
    return {
        'lambda':lam,
        'weights':w.cpu().numpy(),
        'active':active.cpu().numpy(),
        'scales':scales.cpu().numpy(),
        'geometry_error':geom_final,
        'solver_iterations':used_iter,
        'solver_lipschitz':L,
        'fit_seconds':elapsed,
    }


def apply_uvarfs(x: torch.Tensor, result: dict) -> torch.Tensor:
    idx=torch.as_tensor(result['active'],device=x.device,dtype=torch.long)
    scales=torch.as_tensor(result['scales'],device=x.device,dtype=x.dtype)
    return x.index_select(-1,idx)*scales
