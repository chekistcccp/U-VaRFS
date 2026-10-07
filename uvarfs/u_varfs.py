from __future__ import annotations
import math, time
import torch
import torch.nn.functional as F
from .numerics import precise_matmul
from .sparse_support import refine_budget, exchange_budget, forward_budget
from .budget_geometry import audit_budget_geometry


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


def _lipschitz_bound(H, P, beta, rscale):
    """Guaranteed spectral upper bound; avoids an underestimated power iterate."""
    rows = H.abs().sum(1)
    if P.numel():
        absolute = P.abs()
        rows = rows + 2.0 * beta * rscale * (absolute @ absolute.sum(0))
    return max(float(rows.max().item()) * (1.0 + 1e-6), 1e-12)


def _objective_certificate(W,H,P,beta,rscale,lambdas):
    """Same box-constrained objective; convex first-order optimality certificate.

    The box linearization gap bounds F(W)-min F. It is separate from the
    iteration-change criterion and from representation geometry feasibility.
    """
    difference=1-W
    representation=.5*(difference*(H @ difference)).sum(0)
    variability=beta*rscale*(P.T @ W).square().sum(0)
    sparsity=torch.as_tensor(lambdas,device=W.device,dtype=W.dtype)*W.sum(0)
    gradient=_apply_A(W,H,P,beta,rscale)-H.sum(1,keepdim=True)+torch.as_tensor(lambdas,device=W.device,dtype=W.dtype)
    gap=(gradient.clamp_min(0)*W-gradient.clamp_max(0)*(1-W)).sum(0)
    return {'objective':representation+variability+sparsity,
            'representation_term':representation,'variability_term':variability,
            'sparsity_term':sparsity,'box_optimality_gap':gap}


@torch.inference_mode(False)
def _batched_fista(H, P, beta, rscale, lambdas, max_iter, tol, check_every=25, log_every=50, label='uvarfs'):
    """Solve all lambda candidates in parallel.

    Columns correspond to lambda values. This is mathematically the same
    proximal-gradient objective as the old per-lambda FISTA path, but converts
    thousands of synchronized GEMVs into GPU-friendly GEMMs.
    """
    m=H.shape[0]
    lams=torch.as_tensor(lambdas,device=H.device,dtype=H.dtype).reshape(1,-1)
    L=_lipschitz_bound(H,P,beta,rscale)
    W=torch.ones((m,len(lambdas)),device=H.device,dtype=H.dtype)
    Y=W.clone()
    t=torch.ones((1,len(lambdas)),device=H.device,dtype=H.dtype)
    restart_counts=torch.zeros(len(lambdas),device=H.device,dtype=torch.long)
    target=H.sum(1,keepdim=True)
    t0=time.time()
    residual = torch.full((len(lambdas),),float('inf'),device=H.device)
    relative = residual.clone()

    for it in range(1,max_iter+1):
        grad=_apply_A(Y,H,P,beta,rscale)-target
        Z=Y-grad/L
        # On [0,1], the L1 proximal operator is a positive threshold + box projection.
        Wn=torch.clamp(Z-lams/L,0.0,1.0)
        tn=0.5*(1.0+torch.sqrt(1.0+4.0*t*t))
        # Restart each lambda's momentum independently when it points uphill.
        restart=((Y-Wn)*(Wn-W)).sum(0)>0
        momentum=torch.where(restart[None,:],0.0,(t-1.0)/tn)
        Y=Wn+momentum*(Wn-W)
        tn=torch.where(restart[None,:],1.0,tn)
        restart_counts+=restart

        need_check=(it % check_every == 0) or it==max_iter
        need_log=(it==1) or (it % log_every == 0) or it==max_iter
        stop=False
        rel_val=None
        if need_check or need_log:
            # One synchronization every check_every iterations instead of every iteration.
            relative=(torch.linalg.vector_norm(Wn-W,dim=0)/
                 torch.linalg.vector_norm(W,dim=0).clamp_min(1e-8))
            gradient=_apply_A(Wn,H,P,beta,rscale)-target
            projected=torch.clamp(Wn-(gradient+lams)/L,0.0,1.0)
            residual=(torch.linalg.vector_norm(Wn-projected,dim=0)/
                      torch.linalg.vector_norm(Wn,dim=0).clamp_min(1.0))
            rel_val=float(relative.max().item())
            residual_val=float(residual.max().item())
            if need_check and max(rel_val,residual_val) < tol:
                stop=True
        W,t=Wn,tn

        if need_log:
            elapsed=time.time()-t0
            active=(W > 1e-4).sum(dim=0)
            amin=int(active.min().item()); amax=int(active.max().item())
            print(
                f'[{label}] iter={it}/{max_iter} max_rel={rel_val:.3e} '
                f'max_projected_residual={residual_val:.3e} active={amin}-{amax} elapsed={elapsed:.1f}s',
                flush=True
            )
        if stop:
            print(f'[{label}] converged at iter={it}, max_rel={rel_val:.3e}',flush=True)
            break
    certificate=_objective_certificate(W,H,P,beta,rscale,lambdas)
    return W,L,it,{'relative_change':relative.cpu().tolist(),
                   'projected_residual':residual.cpu().tolist(),
                   'converged':((relative<tol)&(residual<tol)).cpu().tolist(),
                   'objective_converged':(certificate['box_optimality_gap']<=tol).cpu().tolist(),
                   'restart_counts':restart_counts.cpu().tolist(),
                   **{k:v.cpu().tolist() for k,v in certificate.items()}}


def _select_solution(W, H, lambdas, cfg, solver, objective_args=None, retained_override=None):
    """Apply the feature budget to EVERY path point before checking geometry.

    Feasibility refers to budgeted weights in the original weighted Gram,
    before detector row normalization. No anomaly labels enter selection.
    """
    m=W.shape[0]
    minf=min(int(cfg.get('min_features',32)),m)
    maxf=min(int(cfg.get('max_features',256)),m)
    if not 1 <= minf <= maxf:
        raise ValueError('feature budgets must satisfy 1 <= min_features <= max_features')
    threshold=float(cfg.get('active_threshold',1e-4))
    tolerance=float(cfg.get('geometry_tolerance',0.05))
    base=H.sum().clamp_min(1e-12)
    retained=torch.zeros_like(W)
    indices=[]
    counts=(W>threshold).sum(0).cpu().tolist()
    for j in range(W.shape[1]):
        active=torch.where(W[:,j]>threshold)[0]
        if len(active)>maxf or len(active)<minf:
            k=maxf if len(active)>maxf else minf
            active=torch.argsort(W[:,j],descending=True,stable=True)[:k].sort().values
        retained[active,j]=W[active,j]
        indices.append(active)
    if retained_override is not None:
        if (retained_override.shape!=W.shape or not torch.isfinite(retained_override).all() or
            (retained_override<0).any() or (retained_override>1).any()):
            raise ValueError('invalid budgeted U-VaRFS weights')
        retained=retained_override
        indices=[]
        for j in range(W.shape[1]):
            active=torch.where(retained[:,j]>0)[0]
            if len(active)>maxf:
                raise ValueError('refined support exceeds the common feature budget')
            if len(active)<minf:
                order=torch.argsort(W[:,j],descending=True,stable=True)
                active=torch.cat([active,order[~torch.isin(order,active)][:minf-len(active)]])
            indices.append(active.sort().values)
    original=1-W
    difference=1-retained
    original_error=torch.sqrt(((original*(H @ original)).sum(0)/base).clamp_min(0)).cpu().tolist()
    errors=torch.sqrt(((difference*(H @ difference)).sum(0)/base).clamp_min(0)).cpu().tolist()
    # H is entrywise nonnegative. On this fixed support, unit weights minimize
    # geometry error over the box, independent of beta/lambda. This is a lower
    # bound for this support only, not for every possible max_features subset.
    support=torch.zeros_like(W)
    for j,active in enumerate(indices):
        support[active,j]=1
    missing=1-support
    support_floor=torch.sqrt(((missing*(H @ missing)).sum(0)/base).clamp_min(0)).cpu().tolist()
    budget_certificate={}
    if objective_args is not None:
        P,beta,rscale=objective_args
        budget_certificate={k:v.cpu().tolist() for k,v in
            _objective_certificate(retained,H,P,beta,rscale,lambdas).items()}
    nonzero=(retained>0).sum(0).cpu().tolist()
    path=[]
    for j,lam in enumerate(lambdas):
        path.append({'lambda':lam,'continuous_active_features':counts[j],
                     'retained_features':len(indices[j]),'nonzero_features':nonzero[j],
                     'continuous_geometry_error':original_error[j],
                     'geometry_error':errors[j],
                     'fixed_support_geometry_floor':support_floor[j],
                     'fixed_support_can_meet_tolerance':support_floor[j]<=tolerance,
                     'feasible':errors[j]<=tolerance and nonzero[j]>=minf,
                     'relative_change':solver['relative_change'][j],
                     'projected_residual':solver['projected_residual'][j],
                     'solver_converged':bool(solver['converged'][j])})
        for field in ['objective','representation_term','variability_term','sparsity_term',
                      'box_optimality_gap','objective_converged','restart_counts']:
            if field in solver:
                path[-1][field]=solver[field][j]
        path[-1]['objective_certificate_scope']='continuous_full_weights'
        for field,values in budget_certificate.items():
            path[-1]['budgeted_'+field]=values[j]
        if budget_certificate:
            path[-1]['budgeted_objective_converged']=budget_certificate['box_optimality_gap'][j]<=float(cfg.get('tol',1e-5))
    feasible=[j for j,p in enumerate(path) if p['feasible']]
    chosen=min(feasible,key=lambda j:(len(indices[j]),errors[j])) if feasible else min(range(len(path)),key=lambda j:errors[j])
    return chosen,indices[chosen],retained[:,chosen],path


def _selection_result(W,selection,lambdas,L,iterations,seconds,strategy):
    chosen,active,final_w,path=selection
    point=path[chosen]; lam=lambdas[chosen]
    return {
        'lambda':lam,'weights':W[:,chosen].cpu().numpy(),
        'active':active.cpu().numpy(),'scales':torch.sqrt(final_w[active].clamp_min(0)).cpu().numpy(),
        'budgeted_weights':final_w.cpu().numpy(),
        'geometry_error':point['geometry_error'],'continuous_geometry_error':point['continuous_geometry_error'],
        'feasible':point['feasible'],
        'selection_reason':'sparsest_feasible' if point['feasible'] else 'minimum_geometry_error_fallback',
        'lambda_at_grid_boundary':lam in (min(lambdas),max(lambdas)),
        'lambda_path':path,'solver_converged':point['solver_converged'],
        'projected_residual':point['projected_residual'],'solver_iterations':iterations,
        'solver_lipschitz':L,'fit_seconds':seconds,'solver_matmul_precision':'highest',
        'objective':point['objective'],'box_optimality_gap':point['box_optimality_gap'],
        'objective_converged':point['objective_converged'],'momentum_restarts':point['restart_counts'],
        'objective_certificate_scope':'continuous_full_weights','sparsity_strategy':strategy,
        'fixed_support_geometry_floor':point['fixed_support_geometry_floor'],
        'fixed_support_can_meet_tolerance':point['fixed_support_can_meet_tolerance'],
        **{k:v for k,v in point.items() if k.startswith('budgeted_') or k in
           {'budget_source','legacy_budgeted_objective','legacy_geometry_error','objective_improvement',
            'geometry_improvement','fixed_support_box_gap','fixed_support_converged','support_refit_iterations'}},
        **{k:v for k,v in point.items() if k.startswith(('exchange_','prune_refit_','forward_'))},
    }


def _merge_exchange_diagnostics(points, previous, exchanges, retained, H, P, beta, rscale, lambdas, tol):
    gradient=_apply_A(retained,H,P,beta,rscale)-H.sum(1,keepdim=True)+torch.as_tensor(lambdas,device=H.device,dtype=H.dtype)
    gaps=(gradient.clamp_min(0)*retained-gradient.clamp_max(0)*(1-retained)*(retained>0)).sum(0).cpu().tolist()
    for j,(point,old,diagnostic) in enumerate(zip(points,previous,exchanges)):
        point.update(old); point.update(diagnostic)
        if diagnostic['exchange_source']!='prune_refit':
            point['budget_source']=diagnostic['exchange_source']
            point['fixed_support_box_gap']=gaps[j]
            point['fixed_support_converged']=gaps[j]<=tol
            point['objective_improvement']+=diagnostic['exchange_objective_improvement']
            point['geometry_improvement']+=diagnostic['exchange_geometry_improvement']


def _merge_forward_diagnostics(points, previous, diagnostics):
    for point,old,diagnostic in zip(points,previous,diagnostics):
        # Preserve old search diagnostics without replacing the new objective,
        # geometry or feasibility computed from the actual retained weights.
        for key,value in old.items():
            if key.startswith(('legacy_','exchange_','prune_refit_')) or key in {
                'budget_source','objective_improvement','geometry_improvement',
                'fixed_support_box_gap','fixed_support_converged','support_refit_iterations'}:
                point[key]=value
        point.update(diagnostic)
        if diagnostic['forward_candidate_accepted']:
            point['budget_source']=diagnostic['forward_source']
            point['fixed_support_box_gap']=diagnostic['forward_fixed_support_box_gap']
            point['fixed_support_converged']=diagnostic['forward_fixed_support_converged']
            point['objective_improvement']+=diagnostic['forward_objective_improvement']
            point['geometry_improvement']+=diagnostic['forward_geometry_improvement']


@precise_matmul()
def fit_uvarfs(X: torch.Tensor, variability_vectors: torch.Tensor, cfg: dict, label: str='uvarfs') -> dict:
    t0=time.time()
    if X.ndim!=2 or X.shape[0]<2 or X.shape[1]<1 or not torch.isfinite(X).all():
        raise ValueError('U-VaRFS requires finite normal-reference features with at least two samples')
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

    P=variability_vectors.to(device=X.device,dtype=X.dtype).T  # M x V
    if P.ndim!=2 or P.shape[0]!=m or not torch.isfinite(P).all() or (P<0).any():
        raise ValueError('variability must be a finite nonnegative V x M matrix')
    if P.numel():
        P=P/(P.norm(dim=0,keepdim=True)+1e-8)

    beta=float(cfg.get('beta',0.05))
    rscale=_r_scale(H,P)
    lambdas=[float(x) for x in cfg.get('lambda_grid',[0.1,0.05,0.01,0.005,0.001])]
    tol_geom=float(cfg.get('geometry_tolerance',0.05))
    max_iter=int(cfg.get('max_iter',400))
    tol=float(cfg.get('tol',1e-5))
    check_every=int(cfg.get('check_every',25))
    log_every=int(cfg.get('log_every',50))
    strategy=cfg.get('sparsity_strategy','objective_forward_refit')

    if (not lambdas or any(not math.isfinite(lam) or lam<0 for lam in lambdas)
        or not math.isfinite(beta) or beta<0 or not math.isfinite(tol) or tol<=0
        or max_iter<1 or check_every<1 or log_every<1 or strategy not in {'top_weights','objective_prune_refit','objective_exchange_refit','objective_forward_refit'}):
        raise ValueError('invalid U-VaRFS path or solver configuration')
    W,L,used_iter,solver=_batched_fista(
        H,P,beta,rscale,lambdas,max_iter,tol,
        check_every=check_every,log_every=log_every,label=label
    )

    legacy_selection=_select_solution(W,H,lambdas,cfg,solver,(P,beta,rscale))
    legacy=_selection_result(W,legacy_selection,lambdas,L,used_iter,time.time()-t0,'top_weights')
    selection=legacy_selection
    prune_refit=None
    exchange_refit=None
    if strategy!='top_weights':
        retained,support_diagnostics=refine_budget(W,H,P,beta,rscale,lambdas,cfg,label)
        selection=_select_solution(W,H,lambdas,cfg,solver,(P,beta,rscale),retained)
        for point,diagnostic in zip(selection[3],support_diagnostics):
            point.update(diagnostic)
        prune_refit=_selection_result(W,selection,lambdas,L,used_iter,time.time()-t0,'objective_prune_refit')
        if strategy in {'objective_exchange_refit','objective_forward_refit'}:
            retained,exchanges=exchange_budget(retained,W,H,P,beta,rscale,lambdas,cfg,label)
            selection=_select_solution(W,H,lambdas,cfg,solver,(P,beta,rscale),retained)
            _merge_exchange_diagnostics(selection[3],support_diagnostics,exchanges,retained,H,P,beta,rscale,lambdas,tol)
            exchange_refit=_selection_result(W,selection,lambdas,L,used_iter,time.time()-t0,'objective_exchange_refit')
        if strategy=='objective_forward_refit':
            previous=selection[3]
            retained,forward=forward_budget(retained,W,H,P,beta,rscale,lambdas,cfg,label)
            selection=_select_solution(W,H,lambdas,cfg,solver,(P,beta,rscale),retained)
            _merge_forward_diagnostics(selection[3],previous,forward)
    result=_selection_result(W,selection,lambdas,L,used_iter,time.time()-t0,strategy)
    if strategy!='top_weights':
        result['legacy_top_weights']=legacy
    if strategy in {'objective_exchange_refit','objective_forward_refit'}:
        result['prune_refit']=prune_refit
    if strategy=='objective_forward_refit':
        result['exchange_refit']=exchange_refit
    budget_audit=audit_budget_geometry(H,cfg)
    for candidate in [result,legacy,prune_refit,exchange_refit]:
        if candidate is not None:
            candidate['budget_geometry_audit']=budget_audit
    print(f'[{label}] normal budget audit: global_geometry_floor>={budget_audit["geometry_error_lower_bound"]:.4f} '
          f'necessary_features>={budget_audit["necessary_features_lower_bound"]} (diagnostic only)',flush=True)
    print(
        f'[{label}] done: strategy={strategy} lambda={result["lambda"]:g} active={len(result["active"])}/{m} '
        f'geometry_error={result["geometry_error"]:.4f} iterations={used_iter} elapsed={result["fit_seconds"]:.1f}s',
        flush=True
    )
    if not result['feasible']:
        print(f'[{label}] fallback: no budgeted representation meets geometry tolerance={tol_geom:g}',flush=True)
    if not result['solver_converged']:
        print(f'[{label}] continuous solve reached iteration limit; residual={result["projected_residual"]:.3e}',flush=True)
    # Diagnostic only: never extend the selected lambda path or tune beta.
    if not result['feasible'] and cfg.get('diagnose_infeasible',False):
        diagnostic_W,_,_,diagnostic_solver=_batched_fista(
            H,P,beta,rscale,[0.0],max_iter,tol,check_every,log_every,
            label=f'{label}:lambda0-diagnostic')
        _,_,_,diagnostic_path=_select_solution(diagnostic_W,H,[0.0],cfg,diagnostic_solver,(P,beta,rscale))
        legacy['zero_lambda_diagnostic']={**diagnostic_path[0],'selection_candidate':False}
        if strategy!='top_weights':
            retained,diagnostics=refine_budget(diagnostic_W,H,P,beta,rscale,[0.0],cfg,f'{label}:lambda0')
            _,_,_,diagnostic_path=_select_solution(diagnostic_W,H,[0.0],cfg,diagnostic_solver,(P,beta,rscale),retained)
            diagnostic_path[0].update(diagnostics[0])
            if prune_refit is not None:
                prune_refit['zero_lambda_diagnostic']={**diagnostic_path[0],'selection_candidate':False}
            if strategy in {'objective_exchange_refit','objective_forward_refit'}:
                retained,diagnostics=exchange_budget(retained,diagnostic_W,H,P,beta,rscale,[0.0],cfg,f'{label}:lambda0')
                _,_,_,updated=_select_solution(diagnostic_W,H,[0.0],cfg,diagnostic_solver,(P,beta,rscale),retained)
                previous={k:v for k,v in diagnostic_path[0].items() if k in
                    {'budget_source','legacy_budgeted_objective','legacy_geometry_error','objective_improvement','geometry_improvement',
                     'fixed_support_box_gap','fixed_support_converged','support_refit_iterations','support_optimality_scope'}}
                _merge_exchange_diagnostics(updated,[previous],diagnostics,retained,H,P,beta,rscale,[0.0],tol)
                diagnostic_path=updated
                if exchange_refit is not None:
                    exchange_refit['zero_lambda_diagnostic']={**diagnostic_path[0],'selection_candidate':False}
            if strategy=='objective_forward_refit':
                previous=diagnostic_path
                retained,diagnostics=forward_budget(retained,diagnostic_W,H,P,beta,rscale,[0.0],cfg,f'{label}:lambda0')
                _,_,_,diagnostic_path=_select_solution(diagnostic_W,H,[0.0],cfg,diagnostic_solver,(P,beta,rscale),retained)
                _merge_forward_diagnostics(diagnostic_path,previous,diagnostics)
        result['zero_lambda_diagnostic']={**diagnostic_path[0],'selection_candidate':False}
    return result


def apply_uvarfs(x: torch.Tensor, result: dict) -> torch.Tensor:
    idx=torch.as_tensor(result['active'],device=x.device,dtype=torch.long)
    scales=torch.as_tensor(result['scales'],device=x.device,dtype=x.dtype)
    return x.index_select(-1,idx)*scales
