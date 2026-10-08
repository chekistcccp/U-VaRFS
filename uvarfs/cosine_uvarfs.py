"""User-approved cosine/simplex/cardinality U-VaRFS; normal data only.

The original quadratic solver remains in u_varfs.py as a complete control.
No convex/global certificate is claimed for this nonconvex finite support pool.
"""
from __future__ import annotations

import time

import numpy as np
from scipy.optimize import minimize
import torch

from .numerics import precise_matmul

OBJECTIVE = 'cosine_simplex_cardinality'


def project_simplex(values, floor_mass=1e-4):
    """Euclidean projection onto sum(p)=1, p_j>=floor_mass/K."""
    if values.ndim != 1 or not len(values) or not torch.isfinite(values).all() or not 0 < floor_mass < 1:
        raise ValueError('invalid simplex or positive floor mass')
    v = values.double()
    floor = floor_mass / len(v)
    shifted = v-floor
    ordered = shifted.sort(descending=True, stable=True).values
    thresholds = (ordered.cumsum(0)-(1-floor_mass))/torch.arange(1,len(v)+1,device=v.device)
    rho = torch.where(ordered>thresholds)[0][-1]
    return ((shifted-thresholds[rho]).clamp_min(0)+floor).to(values.dtype)


class CosineObjective:
    """Exact row-blocked loss and analytic gradient in sample space.

    One full-reference N-by-N Gram is reused; no N-by-N autograd graph is kept.
    The gradient can cover a fixed support or all dimensions for support exchange.
    """
    def __init__(self, x, variability, beta=.002, chunk=512):
        if x.ndim!=2 or min(x.shape)<1 or not torch.isfinite(x).all():
            raise ValueError('normal features must be a finite nonempty matrix')
        if variability.ndim!=2 or variability.shape[1]!=x.shape[1] or not torch.isfinite(variability).all() or (variability<0).any():
            raise ValueError('variability must be finite nonnegative V-by-M')
        if not 0<=beta or not torch.isfinite(torch.tensor(beta)) or chunk<1:
            raise ValueError('invalid beta or geometry chunk')
        # Per-row rescaling is invariant for BOTH full and weighted cosines.
        dtype=torch.float64 if x.dtype==torch.float64 else torch.float32
        x=x.to(dtype)
        row_scale=x.abs().amax(1,keepdim=True)
        if (row_scale==0).any():
            raise ValueError('normal reference contains a zero feature row')
        x=x/row_scale
        self.x=x/x.norm(dim=1,keepdim=True)
        self.reference=self.x@self.x.T
        self.denominator=self.reference.double().square().sum()
        P=variability.T.to(device=x.device,dtype=dtype)
        P=P/(P.norm(dim=0,keepdim=True)+1e-8)  # Same column convention as original U-VaRFS.
        peak=P.abs().max() if P.numel() else P.new_zeros(())
        self.P=P/peak if peak>0 else P
        self.var_base=self.P.mean(0).square().sum()
        self.beta=float(beta); self.chunk=int(chunk)
        self.evaluations=0

    @torch.no_grad()
    def evaluate(self, active, p, gradient=False, full_gradient=False):
        xs=self.x[:,active]
        h=xs.square()@p
        if not torch.isfinite(p).all() or (p<0).any() or not torch.isfinite(h).all() or (h<=0).any():
            raise ValueError('invalid candidate weights or zero cosine row')
        d=h.rsqrt()
        normalized=(xs*torch.sqrt(p))*d[:,None]
        targets=self.x if full_gradient else xs
        if gradient:
            t=targets*d[:,None]
            g=targets.new_zeros(targets.shape[1],dtype=torch.float64)
        squared=self.denominator.new_zeros(())
        for start in range(0,len(h),self.chunk):
            end=min(start+self.chunk,len(h))
            actual=normalized[start:end]@normalized.T
            delta=actual-self.reference[start:end]
            squared+=delta.double().square().sum()
            if gradient:
                # dC_ik/dp_j = d_i*d_k*x_ij*x_kj
                #              -.5*C_ik*(x_ij²/h_i+x_kj²/h_k).
                # Symmetry combines the last two terms into this row reduction.
                first=(t[start:end]*(delta@t)).sum(0,dtype=torch.float64)
                row=(delta*actual).sum(1)
                second=(targets[start:end].square()*(row/h[start:end])[:,None]).sum(0,dtype=torch.float64)
                g+=first-second
        projected=self.P[active].T@p
        var=projected.square().sum()/self.var_base if self.var_base>0 else squared.new_zeros(())
        rep=.5*squared/self.denominator
        value=rep+self.beta*var
        if gradient:
            pv=self.P if full_gradient else self.P[active]
            vg=2*self.beta*(pv@projected)/self.var_base if self.var_base>0 else torch.zeros_like(g)
            g=(g/self.denominator+vg).to(p.dtype)
        if not torch.isfinite(value) or (gradient and not torch.isfinite(g).all()):
            raise ValueError('nonfinite cosine objective or gradient')
        self.evaluations+=1
        terms={'representation_term':float(rep),'variability_term':float(self.beta*var),
               'relative_variability':float(var),'geometry_error':float(torch.sqrt(squared/self.denominator))}
        return value,g if gradient else None,terms


def polish_simplex(engine, active, p, cfg, label, state=None):
    """Guarded SQP endpoint refinement of the SAME fixed-support objective.

    SciPy controls only the <=256-dimensional vector on CPU; the blocked Torch
    loss/analytic gradient keep the original dtype/device, normal data and loss.
    Internal SQP iterates need not be monotone. Only a projected, independently
    evaluated endpoint with strict actual descent replaces the incoming point.
    SciPy success is never used as a first-order or global certificate.
    """
    floor=float(cfg.get('cosine_weight_floor_mass',1e-4))
    maximum=int(cfg.get('cosine_polish_max_iter',200))
    ftol=float(cfg.get('cosine_polish_ftol',1e-12))
    if maximum<0 or not np.isfinite(ftol) or ftol<=0:
        raise ValueError('invalid cosine polish limits')
    value,g,terms=engine.evaluate(active,p,True) if state is None else state
    residual=float((p-project_simplex(p-g,floor)).norm())
    diag={'method':'slsqp_fixed_support','attempted':False,'accepted':False,
          'before':float(value),'after':float(value),'residual_before':residual,
          'residual_after':residual,'optimizer_success':None,'optimizer_status':None,
          'optimizer_message':None,'iterations':0,'objective_evaluations':0,
          'seconds':0.,'termination':'disabled' if maximum==0 else 'already_stationary'}
    if maximum==0 or residual<=float(cfg.get('tol',1e-5)):
        return p,terms,diag
    started=time.perf_counter(); evaluations=engine.evaluations
    print(f'[{label}] cosine-polish start K={len(active)} loss={float(value):.6g} residual={residual:.3e}',flush=True)
    def objective(weights):
        trial=p.new_tensor(weights)
        loss,gradient,_=engine.evaluate(active,trial,True)
        return float(loss),gradient.detach().double().cpu().numpy()
    callback_iterations=0; last_log=started
    def progress(_):
        nonlocal callback_iterations,last_log
        callback_iterations+=1
        now=time.perf_counter()
        if now-last_log>=30:
            print(f'[{label}] cosine-polish iter={callback_iterations} elapsed={now-started:.1f}s',flush=True)
            last_log=now
    solved=minimize(objective,p.detach().double().cpu().numpy(),method='SLSQP',jac=True,
        bounds=[(floor/len(p),1.)]*len(p),
        constraints={'type':'eq','fun':lambda weights:weights.sum()-1.,
                     'jac':lambda weights:np.ones_like(weights)},
        callback=progress,options={'ftol':ftol,'maxiter':maximum})
    diag.update(attempted=True,optimizer_success=bool(solved.success),
                optimizer_status=int(solved.status),optimizer_message=str(solved.message),
                iterations=int(solved.nit),termination='endpoint_rejected')
    # Invalid endpoints cannot replace a valid projected descent solution.
    endpoint=np.asarray(solved.x)
    trial=p.new_tensor(endpoint) if endpoint.shape==tuple(p.shape) else None
    if trial is not None and bool(torch.isfinite(trial).all()):
        trial=project_simplex(trial,floor)
        trial_value,trial_g,trial_terms=engine.evaluate(active,trial,True)
        if trial_value<value:
            p=trial; value=trial_value; g=trial_g; terms=trial_terms
            diag.update(accepted=True,termination='strict_objective_descent')
    residual=float((p-project_simplex(p-g,floor)).norm())
    diag.update(after=float(value),residual_after=residual,
                objective_evaluations=engine.evaluations-evaluations,
                seconds=time.perf_counter()-started)
    print(f'[{label}] cosine-polish done accepted={diag["accepted"]} '
          f'optimizer_status={diag["optimizer_status"]} loss={float(value):.6g} '
          f'residual={residual:.3e} elapsed={diag["seconds"]:.1f}s',flush=True)
    return p,terms,diag


def refit_simplex(engine, active, initial, cfg, label):
    floor=float(cfg.get('cosine_weight_floor_mass',1e-4))
    p=project_simplex(initial,floor)
    maximum=int(cfg.get('cosine_refit_max_iter',cfg.get('max_iter',2000)))
    tolerance=float(cfg.get('tol',1e-5))
    step=float(cfg.get('cosine_initial_step',.05))
    backtracks=int(cfg.get('cosine_line_search_steps',20))
    log_every=max(1,int(cfg.get('log_every',50)))
    status='iteration_limit'; used=0; relative=0.; history=[]
    value,g,terms=engine.evaluate(active,p,True)
    for iteration in range(maximum):
        residual=float((p-project_simplex(p-g,floor)).norm())
        if residual<=tolerance:
            status='fixed_support_stationary'; break
        if iteration%log_every==0:
            print(f'[{label}] cosine-refit iter={iteration} K={len(active)} loss={float(value):.6g} '
                  f'geometry={terms["geometry_error"]:.4f} residual={residual:.3e}',flush=True)
        accepted=False; trial_step=step
        for _ in range(backtracks):
            trial=project_simplex(p-trial_step*g,floor)
            change=trial-p
            if not bool(change.abs().max()>0):
                break
            try:
                trial_value,_,_=engine.evaluate(active,trial)
            except ValueError:
                trial_step*=.5; continue
            # A strict descent test, with Armijo allowance only in the downhill
            # direction; never accept an uphill step for numerical convenience.
            if trial_value<value and trial_value<=value+1e-4*(g.double()*change.double()).sum():
                accepted=True; break
            trial_step*=.5
        if not accepted:
            status='line_search_stalled'; break
        old_p=p; old_g=g; old_value=float(value)
        p=trial
        relative=float((p-old_p).norm()/old_p.norm().clamp_min(1e-12))
        value,g,terms=engine.evaluate(active,p,True)
        history.append({'iteration':iteration+1,'before':old_value,'after':float(value)})
        diff=p-old_p; curvature=(diff.double()*(g-old_g).double()).sum()
        step=float((diff.double().square().sum()/curvature).clamp(1e-8,1e3)) if curvature>0 else trial_step
        used=iteration+1
    projected_status=status; projected_iterations=used
    old_p=p
    p,terms,polish=polish_simplex(engine,active,p,cfg,label,(value,g,terms))
    if polish['accepted']:
        relative=float((p-old_p).norm()/old_p.norm().clamp_min(1e-12))
        history.append({'iteration':used+polish['iterations'],'phase':'slsqp_endpoint',
                        'before':polish['before'],'after':polish['after']})
    residual=polish['residual_after']
    if residual<=tolerance:
        status='fixed_support_stationary'
    elif polish['accepted']:
        status='polished_nonstationary'
    return p,terms,{'fixed_support_first_order_residual':residual,
        'fixed_support_converged':residual<=tolerance,'solver_status':status,
        'solver_iterations':used+polish['iterations'],'projected_refit_iterations':projected_iterations,
        'projected_refit_status':projected_status,'polish':polish,
        'relative_change':relative,'descent_history':history}


def covering_support(x, ranking, count):
    """Deterministic zero-row coverage repair; not a global set-cover solver."""
    initial=ranking[:count]
    if bool((x[:,initial].square().sum(1)>0).all()):
        return initial.sort().values
    chosen=[]; covered=torch.zeros(len(x),device=x.device,dtype=torch.bool)
    while not bool(covered.all()) and len(chosen)<count:
        gain=(x[~covered]!=0).sum(0)
        if chosen:
            gain[chosen]=-1
        best=int(ranking[gain[ranking].argmax()])
        if int(gain[best])<=0:
            return None
        chosen.append(best); covered|=x[:,best]!=0
    if not bool(covered.all()):
        return None
    chosen+= [int(j) for j in ranking.cpu().tolist() if int(j) not in chosen][:count-len(chosen)]
    return torch.tensor(sorted(chosen),device=x.device,dtype=torch.long)


def prune_cosine_support(engine, active, p, count, cfg, label):
    """Compare finite deletions on the same normal objective, with a legacy guard.

    The tangent first-order deletion score vanishes at interior stationary
    simplex solutions. Finite deletions distinguish coordinates there. Ranking
    single deletions is still a heuristic for a MULTI-coordinate reduction:
    evaluate both complete proposed supports, including the new simplex floor,
    and use the exact-ranked proposal only when its actual objective is lower.
    """
    strategy=cfg.get('cosine_pruning_strategy','gradient_direction')
    if strategy not in {'gradient_direction','exact_objective_deletion'} or not 1<=count<len(active):
        raise ValueError('invalid cosine pruning strategy or count')
    started=time.perf_counter(); evaluations=engine.evaluations
    floor=float(cfg.get('cosine_weight_floor_mass',1e-4))
    value,g,terms=engine.evaluate(active,p,True)
    legacy=p/((1-p).clamp_min(1e-12))*((g*p).sum()-g)
    full=torch.zeros(engine.x.shape[1],device=p.device,dtype=p.dtype); full[active]=p
    def proposal(scores):
        priority=torch.full_like(full,-torch.inf); priority[active]=scores
        ranking=priority.argsort(descending=True,stable=True)
        chosen=covering_support(engine.x,ranking,count)
        if chosen is None:
            return None
        weights=full[chosen]
        weights=project_simplex(weights/weights.sum(),floor) if weights.sum()>0 else project_simplex(torch.ones_like(weights)/count,floor)
        try:
            loss,_,point=engine.evaluate(chosen,weights)
        except ValueError:
            return None
        return chosen,weights,float(loss),point['geometry_error']
    old=proposal(legacy); exact=None; deletion_points=[]
    if strategy=='exact_objective_deletion':
        scores=torch.empty_like(p)
        last_log=started
        for j in range(len(active)):
            keep=torch.arange(len(active),device=p.device)!=j
            weights=project_simplex(p[keep]/p[keep].sum(),floor)
            record={'removed_feature':int(active[j]),'remaining_features':len(active)-1}
            try:
                loss,_,point=engine.evaluate(active[keep],weights)
                scores[j]=loss-value
                record.update(valid=True,base_objective=float(loss),
                              loss_delta=float(loss-value),geometry_error=point['geometry_error'])
            except ValueError:
                # A deletion that loses a normal row's coverage must be retained.
                scores[j]=torch.inf
                record.update(valid=False,base_objective=None,loss_delta=None,geometry_error=None)
            deletion_points.append(record)
            now=time.perf_counter()
            if now-last_log>=30:
                print(f'[{label}] cosine-deletion {j+1}/{len(active)} elapsed={now-started:.1f}s',flush=True)
                last_log=now
        exact=proposal(scores)
    accepted=exact is not None and (old is None or exact[2]<old[2])
    chosen=exact if accepted else old
    diag={'strategy':strategy,'source_features':len(active),'target_features':count,
          'active_before':active.cpu().tolist(),'base_objective_before':float(value),
          'geometry_before':terms['geometry_error'],
          'first_order_score_span':float(legacy.max()-legacy.min()),
          'fixed_support_first_order_residual':float((p-project_simplex(p-g,floor)).norm()),
          'finite_deletions':deletion_points,'exact_proposal_accepted':accepted,
          'legacy_proposal':None if old is None else {'active':old[0].cpu().tolist(),'base_objective':old[2],'geometry_error':old[3]},
          'exact_proposal':None if exact is None else {'active':exact[0].cpu().tolist(),'base_objective':exact[2],'geometry_error':exact[3]},
          'selected_proposal':'exact_objective_deletion' if accepted else 'gradient_direction' if old is not None else 'invalid',
          'selected_initial_objective':None if chosen is None else chosen[2],
          'selected_initial_geometry':None if chosen is None else chosen[3],
          'objective_evaluations':engine.evaluations-evaluations,'seconds':time.perf_counter()-started}
    print(f'[{label}] cosine-prune {len(active)}->{count} proposal={diag["selected_proposal"]} '
          f'initial_loss={diag["selected_initial_objective"]} elapsed={diag["seconds"]:.1f}s',flush=True)
    return (None,None,diag) if chosen is None else (chosen[0],chosen[1],diag)


def exchange_simplex(engine, active, p, cfg, label):
    maximum=int(cfg.get('cosine_exchange_max_steps',cfg.get('support_exchange_max_steps',8)))
    proposals=int(cfg.get('cosine_exchange_candidates',8))
    gain=float(cfg.get('support_exchange_min_improvement',1e-8))
    history=[]; total_iterations=0
    value,_,terms=engine.evaluate(active,p)
    for iteration in range(maximum):
        omitted=torch.ones(engine.x.shape[1],device=p.device,dtype=torch.bool); omitted[active]=False
        outside=torch.where(omitted)[0]
        if not len(outside):
            break
        _,g,_=engine.evaluate(active,p,True,True)
        predicted=p[:,None]*(g[outside][None,:]-g[active][:,None])
        order=torch.argsort(predicted.flatten(),stable=True)[:proposals]
        best=None
        for flat in order.cpu().tolist():
            remove,add=divmod(flat,len(outside))
            candidate=active.clone(); candidate[remove]=outside[add]
            sort=candidate.argsort(stable=True); candidate=candidate[sort]; weights=p[sort]
            try:
                new_value,_,new_terms=engine.evaluate(candidate,weights)
            except ValueError:
                continue
            if new_value<value-gain and (best is None or new_value<best[0]):
                best=(new_value,candidate,weights,new_terms,int(active[remove]),int(outside[add]))
        if best is None:
            break
        before=float(value)
        value,active,p,terms,removed,added=best
        p,terms,diag=refit_simplex(engine,active,p,cfg,label)
        value,_,terms=engine.evaluate(active,p)
        total_iterations+=diag['solver_iterations']
        history.append({'removed':removed,'added':added,'before':before,'after':float(value),
                        'refit':diag})
    return active,p,terms,history,total_iterations


@torch.inference_mode()
@precise_matmul()
def fit_cosine_uvarfs(x, variability, cfg, gram_control, label='uvarfs:cosine'):
    t0=time.time()
    m=x.shape[1]; minf=min(int(cfg.get('min_features',32)),m); maxf=min(int(cfg.get('max_features',256)),m)
    step=int(cfg.get('cosine_support_step',32)); tol=float(cfg.get('geometry_tolerance',.05))
    floor=float(cfg.get('cosine_weight_floor_mass',1e-4))
    lambdas=[float(v) for v in cfg['lambda_grid']]
    if not 1<=minf<=maxf or step<1 or not 0<floor<1 or not 0<=tol<=1 or not lambdas or any(v<0 or not torch.isfinite(torch.tensor(v)) for v in lambdas):
        raise ValueError('invalid cosine U-VaRFS budget/path')
    numeric=[float(cfg.get('cosine_initial_step',.05)),float(cfg.get('tol',1e-5)),float(cfg.get('support_exchange_min_improvement',1e-8)),float(cfg.get('cosine_polish_ftol',1e-12))]
    if (not torch.isfinite(torch.tensor(numeric)).all() or numeric[1]<=0 or numeric[2]<0 or numeric[3]<=0 or
        int(cfg.get('cosine_polish_max_iter',200))<0 or
        int(cfg.get('cosine_refit_max_iter',cfg.get('max_iter',2000)))<1 or
        int(cfg.get('cosine_line_search_steps',20))<1 or float(cfg.get('cosine_initial_step',.05))<=0 or
        int(cfg.get('cosine_exchange_max_steps',cfg.get('support_exchange_max_steps',8)))<0 or
        int(cfg.get('cosine_exchange_candidates',8))<1):
        raise ValueError('invalid cosine solver limits')
    pruning_strategy=cfg.get('cosine_pruning_strategy','gradient_direction')
    if pruning_strategy not in {'gradient_direction','exact_objective_deletion'}:
        raise ValueError('invalid cosine pruning strategy')
    engine=CosineObjective(x,variability,float(cfg.get('beta',.002)),int(cfg.get('cosine_geometry_chunk',512)))
    old=torch.as_tensor(gram_control['budgeted_weights'],device=x.device,dtype=engine.x.dtype)
    if old.shape!=(m,) or not torch.isfinite(old).all() or (old<0).any():
        raise ValueError('invalid original Gram initialization')
    sizes=sorted(set(range(minf,maxf+1,step))|{maxf,min(maxf,max(minf,len(gram_control['active'])))},reverse=True)
    starts=[('original_gram',old.argsort(descending=True,stable=True),old),
            ('normal_energy',engine.x.square().sum(0).argsort(descending=True,stable=True),torch.ones_like(old))]
    pool=[]; total_iterations=0; invalid_zero_rows=0; pruning_events=[]

    def optimize_candidate(active,p,source,pruning_event_id=None):
        nonlocal total_iterations
        count=len(active)
        p,terms,diag=refit_simplex(engine,active,p,cfg,f'{label}:{source}:K{count}')
        total_iterations+=diag['solver_iterations']
        active,p,terms,exchanges,used=exchange_simplex(engine,active,p,cfg,f'{label}:{source}:K{count}:exchange')
        total_iterations+=used
        value,g,terms=engine.evaluate(active,p,True)
        residual=float((p-project_simplex(p-g,floor)).norm())
        weights=torch.zeros_like(old); weights[active]=p
        pool.append({'candidate_id':len(pool),'source':source,'pruning_event_id':pruning_event_id,'retained_features':len(active),
            'nonzero_features':int((p>0).sum()),'active':active.cpu().numpy(),
            'weights':weights.cpu().numpy(),'base_objective':float(value),**terms,
            'fixed_support_first_order_residual':residual,'fixed_support_converged':residual<=float(cfg.get('tol',1e-5)),
            'solver_status':diag['solver_status'] if not exchanges else 'support_exchange_completed',
            'relative_change':diag['relative_change'],'relative_change_scope':'initial_refit_before_exchange',
            'simplex_residual':float((p.double().sum()-1).abs()),
            'minimum_relative_weight':float(p.min()),'effective_weight_dimension':float(1/p.double().square().sum()),
            'zero_norm_rows':0,'refit_descent_history':diag['descent_history'],
            'initial_refit':{k:v for k,v in diag.items() if k not in {'descent_history','relative_change'}},
            'exchange_history':exchanges})
        print(f'[{label}] candidate={len(pool)-1} start={source} K={len(active)} '
              f'cosine_error={terms["geometry_error"]:.4f} base_loss={float(value):.6g}',flush=True)
        return active,p

    # Include the EXACT selected v11 support as an initialization, even when it
    # differs from the ranked/pruned trajectory starting at max_features.
    exact=torch.as_tensor(gram_control['active'],device=x.device,dtype=torch.long).sort().values
    if (len(exact.unique())!=len(exact) or not minf<=len(exact)<=maxf or
        (exact<0).any() or (exact>=m).any()):
        raise ValueError('invalid selected original Gram support')
    if bool((engine.x[:,exact].square().sum(1)>0).all()):
        initial=old[exact]
        initial=initial/initial.sum() if initial.sum()>0 else torch.ones_like(initial)/len(initial)
        optimize_candidate(exact,project_simplex(initial,floor),'original_gram_selected')
    else:
        invalid_zero_rows+=1
    for source,ranking,initial in starts:
        active=None; p=None
        for count in sizes:
            if active is None:
                active=covering_support(engine.x,ranking,count)
                if active is None:
                    invalid_zero_rows+=1
                    continue
                p=project_simplex(initial[active],floor)
                # Normalize the original relative weights rather than their
                # absolute box scale before imposing the simplex floor.
                if initial[active].sum()>0:
                    p=project_simplex(initial[active]/initial[active].sum(),floor)
            elif len(active)>count:
                parent=len(pool)-1
                active,p,pruning=prune_cosine_support(engine,active,p,count,cfg,f'{label}:{source}:prune')
                pruning.update(event_id=len(pruning_events),parent_candidate_id=parent,source=source)
                pruning_events.append(pruning)
                if active is None:
                    invalid_zero_rows+=1
                    continue
                active,p=optimize_candidate(active,p,source,pruning['event_id'])
                continue
            active,p=optimize_candidate(active,p,source)
    if not pool:
        raise RuntimeError('No valid cosine U-VaRFS candidate: generated supports contain zero rows; no detector fallback was used')
    path=[]
    for lam in lambdas:
        best=min(pool,key=lambda c:(c['base_objective']+lam*c['retained_features'],c['retained_features'],c['geometry_error'],c['candidate_id']))
        path.append({k:v for k,v in best.items() if k not in {'weights','active','refit_descent_history','initial_refit','exchange_history'}})
        path[-1].update(lambda_value=lam,**{'lambda':lam},sparsity_term=lam*best['retained_features'],
            objective=best['base_objective']+lam*best['retained_features'],
            feasible=best['geometry_error']<=tol and best['nonzero_features']>=minf,
            solver_converged=best['fixed_support_converged'])
    feasible=[point for point in path if point['feasible']]
    point=min(feasible,key=lambda c:(c['retained_features'],c['geometry_error'])) if feasible else min(path,key=lambda c:c['geometry_error'])
    chosen=pool[point['candidate_id']]; weights=chosen['weights']; active=chosen['active']
    refits=[c['initial_refit'] for c in pool]+[e['refit'] for c in pool for e in c['exchange_history']]
    polishes=[r['polish'] for r in refits]
    result={**point,'active':active,'weights':weights,'budgeted_weights':weights,
        'original_gram_selected_dimension':len(gram_control['active']),
        'original_gram_selected_active':exact.cpu().tolist(),
        'scales':weights[active]**.5,'lambda_path':path,'candidate_pool':pool,
        'pruning_events':pruning_events,
        'pruning_summary':{'events':len(pruning_events),
            'exact_proposals_accepted':sum(e['exact_proposal_accepted'] for e in pruning_events),
            'objective_evaluations':sum(e['objective_evaluations'] for e in pruning_events),
            'seconds':sum(e['seconds'] for e in pruning_events)},
        'objective_definition':OBJECTIVE,'geometry_metric':'actual_cosine_gram_relative_frobenius',
        'sparsity_strategy':'cosine_simplex_support_search',
        'objective_certificate_scope':'fixed_support_first_order_only_nonconvex',
        'lambda_optimization_scope':'finite_generated_candidate_pool',
        'global_optimality_claimed':False,'projected_residual':point['fixed_support_first_order_residual'],
        'solver_iterations':total_iterations,
        'polish_summary':{'attempts':sum(d['attempted'] for d in polishes),
            'accepted':sum(d['accepted'] for d in polishes),
            'iterations':sum(d['iterations'] for d in polishes),
            'objective_evaluations':sum(d['objective_evaluations'] for d in polishes),
            'seconds':sum(d['seconds'] for d in polishes)},
        'fit_seconds':time.time()-t0,'solver_matmul_precision':'highest',
        'selection_reason':'sparsest_feasible' if point['feasible'] else 'minimum_geometry_error_fallback',
        'lambda_at_grid_boundary':point['lambda'] in (min(lambdas),max(lambdas)),
        'solver_manifest':{'support_sizes':sizes,'initializations':['original_gram_selected','original_gram','normal_energy'],
            'invalid_zero_row_candidates':invalid_zero_rows,
            'weight_floor_mass':floor,'geometry_rows':len(x),'geometry_chunk':engine.chunk,
            'objective_evaluations':engine.evaluations,'refit_max_iter':int(cfg.get('cosine_refit_max_iter',cfg.get('max_iter',2000))),
            'pruning_strategy':pruning_strategy,
            'pruning_scope':'one_coordinate_finite_deletion_proposal_then_joint_objective_guard',
            'pruning_acceptance':'strict_same_objective_improvement_over_legacy_proposal',
            'polish_method':'slsqp_fixed_support',
            'polish_max_iter':int(cfg.get('cosine_polish_max_iter',200)),
            'polish_ftol':float(cfg.get('cosine_polish_ftol',1e-12)),
            'polish_acceptance':'projected_endpoint_strict_same_objective_descent',
            'line_search_steps':int(cfg.get('cosine_line_search_steps',20)),
            'initial_step':float(cfg.get('cosine_initial_step',.05)),
            'tolerance':float(cfg.get('tol',1e-5)),
            'exchange_min_improvement':float(cfg.get('support_exchange_min_improvement',1e-8)),
            'exchange_max_steps':int(cfg.get('cosine_exchange_max_steps',cfg.get('support_exchange_max_steps',8))),
            'exchange_candidates':int(cfg.get('cosine_exchange_candidates',8)),
            'stability_reference':'uniform full-input relative weights',
            'zero_row_rule':'candidate invalid; never normalized with epsilon'}}
    print(f'[{label}] done: cosine/simplex K={len(active)}/{m} lambda={point["lambda"]:g} '
          f'geometry={point["geometry_error"]:.4f} feasible={point["feasible"]} elapsed={result["fit_seconds"]:.1f}s',flush=True)
    return result
