"""Feasible limited-memory secant proposals for the unchanged matching loss.

Weights are optimized as mass=K*p. Every accepted step is checked against the
actual objective; the old feasible solution is retained. The diagnostic is a
current-nearest-branch projected residual, never a global/nonsmooth certificate.
"""
from __future__ import annotations
import math,time
import numpy as np
import torch
from .numerics import precise_matmul

VERSION='fixed-support-precision-v1'
SETTINGS={'evaluation_version':VERSION,'feature_budget':256,'residual_tolerance':1e-5,
          'maximum_iterations':3000,'maximum_evaluations':12000,'history_size':12,
          'line_search_steps':24,'armijo':1e-4,'curvature_relative_floor':1e-10,
          'normal_roles':{'fit_images':144,'selection_images':48,'audit_images':64,
                          'nuisance_images':96,'patches_per_image':16},
          'source_weight_policy':'continue_each_frozen_800_step_solution',
          'support_policy':'all_parent_selected_and_random_and_dense_supports_fixed',
          'global_evaluation_gate':'all_252_weighted_fits_pass_actual_projected_residual'}
RANDOM_SEEDS=[42,123,3407,2026,2027]

def weighted_names():
    names={'precision_sparse':'mechanism_sparse','precision_beta_zero':'mechanism_beta_zero',
           'precision_dense_weighted':'dense_weighted','precision_dense_beta_zero':'dense_beta_zero'}
    for seed in RANDOM_SEEDS:
        names[f'precision_random_weighted_seed{seed}']=f'random_weighted_seed{seed}'
        names[f'precision_random_beta_zero_seed{seed}']=f'random_beta_zero_seed{seed}'
    return names

def methods():return set(weighted_names())|{'previous_sparse_anchor','previous_dense_anchor','asls_raw'}

@torch.no_grad()
@precise_matmul()
def refit_fixed_support(engine,active,initial,settings=None,log=None):
    """No selection/audit/test API. Stop honestly on budget or line-search failure."""
    cfg=dict(SETTINGS if settings is None else settings)
    for key in ['maximum_iterations','maximum_evaluations','history_size','line_search_steps']:
        if type(cfg[key]) is not int or cfg[key]<1:raise ValueError('positive integer solver budget required')
    tol=float(cfg['residual_tolerance']);armijo=float(cfg['armijo']);curv=float(cfg['curvature_relative_floor'])
    if not math.isfinite(tol) or tol<=0 or not 0<armijo<1 or not 0<curv<1:
        raise ValueError('invalid precision solver tolerances')
    ids=torch.as_tensor(active,device=engine.q.device,dtype=torch.long)
    p=torch.as_tensor(initial,device=engine.q.device,dtype=torch.float64)
    if p.ndim!=1 or not torch.isfinite(p).all() or p.shape!=ids.shape or not len(p):raise ValueError('invalid initial feasible weights')
    projected=engine.project(p)
    if float((p-projected).abs().max())>1e-10:raise ValueError('initial weights violate original simplex')
    p=projected;k=len(p);started=time.perf_counter();calls=0
    def evaluate(mass):
        nonlocal calls
        if calls>=cfg['maximum_evaluations']:return None
        value,g,terms=engine.evaluate(ids,mass/k,True);calls+=1
        if not math.isfinite(value) or not torch.isfinite(g).all():raise ValueError('nonfinite actual matching loss/gradient')
        return value,g/k,terms
    def project(mass):return k*engine.project(mass/k)
    mass=k*p;value,g,terms=evaluate(mass);initial_value=value
    history=[value];pairs=[];residuals=[];rejections=0;iterations=0;reason='maximum_iterations'
    gamma=float(k*k)
    for iteration in range(cfg['maximum_iterations']+1):
        p=mass/k;residual=float((p-engine.project(p-k*g)).abs().max());residuals.append(residual)
        if residual<=tol:reason='residual_met';break
        if iteration==cfg['maximum_iterations']:break
        if calls>=cfg['maximum_evaluations']:reason='maximum_evaluations';break
        # Two-loop positive-curvature secant approximation, in mass coordinates.
        q=g.clone();alphas=[]
        for s,y,rho in reversed(pairs):
            alpha=rho*float(s@q);alphas.append(alpha);q-=alpha*y
        z=gamma*q
        for (s,y,rho),alpha in zip(pairs,reversed(alphas)):
            z+=s*(alpha-rho*float(y@z))
        directions=[project(mass-z)-mass,project(mass-gamma*g)-mass]
        accepted=None
        for direction in directions:
            slope=float(g@direction)
            if not math.isfinite(slope) or slope>=0 or float(direction.abs().max())<1e-15:continue
            step=1.
            for _ in range(cfg['line_search_steps']):
                candidate=project(mass+step*direction)
                actual_direction=candidate-mass;actual_slope=float(g@actual_direction)
                result=evaluate(candidate)
                if result is None:break
                nv,ng,nt=result
                if actual_slope<0 and nv<value and nv<=value+armijo*actual_slope:
                    accepted=(candidate,nv,ng,nt);break
                step*=.5;rejections+=1
            if accepted is not None or calls>=cfg['maximum_evaluations']:break
        if accepted is None:
            reason='maximum_evaluations' if calls>=cfg['maximum_evaluations'] else 'line_search_stalled'
            break
        candidate,nv,ng,nt=accepted;s=candidate-mass;y=ng-g;sy=float(s@y)
        if sy>curv*max(float(s.norm()*y.norm()),1e-30):
            pairs.append((s.clone(),y.clone(),1/sy));pairs=pairs[-cfg['history_size']:]
            gamma=min(1e8,max(1e-8,sy/max(float(y@y),1e-30)))
        else:pairs=[];gamma=float(k*k)
        mass,value,g,terms=candidate,nv,ng,nt;iterations+=1;history.append(value)
        if log is not None and (iterations==1 or iterations%100==0):
            log(f'[precision] K={k} iter={iterations} calls={calls} loss={value:.9g} residual={residual:.5g} seconds={time.perf_counter()-started:.1f}')
    p=mass/k
    final_residual=float((p-engine.project(p-k*g)).abs().max())
    if value>initial_value+1e-12 or abs(float(p.sum())-1)>1e-10:raise ValueError('actual loss or original feasibility not preserved')
    return p,value,terms,{'residual':final_residual,'residual_tolerance':tol,'precision_pass':final_residual<=tol,
        'termination':reason,'iterations':iterations,'objective_gradient_calls':calls,'training_history':history,
        'residual_history':residuals,'line_search_rejections':rejections,'seconds':time.perf_counter()-started,
        'initial_objective':initial_value,'variable':'mass=K*p; gradient_mass=gradient_p/K',
        'initialization':'immutable_parent_800_step_weights','solver':'feasible_limited_memory_secant_actual_armijo',
        'support_changed':False,'global_optimality_claimed':False,
        'certificate':'fixed_support_current_nearest_branch_projected_residual_only'}
