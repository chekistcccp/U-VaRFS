"""Normal-only performance-first U-VaRFS: fixed K and capped simplex.

A changed feasible set/selection policy, not an equivalent speedup.
"""
from __future__ import annotations
import time
import numpy as np
import torch
from .cosine_uvarfs import CosineObjective, covering_support, refit_simplex, exchange_simplex
from .numerics import precise_matmul
from .objectives import PERFORMANCE


def project_bounded_simplex(values, floor_mass=1e-4, cap_factor=4.):
    """Piecewise-linear root: sum(p)=1, floor/K<=p<=min(1,cap/K)."""
    if (values.ndim!=1 or not len(values) or not torch.isfinite(values).all() or
        not 0<floor_mass<1 or not np.isfinite(cap_factor) or cap_factor<1):
        raise ValueError('invalid capped simplex')
    if cap_factor==1:
        return torch.full_like(values,1/len(values))
    v=values.double(); v=v-v.max()
    lower=floor_mass/len(v); capacity=min(1.,cap_factor/len(v))-lower
    shifted=v-lower; target=1-floor_mass
    breaks=torch.cat([shifted,shifted-capacity])
    signs=torch.cat([torch.ones_like(v),-torch.ones_like(v)])
    order=breaks.argsort(descending=True,stable=True); breaks=breaks[order]
    free=signs[order].cumsum(0)
    widths=torch.cat([breaks.new_zeros(1),breaks[:-1]-breaks[1:]])
    mass=(torch.cat([free.new_zeros(1),free[:-1]])*widths).cumsum(0)
    next_mass=torch.cat([mass[1:],mass[-1:]])
    valid=(free>0)&(mass<=target+1e-12)&(next_mass>=target-1e-12)
    interval=torch.where(valid)[0][0]
    threshold=breaks[interval]-(target-mass[interval])/free[interval]
    return ((shifted-threshold).clamp(0,capacity)+lower).to(values.dtype)


@torch.inference_mode()
@precise_matmul()
def fit_performance_uvarfs(x,variability,cfg,gram_control,label='uvarfs:performance'):
    started=time.time(); m=x.shape[1]
    count=min(int(cfg.get('max_features',256)),m); minimum=min(int(cfg.get('min_features',32)),m)
    floor=float(cfg.get('cosine_weight_floor_mass',1e-4)); cap=float(cfg.get('performance_weight_cap_factor',4.))
    tolerance=float(cfg.get('geometry_tolerance',.05))
    numeric=[floor,cap,tolerance,float(cfg.get('tol',1e-5)),float(cfg.get('cosine_initial_step',.05)),
             float(cfg.get('cosine_polish_ftol',1e-12)),float(cfg.get('support_exchange_min_improvement',1e-8))]
    if (not np.isfinite(numeric).all() or not 1<=minimum<=count or not 0<floor<1 or cap<1 or
        not 0<=tolerance<=1 or min(numeric[3:6])<=0 or numeric[6]<0 or
        int(cfg.get('cosine_refit_max_iter',cfg.get('max_iter',2000)))<1 or
        int(cfg.get('cosine_polish_max_iter',200))<0 or int(cfg.get('cosine_line_search_steps',20))<1 or
        int(cfg.get('cosine_exchange_max_steps',8))<0 or int(cfg.get('cosine_exchange_candidates',8))<1):
        raise ValueError('invalid performance-first constraints')
    engine=CosineObjective(x,variability,float(cfg.get('beta',.002)),int(cfg.get('cosine_geometry_chunk',512)))
    old=torch.as_tensor(gram_control['budgeted_weights'],device=x.device,dtype=engine.x.dtype)
    exact=torch.as_tensor(gram_control['active'],device=x.device,dtype=torch.long).sort().values
    if (old.shape!=(m,) or not torch.isfinite(old).all() or (old<0).any() or
        not 1<=len(exact)<=count or len(exact.unique())!=len(exact) or (exact<0).any() or (exact>=m).any()):
        raise ValueError('invalid original Gram initialization')
    energy=engine.x.square().sum(0).argsort(descending=True,stable=True)
    omitted=torch.ones(m,device=x.device,dtype=torch.bool); omitted[exact]=False
    expanded=torch.cat([exact,energy[omitted[energy]]])
    starts=[('original_gram',old.argsort(descending=True,stable=True),old),
            ('normal_energy',energy,torch.ones_like(old)),
            ('original_gram_selected_expanded',expanded,old)]
    project=lambda weights,mass:project_bounded_simplex(weights,mass,cap)
    pool=[]; iterations=0; invalid=0
    def append(active,p,source,kind,refit=None,exchanges=None):
        value,g,terms=engine.evaluate(active,p,True)
        residual=float((p-project(p-g,floor)).norm())
        weights=torch.zeros_like(old); weights[active]=p
        pool.append({'candidate_id':len(pool),'source':source,'candidate_type':kind,
            'retained_features':count,'nonzero_features':int((p>0).sum()),
            'active':active.cpu().numpy(),'weights':weights.cpu().numpy(),
            'base_objective':float(value),**terms,'zero_norm_rows':0,
            'simplex_residual':float((p.double().sum()-1).abs()),
            'minimum_relative_weight':float(p.min()),'maximum_relative_weight':float(p.max()),
            'effective_weight_dimension':float(1/p.double().square().sum()),
            'fixed_support_first_order_residual':residual,
            'fixed_support_converged':residual<=float(cfg.get('tol',1e-5)),
            'solver_status':'unrefined_reference' if refit is None else
                'support_exchange_completed' if exchanges else refit['solver_status'],
            'relative_change':0. if refit is None else refit['relative_change'],
            'relative_change_scope':'initial_refit_before_exchange',
            'initial_refit':refit,'exchange_history':exchanges or []})
        print(f'[{label}] candidate={len(pool)-1} {source}/{kind} K={count} '
              f'geometry={terms["geometry_error"]:.4f} ESS={pool[-1]["effective_weight_dimension"]:.1f}',flush=True)
    for source,ranking,initial in starts:
        active=covering_support(engine.x,ranking,count)
        if active is None:
            invalid+=1; continue
        # Keep uniform references even if variability refinement worsens geometry.
        append(active,torch.full_like(old[active],1/count),source,'uniform_reference')
        p=initial[active]; p=p/p.sum() if p.sum()>0 else torch.ones_like(p)/count
        p=project(p,floor)
        p,_,refit=refit_simplex(engine,active,p,cfg,f'{label}:{source}',project,cap)
        iterations+=refit['solver_iterations']
        active,p,_,exchanges,used=exchange_simplex(engine,active,p,cfg,f'{label}:{source}:exchange',project,cap)
        iterations+=used; append(active,p,source,'optimized',refit,exchanges)
    if not pool:
        raise RuntimeError('No valid fixed-budget support; zero-row candidates cannot be detector fallbacks')
    eligible=[c for c in pool if c['geometry_error']<=tolerance]
    chosen=min(eligible,key=lambda c:(c['base_objective'],c['geometry_error'],c['candidate_id'])) if eligible else min(pool,key=lambda c:(c['geometry_error'],c['base_objective'],c['candidate_id']))
    point={k:v for k,v in chosen.items() if k not in {'active','weights','initial_refit','exchange_history'}}
    point.update(**{'lambda':0.},sparsity_term=0.,objective=chosen['base_objective'],
                 feasible=bool(eligible),solver_converged=chosen['fixed_support_converged'])
    refits=[c['initial_refit'] for c in pool if c['initial_refit'] is not None]+[e['refit'] for c in pool for e in c['exchange_history']]
    polishes=[r['polish'] for r in refits]
    result={**point,'active':chosen['active'],'weights':chosen['weights'],
        'budgeted_weights':chosen['weights'],'scales':chosen['weights'][chosen['active']]**.5,
        'candidate_pool':pool,'lambda_path':[point],
        'objective_definition':PERFORMANCE,'geometry_metric':'actual_cosine_gram_relative_frobenius',
        'sparsity_strategy':'fixed_budget_capped_simplex_support_search','selection_policy':'fixed_budget_feasible_objective',
        'selection_reason':'fixed_budget_best_feasible_objective' if eligible else 'minimum_geometry_error_fallback',
        'lambda_optimization_scope':'not_used_fixed_budget','lambda_at_grid_boundary':False,
        'objective_certificate_scope':'fixed_support_first_order_only_nonconvex',
        'global_optimality_claimed':False,'box_optimality_gap':None,'budget_geometry_audit':None,
        'weight_cap_factor':cap,'effective_dimension_lower_bound':count/min(cap,count),
        'original_gram_selected_dimension':len(exact),'original_gram_selected_active':exact.cpu().tolist(),
        'projected_residual':point['fixed_support_first_order_residual'],'solver_iterations':iterations,
        'fit_seconds':time.time()-started,'solver_matmul_precision':'highest',
        'polish_summary':{'attempts':sum(d['attempted'] for d in polishes),
            'accepted':sum(d['accepted'] for d in polishes),'iterations':sum(d['iterations'] for d in polishes),
            'objective_evaluations':sum(d['objective_evaluations'] for d in polishes),
            'seconds':sum(d['seconds'] for d in polishes)},
        'solver_manifest':{'fixed_features':count,'weight_floor_mass':floor,'weight_cap_factor':cap,
            'initializations':[s[0] for s in starts],'candidate_types':['uniform_reference','optimized'],
            'invalid_zero_row_candidates':invalid,'objective_evaluations':engine.evaluations,
            'geometry_rows':len(x),'geometry_chunk':engine.chunk,
            'selection_metric':'normal_feasibility_then_representation_plus_variability',
            'refit_max_iter':int(cfg.get('cosine_refit_max_iter',cfg.get('max_iter',2000))),
            'polish_max_iter':int(cfg.get('cosine_polish_max_iter',200)),
            'polish_ftol':float(cfg.get('cosine_polish_ftol',1e-12)),
            'exchange_max_steps':int(cfg.get('cosine_exchange_max_steps',8)),
            'exchange_candidates':int(cfg.get('cosine_exchange_candidates',8)),
            'exchange_min_improvement':float(cfg.get('support_exchange_min_improvement',1e-8)),
            'line_search_steps':int(cfg.get('cosine_line_search_steps',20)),
            'initial_step':float(cfg.get('cosine_initial_step',.05)),
            'tolerance':float(cfg.get('tol',1e-5)),
            'zero_row_rule':'candidate invalid; never normalized with epsilon',
            'stability_reference':'uniform full-input relative weights'}}
    print(f'[{label}] performance-first K={count}/{m} geometry={point["geometry_error"]:.4f} '
          f'feasible={point["feasible"]} cap={cap:g} ESS={point["effective_weight_dimension"]:.1f}',flush=True)
    return result
