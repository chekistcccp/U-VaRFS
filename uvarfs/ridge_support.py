"""Fit-only ridge-leverage coordinate proposals for the unchanged matching loss.

Whitening is used to select columns, never as the inference representation.
The fixed-K greedy heuristic has no inherited randomized sampling/AD guarantee.
"""
from __future__ import annotations
import copy, hashlib, json, math, time
import torch
import torch.nn.functional as F
from .local_uvarfs import neighbor_pairs
from .matching_uvarfs import MatchingObjective
from .matching_polish import polish_fixed_support
from .normal_tail_selection import validate_frozen_pool, VALIDATION_FIELDS
from .numerics import precise_matmul

VERSION='ridge-support-continuation-v1'


def digest(pool):
    clean=[{k:v for k,v in c.items() if k not in VALIDATION_FIELDS} for c in pool]
    return hashlib.sha256(json.dumps(clean,sort_keys=True,allow_nan=False).encode()).hexdigest()


@torch.no_grad()
@precise_matmul()
def greedy_projection_coordinates(matrix,k):
    """Exact one-column reduction in ||(I-P_selected) matrix||_F^2.

    Unlike maximum residual column norm, the numerator accounts for the
    column's correlation with every remaining coordinate. No rotated feature
    is emitted. The greedy solution is not globally optimal.
    """
    if matrix.ndim!=2 or not 1<=k<=matrix.shape[1] or not torch.isfinite(matrix).all():
        raise ValueError('invalid normal projection matrix or budget')
    a=matrix.double();gram=a.T@a;initial=gram.diagonal().clamp_min(0).clone()
    used=torch.zeros(a.shape[1],device=a.device,dtype=torch.bool)
    selected=[];gains=[]
    cutoff=max(1.,float(initial.max()))*torch.finfo(torch.float64).eps*max(a.shape)*8
    for _ in range(k):
        diagonal=gram.diagonal().clamp_min(0)
        score=(gram.square().sum(0)/diagonal.clamp_min(cutoff)).masked_fill(used | (diagonal<=cutoff),-torch.inf)
        index=int(score.argmax())
        if not torch.isfinite(score[index]):
            remaining=torch.arange(a.shape[1],device=a.device)[~used]
            order=initial[~used].argsort(descending=True,stable=True)
            selected.extend(remaining[order[:k-len(selected)]].tolist());break
        selected.append(index);used[index]=True;gains.append(float(score[index]))
        column=gram[:,index].clone();gram-=column[:,None]*column[None,:]/diagonal[index]
    active=torch.tensor(selected,device=a.device,dtype=torch.long).sort().values
    if len(active)!=k or len(active.unique())!=k:raise ValueError('normal projection did not fill exact K')
    return active,{'greedy_actual_projection_gains':gains,'rank_steps':len(gains),
        'remaining_projection_energy':float(gram.diagonal().sum().clamp_min(0)),
        'coordinate_selection_only':True,'global_optimality_claimed':False,
        'rank_deficient_fill':'descending_original_column_energy'}


@torch.no_grad()
@precise_matmul()
def ridge_whiten_columns(matrix,k):
    """Z=A(A'A+lambda I)^-1/2; lambda=rank-K tail energy / K, floored.

    Z'Z=C(C+lambda I)^-1; its diagonal gives ridge column leverage.
    In rank-deficient cases a scale-relative float64 floor avoids noise blowup.
    """
    if (matrix.ndim!=2 or not len(matrix) or isinstance(k,bool) or not isinstance(k,int)
        or not 1<=k<=matrix.shape[1] or not torch.isfinite(matrix).all()):
        raise ValueError('invalid normal ridge matrix or budget')
    a=matrix.double();cov=a.T@a
    values,vectors=torch.linalg.eigh(cov)
    scale=float(values[-1].clamp_min(0))
    tolerance=scale*torch.finfo(torch.float64).eps*max(a.shape)
    if float(values[0]) < -max(tolerance,1e-30)*8:
        raise ValueError('normal covariance is numerically non-PSD')
    values=values.clamp_min(0)
    tail=float(values[:-k].sum()) if k<len(values) else 0.
    ridge=max(tail/k,tolerance,torch.finfo(torch.float64).tiny)
    # Multiplication order avoids materializing a dense inverse square root.
    z=((a@vectors)/(values+ridge).sqrt())@vectors.T
    leverage=z.square().sum(0)
    return z,{'ridge_lambda':ridge,'rank_k_tail_energy':tail,'numerical_floor':tolerance,
        'effective_dimension':float((values/(values+ridge)).sum()),
        'ridge_leverage_scores':leverage.cpu().tolist(),
        'numerical_rank':int((values>tolerance).sum()),'coordinate_selection_only':True,
        'ridge_policy':'max(rank_K_tail_energy/K,float64_scale_relative_floor)'}


@torch.no_grad()
@precise_matmul()
def ridge_starts(x,groups,paired,beta,k,rank=8):
    near,far=neighbor_pairs(x,x,groups,groups,rank=rank)
    xx=x.double();distance=(1-F.cosine_similarity(x,x[near])).clamp_min(0)
    weight=(1+distance/distance.mean().clamp_min(1e-4)).clamp_max(4)
    weight/=weight.mean()
    contrast=torch.cat([xx-xx[near],xx-xx[far]])*weight.sqrt().repeat(2)[:,None]/math.sqrt(2)
    stability=torch.zeros(x.shape[1],device=x.device,dtype=torch.float64)
    if beta and paired is not None:
        clean,pert=paired
        if (clean.ndim!=2 or pert.ndim!=3 or pert.shape[1:]!=clean.shape or
            clean.shape[1]!=x.shape[1] or not torch.isfinite(clean).all() or not torch.isfinite(pert).all()):
            raise ValueError('invalid normal nuisance inputs')
        stability=(pert.double()-clean.double()[None]).square().mean((0,1))/(xx.square().mean(0)+1e-6)
    variants=[('ridge_normal_covariance',xx-xx.mean(0)),('ridge_normal_neighbor',contrast),
              ('ridge_stable_neighbor',contrast/(1+stability).sqrt())]
    starts=[];diagnostics={}
    for name,matrix in variants:
        z,info=ridge_whiten_columns(matrix,k);active,pivot=greedy_projection_coordinates(z,k)
        starts.append((name,active));diagnostics[name]={**info,**pivot}
    return starts,diagnostics


@torch.no_grad()
@precise_matmul()
def extend_fit_pool(x,groups,paired,solution,settings,progress=print):
    """No holdout argument; no old validation fields affect generation or warm starts."""
    original=validate_frozen_pool(solution,x.shape[1]);k=solution['fixed_feature_budget']
    pool=[{key:copy.deepcopy(value) for key,value in c.items() if key not in VALIDATION_FIELDS}
          for c in solution['candidate_pool']]
    beta=float(solution['beta'])
    engine=MatchingObjective(x,x,groups,groups,solution['temperature_fit_only'],paired if beta else None,
        beta,int(settings['chunk']),solution['score_weight'],solution['distribution_weight'],
        solution['weight_floor_mass'],solution['weight_cap_factor'])
    # Repeat original TRAINING values before trusting the pool, not its holdout winner.
    max_error=0.
    for c in pool:
        value,_,terms=engine.evaluate(c['active'],c['weights'])
        error=abs(value-c['training_objective']);max_error=max(max_error,error)
        if not math.isclose(value,c['training_objective'],rel_tol=1e-8,abs_tol=1e-8):
            raise ValueError('source normal training objective does not reproduce')
        if any(not math.isclose(terms[key],v,rel_tol=1e-8,abs_tol=1e-8) for key,v in c['training_terms'].items()):
            raise ValueError('source normal training terms do not reproduce')
    starts,diagnostics=ridge_starts(x,groups,paired if beta else None,beta,k,int(settings['rank']))
    seen=set();attempts=[];begin=time.perf_counter()
    def record(active,p,name,diag=None):
        value,g,terms=engine.evaluate(active,p,True)
        pool.append({'candidate_id':len(pool),'source':name,'active':active.cpu().tolist(),
            'weights':p.cpu().tolist(),'training_objective':value,'training_terms':terms,
            'fixed_support_first_order_residual':float((p-engine.project(p-g)).abs().max()),
            'solver_iterations':0 if diag is None else diag['iterations'],'diagnostics':diag})
    for name,active in starts:
        if tuple(active.tolist()) in seen:
            attempts.append({'start':name,'duplicate_new_support':True});continue
        seen.add(tuple(active.tolist()));t=time.perf_counter()
        uniform=torch.full((k,),1/k,device=x.device,dtype=torch.float64)
        try:
            record(active,uniform,name+':uniform')
            p,_,_,diag=engine.refit(active,uniform,int(settings['refit_max_iter']),float(settings['tol']))
            record(active,p,name+':refit',diag);parent=len(pool)-1
            q,value,_,polish=polish_fixed_support(engine,active,p,int(settings['polish_max_iter']),float(settings['polish_ftol']))
            if value<pool[parent]['training_objective']-1e-8:
                polish['parent_candidate_id']=parent;record(active,q,name+':polish',polish)
            attempts.append({'start':name,'parent_candidate_id':parent,'polish':polish,
                'seconds':time.perf_counter()-t,'duplicate_new_support':False})
        except ValueError as error:
            if 'zero sparse' not in str(error):raise
            attempts.append({'start':name,'invalid_zero_rows':True,'seconds':time.perf_counter()-t})
        progress(f'[ridge] {name}: {len(pool)-len(solution["candidate_pool"])} new candidates; {time.perf_counter()-begin:.1f}s')
    if digest(pool[:len(solution['candidate_pool'])])!=original:
        raise ValueError('original v22 pool changed')
    return engine,pool,{'source_pool_sha256':original,'source_pool_count':len(solution['candidate_pool']),
        'source_training_reproduction_max_error':max_error,'ridge_diagnostics':diagnostics,
        'attempts':attempts,'generation_uses_holdout':False}


@torch.no_grad()
@precise_matmul()
def fit_ridge_extension(x,heldout,groups,heldout_groups,paired,solution,settings,progress=print):
    if set(groups.cpu().tolist())&set(heldout_groups.cpu().tolist()):
        raise ValueError('normal fit and holdout groups must be disjoint')
    before=copy.deepcopy(solution);started=time.perf_counter()
    engine,pool,info=extend_fit_pool(x,groups,paired,solution,settings,progress)
    fit_digest=digest(pool)  # Freeze before creating any holdout validator.
    validator=MatchingObjective(heldout,x,heldout_groups,groups,engine.temperature,beta=0.,chunk=engine.chunk,
        score_weight=engine.score_weight,distribution_weight=engine.distribution_weight,
        floor_mass=engine.floor_mass,cap_factor=engine.cap_factor)
    for c in pool:
        try:value,_,terms=validator.evaluate(c['active'],c['weights'])
        except ValueError as error:
            if 'zero sparse' not in str(error):raise
            c.update(validation_valid=False,validation_error=str(error));continue
        c.update(validation_valid=True,validation_objective=value,validation_terms=terms,
            selection_score=value+c['training_terms']['variability_term'])
    for c,old in zip(pool,solution['candidate_pool']):
        if bool(c['validation_valid'])!=bool(old['validation_valid']):raise ValueError('source normal validity differs')
        if c['validation_valid'] and not math.isclose(c['selection_score'],old['selection_score'],rel_tol=1e-8,abs_tol=1e-8):
            raise ValueError('source normal holdout loss does not reproduce')
    valid=[c for c in pool if c['validation_valid']]
    best=min(valid,key=lambda c:(c['selection_score'],c['candidate_id']))
    new=[c for c in valid if c['candidate_id']>=info['source_pool_count']]
    ridge_only=min(new,key=lambda c:(c['selection_score'],c['candidate_id'])) if new else best
    if solution!=before:raise ValueError('source solution was mutated')
    def representation(c):return {'active':c['active'],'scales':[v**.5 for v in c['weights']]}
    return {'extension_version':VERSION,'objective_definition':solution['objective_definition'],
        'candidate_pool':pool,'candidate_pool_fit_only_sha256':fit_digest,'source':info,
        'winner_candidate_id':best['candidate_id'],'ridge_only_candidate_id':ridge_only['candidate_id'],
        'ridge_only_fallback_to_expanded_winner':not bool(new),'beta':engine.beta,
        'fixed_feature_budget':solution['fixed_feature_budget'],'fit_seconds':time.perf_counter()-started,
        'representations':{'expanded':representation(best),'ridge_only':representation(ridge_only)},
        'weight_floor_mass':engine.floor_mass,'weight_cap_factor':engine.cap_factor,
        'support_generation_uses_holdout':False,'global_optimality_claimed':False}
