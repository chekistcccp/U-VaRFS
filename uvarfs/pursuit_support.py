"""Fit-only projected support pursuit of the unchanged matching objective."""
from __future__ import annotations
import copy,math,time
import torch
from .matching_uvarfs import MatchingObjective
from .matching_polish import polish_fixed_support
from .normal_tail_selection import validate_frozen_pool,VALIDATION_FIELDS
from .performance_uvarfs import project_bounded_simplex
from .numerics import precise_matmul
from .ridge_support import digest
VERSION='pursuit-support-continuation-v1'

@torch.no_grad()
def project_sparse_capped_simplex(values,k,floor_mass=1e-4,cap_factor=4.):
    """Exact Euclidean projection onto exactly K positive bounded coordinates.

    Identical positive bounds imply that exchanging a selected smaller input
    with an omitted larger input never increases squared distance. Top-K then
    bounded-simplex projection is optimal for this step, including negative
    inputs. This is not a certificate for an optimum of the matching loss.
    """
    if (values.ndim!=1 or isinstance(k,bool) or not isinstance(k,int) or
        not 1<=k<=len(values) or not torch.isfinite(values).all()):
        raise ValueError('invalid sparse simplex projection')
    active=values.argsort(descending=True,stable=True)[:k].sort().values
    return active,project_bounded_simplex(values[active],floor_mass,cap_factor)


@torch.no_grad()
def first_support_breakpoint(active,p,gradient,quantile=0.):
    """Quantile of active-vs-best-outside crossing times; zero is first entry."""
    active=torch.as_tensor(active,device=gradient.device,dtype=torch.long)
    p=torch.as_tensor(p,device=gradient.device,dtype=gradient.dtype)
    if (not math.isfinite(quantile) or not 0<=quantile<=1 or gradient.ndim!=1 or p.shape!=active.shape or not len(active) or
        not torch.isfinite(gradient).all() or not torch.isfinite(p).all() or
        (p<=0).any() or len(active.unique())!=len(active) or
        active.min()<0 or active.max()>=len(gradient)):
        raise ValueError('invalid support breakpoint inputs')
    outside=torch.ones(len(gradient),device=gradient.device,dtype=torch.bool);outside[active]=False
    if not bool(outside.any()):return None
    difference=gradient[active]-gradient[outside].min();positive=difference>0
    if not bool(positive.any()):return None
    threshold=float(torch.quantile(p[positive]/difference[positive],quantile))
    return threshold if math.isfinite(threshold) and threshold>0 else None


@torch.no_grad()
@precise_matmul()
def pursue_support(engine,active,initial,settings,record,progress=print):
    """Joint proposals; actual loss and current nearest neighbors are reevaluated.

    This API has no holdout. All candidates obey original K/floor/cap. The full
    gradient is a current-neighbor branch derivative, not a smoothness guarantee.
    """
    active=torch.as_tensor(active,device=engine.q.device,dtype=torch.long)
    p=engine.project(torch.as_tensor(initial,device=engine.q.device,dtype=torch.float64))
    value,_,_=engine.evaluate(active,p);trace=[];attempts=[]
    for iteration in range(int(settings['rounds'])):
        before=value;_,gradient,_=engine.evaluate(active,p,True,True)
        threshold=first_support_breakpoint(active,p,gradient)
        if threshold is None:break
        dense=torch.zeros_like(gradient);dense[active]=p
        best=None;seen={tuple(active.tolist())}
        for quantile in settings['breakpoint_quantiles']:
            crossing=first_support_breakpoint(active,p,gradient,quantile)
            step=crossing*float(settings['breakpoint_multiplier'])
            if not math.isfinite(step):continue
            proposal=dense-step*gradient
            if not torch.isfinite(proposal).all():continue
            trial,q=project_sparse_capped_simplex(proposal,len(active),engine.floor_mass,engine.cap_factor)
            support=tuple(trial.tolist())
            if support in seen:continue
            seen.add(support);item={'round':iteration+1,'first_breakpoint':threshold,'crossing_quantile':quantile,'step':step,
                                  'changed_coordinates':len(set(active.tolist())-set(support))}
            try:
                record(trial,q,'projected',{'iterations':0,**item})
                uniform=torch.full_like(q,1/len(q));record(trial,uniform,'uniform',{'iterations':0,**item})
                if engine.evaluate(trial,uniform)[0]<engine.evaluate(trial,q)[0]:q=uniform
                q,v,_,diag=engine.refit(trial,q,int(settings['refit_max_iter']),float(settings['tol']))
                diag.update(item);candidate_id=record(trial,q,'refit',diag)
                item.update(training_objective=v,candidate_id=candidate_id)
                if v<value-1e-8 and (best is None or v<best[2]):best=(trial,q,v,candidate_id)
            except ValueError as error:
                if 'zero sparse' not in str(error):raise
                item['invalid_zero_rows']=True
            attempts.append(item)
        if best is None:break
        active,p,value,candidate_id=best
        trace.append({'round':iteration+1,'before':before,'after':value,'candidate_id':candidate_id})
        progress(f'[pursuit] round={iteration+1} objective={value:.9g}; actual decrease={before-value:.3g}')
    return active,p,{'accepted_trace':trace,'attempts':attempts}


@torch.no_grad()
@precise_matmul()
def extend_fit_pool(x,groups,paired,solution,settings,progress=print):
    """Start selection, every update and final polish use training data only."""
    for key in ['starts','rounds','chunk','refit_max_iter','polish_max_iter']:
        if isinstance(settings[key],bool) or not isinstance(settings[key],int) or settings[key]<1:
            raise ValueError('invalid pursuit solver budget')
    if (not math.isfinite(settings['tol']) or settings['tol']<=0 or
        not math.isfinite(settings['polish_ftol']) or settings['polish_ftol']<=0 or
        not math.isfinite(settings['breakpoint_multiplier']) or settings['breakpoint_multiplier']<=1 or
        not settings['breakpoint_quantiles'] or
        any(not math.isfinite(v) or not 0<=v<=1 for v in settings['breakpoint_quantiles'])):
        raise ValueError('invalid pursuit step policy')
    original=validate_frozen_pool(solution,x.shape[1]);k=solution['fixed_feature_budget']
    pool=[{key:copy.deepcopy(v) for key,v in c.items() if key not in VALIDATION_FIELDS}
          for c in solution['candidate_pool']]
    beta=float(solution['beta'])
    engine=MatchingObjective(x,x,groups,groups,solution['temperature_fit_only'],paired if beta else None,
        beta,int(settings['chunk']),solution['score_weight'],solution['distribution_weight'],
        solution['weight_floor_mass'],solution['weight_cap_factor'])
    max_error=0.
    for c in pool:
        value,_,terms=engine.evaluate(c['active'],c['weights']);error=abs(value-c['training_objective'])
        max_error=max(max_error,error)
        if not math.isclose(value,c['training_objective'],rel_tol=1e-8,abs_tol=1e-8):
            raise ValueError('source normal training objective does not reproduce')
        if any(not math.isclose(terms[key],v,rel_tol=1e-8,abs_tol=1e-8) for key,v in c['training_terms'].items()):
            raise ValueError('source normal training terms do not reproduce')
    unique={}
    for c in pool:
        key=tuple(c['active'])
        if key not in unique or (c['training_objective'],c['candidate_id'])<(unique[key]['training_objective'],unique[key]['candidate_id']):unique[key]=c
    starts=sorted(unique.values(),key=lambda c:(c['training_objective'],c['candidate_id']))[:int(settings['starts'])]
    attempts=[];started=time.perf_counter()
    def record(active,p,name,diag=None):
        value,g,terms=engine.evaluate(active,p,True)
        pool.append({'candidate_id':len(pool),'source':name,'active':active.cpu().tolist(),
            'weights':p.cpu().tolist(),'training_objective':value,'training_terms':terms,
            'fixed_support_first_order_residual':float((p-engine.project(p-g)).abs().max()),
            'solver_iterations':0 if diag is None else diag['iterations'],'diagnostics':diag})
        return len(pool)-1
    for parent in starts:
        start=time.perf_counter();active=torch.tensor(parent['active'],device=x.device)
        initial=torch.tensor(parent['weights'],device=x.device,dtype=torch.float64)
        prefix=f'pursuit_from_fit_candidate{parent["candidate_id"]}'
        rec=lambda a,p,n,d=None:record(a,p,prefix+':'+n,d)
        p,value,_,polish=polish_fixed_support(engine,active,initial,int(settings['polish_max_iter']),float(settings['polish_ftol']))
        if value<parent['training_objective']-1e-8:rec(active,p,'initial_polish',polish)
        active,p,trace=pursue_support(engine,active,p,settings,rec,progress)
        final_id=rec(active,p,'continued')
        q,value,_,final_polish=polish_fixed_support(engine,active,p,int(settings['polish_max_iter']),float(settings['polish_ftol']))
        if value<pool[final_id]['training_objective']-1e-8:rec(active,q,'final_polish',final_polish)
        attempts.append({'parent_candidate_id':parent['candidate_id'],'initial_polish':polish,
            'pursuit':trace,'final_polish':final_polish,'seconds':time.perf_counter()-start})
        progress(f'[pursuit] parent={parent["candidate_id"]} new={len(pool)-len(solution["candidate_pool"])} elapsed={time.perf_counter()-started:.1f}s')
    if digest(pool[:len(solution['candidate_pool'])])!=original:raise ValueError('original v22 pool changed')
    return engine,pool,{'source_pool_sha256':original,'source_pool_count':len(solution['candidate_pool']),
       'source_training_reproduction_max_error':max_error,'start_policy':'lowest_fit_loss_unique_supports',
       'attempts':attempts,'generation_uses_holdout':False,'settings':copy.deepcopy(settings)}

@torch.no_grad()
@precise_matmul()
def fit_pursuit_extension(x,heldout,groups,heldout_groups,paired,solution,settings,progress=print):
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
    pursuit_only=min(new,key=lambda c:(c['selection_score'],c['candidate_id'])) if new else best
    if solution!=before:raise ValueError('source solution was mutated')
    def representation(c):return {'active':c['active'],'scales':[v**.5 for v in c['weights']]}
    return {'extension_version':VERSION,'objective_definition':solution['objective_definition'],
        'candidate_pool':pool,'candidate_pool_fit_only_sha256':fit_digest,'source':info,
        'winner_candidate_id':best['candidate_id'],'pursuit_only_candidate_id':pursuit_only['candidate_id'],
        'pursuit_only_fallback_to_expanded_winner':not bool(new),'beta':engine.beta,
        'fixed_feature_budget':solution['fixed_feature_budget'],'fit_seconds':time.perf_counter()-started,
        'representations':{'expanded':representation(best),'pursuit_only':representation(pursuit_only)},
        'weight_floor_mass':engine.floor_mass,'weight_cap_factor':engine.cap_factor,
        'support_generation_uses_holdout':False,'global_optimality_claimed':False}
