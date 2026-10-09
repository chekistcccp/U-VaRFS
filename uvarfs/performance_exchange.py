"""Fixed-K support proposals are judged after capped weight refitting.

Only normal data and the existing CosineObjective are used. All attempted
refits, including rejected proposals, are recorded in the solve cost.
"""
from __future__ import annotations
import time
import torch
from .cosine_uvarfs import refit_simplex


def exchange_after_refit(engine, active, p, cfg, label, project, cap):
    maximum=int(cfg.get('performance_exchange_max_steps',8))
    proposals=int(cfg.get('performance_exchange_candidates',8))
    gain=float(cfg.get('support_exchange_min_improvement',1e-8))
    tolerance=float(cfg.get('geometry_tolerance',.05))
    if maximum<0 or proposals<1:
        raise ValueError('invalid performance exchange limits')
    started=time.perf_counter();evaluations=engine.evaluations
    attempts=[];steps=[];iterations=0
    value,_,terms=engine.evaluate(active,p)
    initial_value=float(value);initial_geometry=terms['geometry_error']
    termination='disabled' if maximum==0 else 'iteration_limit'
    for iteration in range(maximum):
        mask=torch.ones(engine.x.shape[1],device=p.device,dtype=torch.bool);mask[active]=False
        outside=torch.where(mask)[0]
        if not len(outside):
            termination='full_support';break
        _,gradient,_=engine.evaluate(active,p,True,True)
        predicted=p[:,None]*(gradient[outside][None,:]-gradient[active][:,None])
        order=predicted.flatten().argsort(stable=True)[:proposals]
        best=None
        for rank,flat in enumerate(order.cpu().tolist()):
            remove,add=divmod(flat,len(outside))
            candidate=active.clone();candidate[remove]=outside[add]
            sort=candidate.argsort(stable=True);candidate=candidate[sort];weights=p[sort]
            record={'attempt_id':len(attempts),'step':iteration,'proposal_rank':rank,
                'removed':int(active[remove]),'added':int(outside[add]),
                'before':float(value),'geometry_before':terms['geometry_error'],
                'initial_valid':False,'before_refit':None,'after_refit':None,
                'geometry_after':None,'eligible':False,'accepted':False,'refit':None}
            try:
                trial,_,_=engine.evaluate(candidate,weights)
            except ValueError:
                attempts.append(record);continue
            # A finite swap can initially raise the loss even when its optimized
            # support is better. Do not apply the old pre-refit descent filter.
            weights,new_terms,refit=refit_simplex(engine,candidate,weights,cfg,
                f'{label}:step{iteration}:proposal{rank}',project,cap)
            after,_,new_terms=engine.evaluate(candidate,weights)
            iterations+=refit['solver_iterations']
            feasible_guard=terms['geometry_error']>tolerance or new_terms['geometry_error']<=tolerance
            eligible=float(after)<float(value)-gain and feasible_guard
            record.update(initial_valid=True,before_refit=float(trial),after_refit=float(after),
                geometry_after=new_terms['geometry_error'],eligible=bool(eligible),refit=refit)
            attempts.append(record)
            if eligible and (best is None or float(after)<best[0]):
                best=(float(after),candidate,weights,new_terms,record['attempt_id'])
        if best is None:
            termination='no_improving_refitted_proposal';break
        after,active,p,new_terms,selected=best
        attempts[selected]['accepted']=True
        steps.append({'step':iteration,'attempt_id':selected,'before':float(value),'after':after,
            'geometry_before':terms['geometry_error'],'geometry_after':new_terms['geometry_error'],
            'active':active.cpu().tolist(),'relative_weights':p.cpu().tolist()})
        value=after;terms=new_terms
        print(f'[{label}] refitted exchange {iteration+1}/{maximum} loss={after:.6g} '
              f'geometry={terms["geometry_error"]:.4f}',flush=True)
    return active,p,{'attempts':attempts,'steps':steps,'initial_objective':initial_value,
        'initial_geometry':initial_geometry,'final_objective':float(value),
        'final_geometry':terms['geometry_error'],'termination':termination,
        'solver_iterations':iterations,'objective_evaluations':engine.evaluations-evaluations,
        'seconds':time.perf_counter()-started}
