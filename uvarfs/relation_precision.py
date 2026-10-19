"""Fixed-support numerical continuation for the frozen smooth relation loss.

The registered Main and all parent artifacts remain unchanged. No labels,
selection or audit feature values are accepted by the continuation API.
"""
from __future__ import annotations
import copy
import numpy as np
import torch
from .precision_refit import refit_fixed_support
from .relation_selection import RelationObjective, clean_graph, SETTINGS as RELATION_SETTINGS
from .mechanism_validation import RANDOM_SEEDS, ROLES, artifact_digest

VERSION = 'clean-relation-fixed-support-precision-v1'
SETTINGS = dict(evaluation_version=VERSION, feature_budget=256,
    residual_tolerance=1e-5, maximum_iterations=3000, maximum_evaluations=12000,
    history_size=12, line_search_steps=24, armijo=1e-4, curvature_relative_floor=1e-10,
    normal_roles=ROLES, source_weight_policy='continue_each_frozen_800_step_solution',
    support_policy='all_relation_parent_supports_fixed_including_simple_and_same_support_beta0',
    global_evaluation_gate='all_342_weighted_fits_pass_smooth_fixed_support_projected_residual')

def weighted_names():
    result = {'precision_sparse':'relation_sparse', 'precision_beta_zero':'relation_beta_zero',
        'precision_same_support_beta_zero':'same_support_beta_zero',
        'precision_dense_weighted':'dense_weighted', 'precision_dense_beta_zero':'dense_beta_zero'}
    for seed in RANDOM_SEEDS:
        result[f'precision_random_weighted_seed{seed}']=f'random_weighted_seed{seed}'
        result[f'precision_random_beta_zero_seed{seed}']=f'random_beta_zero_seed{seed}'
    for name in ['laplacian','repeatability']:
        for suffix in ['weighted','beta_zero']:
            result[f'precision_{name}_{suffix}']=f'{name}_{suffix}'
    return result

ANCHORS = {'previous_sparse_anchor':'relation_sparse',
           'previous_dense_anchor':'dense_weighted','asls_raw':'asls_raw'}
def methods():return set(weighted_names()) | set(ANCHORS)

def frozen_parent(branch, previous, same_support=None):
    if previous=='same_support_beta_zero':
        if same_support is None:raise ValueError('frozen same-support beta0 fit required')
        return copy.deepcopy(same_support)
    if previous.startswith('random_'):return copy.deepcopy(branch['random'][previous.rsplit('seed',1)[1]])
    if previous.startswith('dense_'):return copy.deepcopy(branch['dense'])
    if previous.startswith(('laplacian_','repeatability_')):
        return copy.deepcopy(branch['baseline'][previous.split('_')[0]])
    if previous not in {'relation_sparse','relation_beta_zero'}:raise ValueError('unknown frozen parent')
    return copy.deepcopy(branch['learned'][branch['winner_index']])

def frozen_graph(x, groups, branches, seed):
    graph, info=clean_graph(x,x,groups,groups,RELATION_SETTINGS['graph'],seed)
    digest=artifact_digest(graph)
    for b in branches.values():
        if b['graph_weights_sha256']!=digest or b['graph']!=info:
            raise ValueError('immutable clean graph does not reproduce')
    if branches['nuisance']['temperature_fit_only']!=branches['beta_zero']['temperature_fit_only']:
        raise ValueError('beta branches have different frozen temperature')
    return graph,info

def continue_parents(engine, parents, settings=SETTINGS, log=print):
    result={}
    for name,parent in parents.items():
        initial=torch.tensor(parent['weights'],device=engine.q.device,dtype=torch.float64)
        value,_,_=engine.evaluate(parent['active'],initial)
        if not np.isclose(value,parent['training_objective'],atol=1e-10,rtol=1e-9):
            raise ValueError('original relation objective does not reproduce')
        uniform=torch.ones_like(initial)/len(initial)
        uniform_value,_,_=engine.evaluate(parent['active'],uniform)
        if log:log(f'[relation-precision:{name}] K={len(initial)} parent_loss={value:.9g}')
        p,v,terms,diag=refit_fixed_support(engine,parent['active'],initial,settings,log)
        verified,gradient,verified_terms=engine.evaluate(parent['active'],p,True)
        residual=float((p-engine.project(p-gradient)).abs().max())
        if (not np.isclose(verified,v,atol=1e-12,rtol=1e-10) or
            not np.isclose(residual,diag['residual'],atol=1e-12,rtol=1e-9) or
            any(not np.isclose(terms[k],verified_terms[k],atol=1e-12,rtol=1e-10) for k in terms)):
            raise ValueError('independent smooth final gradient/loss verification failed')
        diag['certificate']='smooth_fixed_graph_fixed_support_projected_residual_not_global'
        diag['parent_reproduction_calls']=1
        diag['uniform_reference_calls']=1
        diag['final_verification_calls']=1
        diag['total_actual_calls']=diag['objective_gradient_calls']+3
        result[name]=dict(active=parent['active'],weights=p.cpu().tolist(),training_objective=v,
            training_terms=terms,diagnostics=diag,parent_training_objective=value,
            parent_residual=parent['diagnostics']['residual'],
            parent_weights_sha256=artifact_digest(parent['weights']),
            uniform_training_objective=uniform_value,
            final_gradient_p=gradient.cpu().tolist(),final_verification_calls=1)
        if log:log(f'[relation-precision:{name}] done residual={residual:.7g} pass={diag["precision_pass"]}')
    return result

def audit_record(name, c, previous_spec=None):
    """Independent CPU feasibility, trace/cost and stored-gradient residual audit."""
    from .performance_uvarfs import project_bounded_simplex
    if name not in weighted_names():raise ValueError('unexpected weighted identity')
    ids=np.asarray(c['active']);p=np.asarray(c['weights'],dtype=np.float64)
    g=np.asarray(c['final_gradient_p'],dtype=np.float64);d=c['diagnostics'];k=len(p)
    if (p.ndim!=1 or not k or ids.shape!=p.shape or g.shape!=p.shape or
        not np.issubdtype(ids.dtype,np.integer) or len(np.unique(ids))!=k or (ids<0).any() or
        not np.isfinite(p).all() or not np.isfinite(g).all() or abs(p.sum()-1)>1e-10 or
        p.min()<1e-4/k-1e-12 or p.max()>min(1,4/k)+1e-12 or
        ('dense_' not in name and k!=256)):
        raise ValueError('fixed precision support/weight feasibility differs')
    pp=torch.tensor(p);gg=torch.tensor(g)
    residual=float((pp-project_bounded_simplex(pp-gg,1e-4,4.)).abs().max())
    if not np.isclose(residual,d['residual'],atol=1e-11,rtol=1e-8):
        raise ValueError('independent original-unit residual differs')
    history=d['training_history'];rh=d['residual_history']
    if (len(history)!=d['iterations']+1 or len(rh)!=len(history) or
        not np.isfinite(history).all() or not np.isfinite(rh).all() or
        any(b>=a for a,b in zip(history,history[1:])) or
        not np.isclose(history[0],c['parent_training_objective'],atol=1e-10,rtol=1e-9) or
        not np.isclose(history[-1],c['training_objective'],atol=1e-12,rtol=1e-10) or
        c['training_objective']>c['parent_training_objective']+1e-12 or
        d['iterations']>SETTINGS['maximum_iterations'] or d['iterations']<0 or
        not 1<=d['objective_gradient_calls']<=SETTINGS['maximum_evaluations'] or
        d['objective_gradient_calls']<len(history) or d['support_changed'] or
        d['total_actual_calls']!=d['objective_gradient_calls']+3 or
        d['precision_pass']!=(residual<=SETTINGS['residual_tolerance']) or
        d['certificate']!='smooth_fixed_graph_fixed_support_projected_residual_not_global'):
        raise ValueError('actual precision trace, cost or gate differs')
    if 'beta_zero' in name and c['training_terms']['variability_term']!=0:
        raise ValueError('beta0 nuisance leaked')
    if previous_spec is not None and ids.tolist()!=previous_spec['obj']['active']:
        raise ValueError('frozen support changed')
    return residual
