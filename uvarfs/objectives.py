"""Objective identity and paired protocol paths; no fitting/runtime imports."""
from .representation import normalization_modes, normalization_prefix

QUADRATIC = 'quadratic_gram'
COSINE = 'cosine_simplex_cardinality'
PERFORMANCE = 'cosine_simplex_fixed_budget'


def objective_modes(cfg):
    settings=cfg.get('uvarfs',{})
    primary=settings.get('objective',QUADRATIC)
    other=settings.get('ablation_objective')
    if primary not in {QUADRATIC,COSINE,PERFORMANCE} or other not in {None,QUADRATIC,COSINE,PERFORMANCE} or primary==other:
        raise ValueError('invalid or duplicate U-VaRFS objective branches')
    return primary,[primary]+([other] if other is not None else [])


def experiment_branches(cfg):
    primary_input,inputs=normalization_modes(cfg)
    primary_objective,objectives=objective_modes(cfg)
    branches=[]
    for mode in inputs:
        input_prefix='' if mode==primary_input else normalization_prefix(mode)
        for objective in objectives:
            objective_prefix='' if objective==primary_objective else ('gram_' if objective==QUADRATIC else 'cosine_' if objective==COSINE else 'performance_')
            branches.append({'layer_normalization':mode,'objective':objective,
                'prefix':input_prefix+objective_prefix,
                'folder':'/'.join(part for part in [input_prefix.rstrip('_'),objective_prefix.rstrip('_')] if part)})
    return branches
