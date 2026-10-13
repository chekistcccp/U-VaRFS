"""Objective identity and paired protocol paths; no fitting/runtime imports."""
from .representation import normalization_modes, normalization_prefix

QUADRATIC = 'quadratic_gram'
COSINE = 'cosine_simplex_cardinality'
PERFORMANCE = 'cosine_simplex_fixed_budget'
LOCAL = 'normal_local_fixed_budget'
MATCHING = 'normal_matching_distribution_fixed_budget'


def methods_for_objective(cfg, objective):
    restrictions = {'asls_no_variability_uvarfs':{LOCAL,MATCHING},
                    'asls_no_rank_uvarfs':{LOCAL},
                    'asls_no_score_uvarfs':{MATCHING},
                    'asls_no_distribution_uvarfs':{MATCHING}}
    return [m for m in cfg['methods'] if m not in restrictions or objective in restrictions[m]]


def objective_modes(cfg):
    settings=cfg.get('uvarfs',{})
    primary=settings.get('objective',QUADRATIC)
    other=settings.get('ablation_objective')
    controls = settings.get('control_objectives',[])
    if not isinstance(controls,list):
        raise ValueError('control_objectives must be a list')
    modes = [primary]+([other] if other is not None else [])+controls
    if any(o not in {QUADRATIC,COSINE,PERFORMANCE,LOCAL,MATCHING} for o in modes) or len(set(modes))!=len(modes):
        raise ValueError('invalid or duplicate U-VaRFS objective branches')
    return primary,modes


def experiment_branches(cfg):
    primary_input,inputs=normalization_modes(cfg)
    primary_objective,objectives=objective_modes(cfg)
    v21_control = cfg.get('uvarfs',{}).get('matching_v21_control',False)
    polish = cfg.get('uvarfs',{}).get('matching_polish_max_iter',0)
    if not isinstance(v21_control,bool) or (v21_control and (
            primary_objective!=MATCHING or isinstance(polish,bool) or not isinstance(polish,int) or polish<=0)):
        raise ValueError('v21 solver control requires a polished matching Main')
    branches=[]
    for mode in inputs:
        input_prefix='' if mode==primary_input else normalization_prefix(mode)
        for objective in objectives:
            objective_prefix='' if objective==primary_objective else {QUADRATIC:'gram_',COSINE:'cosine_',LOCAL:'local_',PERFORMANCE:'performance_',MATCHING:'matching_'}[objective]
            branches.append({'layer_normalization':mode,'objective':objective,
                'prefix':input_prefix+objective_prefix,
                'folder':'/'.join(part for part in [input_prefix.rstrip('_'),objective_prefix.rstrip('_')] if part)})
            if v21_control and objective==MATCHING:
                branches.append({'layer_normalization':mode,'objective':MATCHING,
                    'prefix':input_prefix+'v21_',
                    'folder':'/'.join(part for part in [input_prefix.rstrip('_'),'v21'] if part),
                    'matching_polish_max_iter':0,'matching_diverse_starts':False})
    return branches
