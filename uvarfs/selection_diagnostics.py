"""Read-only scope of cosine feasibility; never used by the solver/selection."""
from __future__ import annotations

from .objectives import COSINE, PERFORMANCE


def cosine_feasibility_diagnostics(result, cfg):
    """Distinguish generated candidates from prescribed lambda-path minima.

    Absence in either finite set is not a certificate covering every support.
    Callers audit the underlying candidates/path before trusting these fields.
    """
    if result.get('objective_definition') not in {COSINE,PERFORMANCE}:
        raise ValueError('cosine feasibility diagnostics require the cosine objective')
    dimension=len(result['weights'])
    minimum=min(int(cfg['min_features']),dimension)
    maximum=min(int(cfg['max_features']),dimension)
    tolerance=float(cfg['geometry_tolerance'])
    def feasible(point):
        return (minimum <= point['retained_features'] <= maximum and
                point['nonzero_features'] >= minimum and
                point['geometry_error'] <= tolerance and point.get('zero_norm_rows',0)==0)
    pool=result['candidate_pool']; path=result['lambda_path']
    generated=[point for point in pool if feasible(point)]
    selected=[point for point in path if feasible(point)]
    performance=result['objective_definition']==PERFORMANCE
    reason=('feasible_fixed_budget_candidate' if performance and generated else
            'feasible_lambda_path_point' if selected else
            'feasible_generated_candidates_outside_lambda_path' if generated else
            'no_feasible_generated_candidate')
    return {'selection_candidate':False,'feasibility_scope':'fixed_budget_generated_pool' if performance else 'prescribed_lambda_path',
            'generated_feasible_candidates':len(generated),
            'lambda_path_feasible_points':None if performance else len(selected),
            'generated_min_geometry_error':min((p['geometry_error'] for p in pool),default=None),
            'generated_sparsest_feasible_dimension':min((p['retained_features'] for p in generated),default=None),
            'feasibility_diagnosis':reason,'global_budget_infeasibility_claimed':False}
