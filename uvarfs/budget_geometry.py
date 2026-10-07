"""Normal-only necessary conditions for the ORIGINAL box/feature budget."""
from __future__ import annotations

import torch


def audit_budget_geometry(H, cfg):
    """A global lower bound over every support of size at most K.

    Let b=H*1 and B=1^T H 1. By projection onto the reference Gram,
    ||K-K_w||_F/||K||_F >= 1-b^T w/B. With 0<=w<=1 and at most K
    nonzeros, b^T w cannot exceed the sum of the K largest b entries.
    This is a necessary bound, not a constructive feasible solution.
    """
    capacity=min(int(cfg.get('max_features',256)),H.shape[0])
    tolerance=float(cfg.get('geometry_tolerance',.05))
    if capacity<1 or not 0<=tolerance<=1:
        raise ValueError('invalid geometry audit budget or tolerance')
    mass=H.sum(1,dtype=torch.float64)
    total=mass.sum()
    zero=bool(total<=0)
    shares=mass/total if not zero else torch.zeros_like(mass)
    cumulative=shares.sort(descending=True,stable=True).values.cumsum(0)
    captured=float(cumulative[capacity-1]) if not zero else 1.
    bound=max(0.,1-captured)
    necessary=0 if zero else min(H.shape[0],int(torch.searchsorted(cumulative,1-tolerance))+1)
    margin=1e-6  # Reporting margin; does not change training/selection tolerance.
    return {'selection_candidate':False,'scope':'all_supports_with_at_most_max_features_and_box_weights',
            'max_features':capacity,'geometry_tolerance':tolerance,
            'geometry_error_lower_bound':bound,'max_reference_alignment_fraction':captured,
            'necessary_features_lower_bound':necessary,
            'budget_ruled_out_at_working_precision':bound>tolerance+margin,
            'numerical_reporting_margin':margin,
            'arithmetic':'float64 reductions of the fitted H; not interval arithmetic',
            'zero_reference_gram':zero,'coordinate_reference_alignment':shares.cpu().tolist()}
