"""Fit-only numerical polish of the unchanged normal matching objective.

SLSQP operates on mass=K*p to keep its variables near one. Bounds, the
equality and the analytic gradient are transformed exactly. Only independently
projected, feasible points that improve the actual objective can replace the
initial point. Optimizer success is not a nonsmooth/global certificate.
"""
from __future__ import annotations

import math
import numpy as np
import torch


def polish_fixed_support(engine, active, initial, maximum=200, ftol=1e-12):
    if isinstance(maximum,bool) or not isinstance(maximum, int) or maximum < 1 or not math.isfinite(ftol) or ftol <= 0:
        raise ValueError('invalid normal matching polish budget')
    active = torch.as_tensor(active, device=engine.q.device, dtype=torch.long)
    initial = engine.project(torch.as_tensor(initial, device=engine.q.device, dtype=torch.float64))
    initial_value, _, _ = engine.evaluate(active, initial)
    k = len(active)
    best, best_value = initial.clone(), initial_value
    history, calls = [initial_value], 0

    def objective(mass):
        nonlocal best, best_value, calls
        p = torch.as_tensor(mass/k, device=engine.q.device, dtype=torch.float64)
        value, gradient, _ = engine.evaluate(active, p, True)
        calls += 1
        if (abs(float(p.sum())-1) <= 1e-8 and
                float(p.min()) >= engine.floor_mass/k-1e-10 and
                float(p.max()) <= min(1., engine.cap_factor/k)+1e-10):
            feasible = engine.project(p)
            fv, _, _ = engine.evaluate(active, feasible)
            if fv < best_value-1e-12:
                best, best_value = feasible.clone(), fv
                history.append(fv)
        return value, (gradient/k).cpu().numpy()

    if k == 1 or engine.floor_mass == 1 or engine.cap_factor == 1:
        success, status, message, iterations = True, 0, 'unique feasible weights', 0
    else:
        from scipy.optimize import minimize
        result = minimize(objective, (initial*k).cpu().numpy(), jac=True, method='SLSQP',
            bounds=[(engine.floor_mass, min(float(k), engine.cap_factor))]*k,
            constraints={'type':'eq', 'fun':lambda mass:mass.sum()-k,
                         'jac':lambda mass:np.ones(k)},
            options={'maxiter':maximum, 'ftol':ftol, 'disp':False})
        if np.isfinite(result.x).all():
            final = engine.project(torch.as_tensor(result.x/k, device=engine.q.device, dtype=torch.float64))
            value, _, _ = engine.evaluate(active, final)
            if value < best_value-1e-12:
                best, best_value = final, value
                history.append(value)
        success, status, message = bool(result.success), int(result.status), str(result.message)
        iterations = int(result.nit)
    value, gradient, terms = engine.evaluate(active, best, True)
    residual = float((best-engine.project(best-gradient)).abs().max())
    if not math.isfinite(value) or value > initial_value+1e-10:
        raise ValueError('normal matching polish failed to preserve initial objective')
    return best, value, terms, {'residual':residual, 'iterations':iterations,
        'training_history':history, 'objective_gradient_calls':calls,
        'scipy_success':success, 'scipy_status':status, 'scipy_message':message,
        'scaled_variable':'mass=K*p; gradient_mass=gradient_p/K',
        'maximum':maximum, 'ftol':ftol,
        'certificate':'fixed_support_current_nearest_branch_projected_residual_only'}
