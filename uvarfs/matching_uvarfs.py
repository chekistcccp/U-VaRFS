"""Normal-reference matching preservation; no anomaly labels or test inputs.

The student recomputes its own nearest reference at every evaluation. The full
reference softmax includes all other-image normal patches, not teacher edges.
Support proposals and temperature are fitted before constructing the validator.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
import numpy as np
import torch
import torch.nn.functional as F

from .numerics import precise_matmul
from .local_uvarfs import LocalObjective, neighbor_pairs
from .performance_uvarfs import project_bounded_simplex

MATCHING = 'normal_matching_distribution_fixed_budget'


def _features(query, reference, query_groups, reference_groups):
    if query.ndim != 2 or reference.ndim != 2 or not len(query) or not len(reference):
        raise ValueError('normal matching requires nonempty feature matrices')
    if query.shape[1] != reference.shape[1] or query.device != reference.device:
        raise ValueError('normal matching dimensions and devices must agree')
    if not torch.isfinite(query).all() or not torch.isfinite(reference).all():
        raise ValueError('nonfinite normal matching features')
    if bool((query.double().norm(dim=1) <= 1e-12).any() or
            (reference.double().norm(dim=1) <= 1e-12).any()):
        raise ValueError('zero dense rows cannot define a normal cosine teacher')
    qg = torch.as_tensor(query_groups, device=query.device)
    rg = torch.as_tensor(reference_groups, device=query.device)
    if qg.ndim != 1 or rg.ndim != 1 or len(qg) != len(query) or len(rg) != len(reference):
        raise ValueError('invalid normal image groups')
    if not torch.isfinite(qg).all() or not torch.isfinite(rg).all():
        raise ValueError('nonfinite normal image groups')
    return qg, rg


@torch.no_grad()
@precise_matmul()
def normal_temperature(x, groups, rank=8, chunk=256):
    """Fit-only median nearest-to-rank gap; use available rank in tiny fixtures."""
    g, _ = _features(x, x, groups, groups)
    if rank < 2 or chunk < 1:
        raise ValueError('invalid temperature rank or chunk')
    z = F.normalize(x.double(), dim=1)
    gaps = []
    for start in range(0, len(x), chunk):
        score = z[start:start+chunk] @ z.T
        allowed = g[start:start+chunk, None] != g[None, :]
        if int(allowed.sum(1).min()) < 2:
            raise ValueError('normal matching needs two other-image reference patches')
        score.masked_fill_(~allowed, -torch.inf)
        # Each row uses rank 8, or its last available reference if fewer exist.
        values = score.topk(min(rank, len(x)), dim=1).values
        ids = allowed.sum(1).clamp_max(rank).long()-1
        gaps.append(values[:, 0]-values.gather(1, ids[:, None]).squeeze(1))
    return max(.01, float(torch.cat(gaps).median()))


class MatchingObjective:
    """Blocked float64 objective and analytic derivatives for all coordinates.

    The max term is nonsmooth at neighbor ties. Its current active-branch
    derivative is a solver diagnostic, not a global optimality certificate.
    """
    def __init__(self, query, reference, query_groups, reference_groups,
                 temperature, perturbations=None, beta=.002, chunk=256,
                 score_weight=1., distribution_weight=1., floor_mass=1e-4,
                 cap_factor=4.):
        self.qg, self.rg = _features(query, reference, query_groups, reference_groups)
        values = (temperature, beta, score_weight, distribution_weight, floor_mass, cap_factor)
        if (not all(math.isfinite(float(v)) for v in values) or temperature <= 0 or
                min(beta, score_weight, distribution_weight) < 0 or
                score_weight+distribution_weight <= 0 or not 0 < floor_mass <= 1 or
                cap_factor < 1 or chunk < 1):
            raise ValueError('invalid normal matching objective parameters')
        self.q, self.r = query.double(), reference.double()
        self.temperature, self.beta, self.chunk = float(temperature), float(beta), int(chunk)
        self.score_weight, self.distribution_weight = float(score_weight), float(distribution_weight)
        self.floor_mass, self.cap_factor = float(floor_mass), float(cap_factor)
        self.teacher, info, distance = [], [], []
        r = F.normalize(self.r, dim=1)
        for start in range(0, len(self.q), self.chunk):
            q = F.normalize(self.q[start:start+self.chunk], dim=1)
            allowed = self.qg[start:start+self.chunk, None] != self.rg[None, :]
            if int(allowed.sum(1).min()) < 2:
                raise ValueError('normal matching needs two other-image reference patches')
            cosine = (q @ r.T).masked_fill(~allowed, -torch.inf)
            logt = F.log_softmax(cosine/self.temperature, dim=1)
            prob = logt.exp()
            logt = logt.masked_fill(~allowed, 0)
            self.teacher.append((allowed, prob, logt))
            distance.append((1-cosine.max(1).values).clamp_min(0))
            info.append(allowed.sum(1).double().log()+(prob*logt).sum(1))
        dd = torch.cat(distance)
        self.weights = (1+dd/dd.mean().clamp_min(1e-4)).clamp_max(4)
        self.weights /= self.weights.mean()
        self.scale = (self.weights*torch.cat(info)).mean().clamp_min(1e-6)
        self.max_target = 1-dd
        self.score_scale = (self.weights*dd.square()).mean().clamp_min(1e-4)
        self.va = self.vb = None
        self.vscale = self.q.new_tensor(1.)
        # beta=0 deliberately does not inspect nuisance data, including starts.
        if perturbations is not None and self.beta:
            clean, perturbed = perturbations
            if (clean.ndim != 2 or perturbed.ndim != 3 or not len(clean) or not len(perturbed) or
                    perturbed.shape[1:] != clean.shape or clean.shape[1] != self.q.shape[1] or
                    clean.device != self.q.device or perturbed.device != self.q.device or
                    not torch.isfinite(clean).all() or not torch.isfinite(perturbed).all()):
                raise ValueError('invalid paired normal nuisance features')
            self.va = clean.double().repeat(len(perturbed), 1)
            self.vb = perturbed.double().reshape(-1, clean.shape[1])
            if bool((self.va.norm(dim=1) <= 1e-12).any() or (self.vb.norm(dim=1) <= 1e-12).any()):
                raise ValueError('zero dense nuisance rows')
            self.vscale = (1-F.cosine_similarity(self.va, self.vb)).mean().clamp_min(1e-4)
        self.evaluations = 0

    def project(self, p):
        return project_bounded_simplex(p, self.floor_mass, self.cap_factor)

    @torch.no_grad()
    @precise_matmul()
    def evaluate(self, active, p, gradient=False, full_gradient=False):
        active = torch.as_tensor(active, device=self.q.device, dtype=torch.long)
        p = torch.as_tensor(p, device=self.q.device, dtype=torch.float64)
        if (active.ndim != 1 or p.shape != active.shape or not len(active) or
                active.min() < 0 or active.max() >= self.q.shape[1] or
                len(active.unique()) != len(active) or not torch.isfinite(p).all() or (p < 0).any()):
            raise ValueError('invalid matching support or weights')
        rx = self.r[:, active]
        rb = rx.square() @ p
        if bool((rb <= 1e-20).any()):
            raise ValueError('zero sparse reference rows')
        loss, score_loss = p.new_zeros(()), p.new_zeros(())
        gradx = self.r if full_gradient else rx
        grad = torch.zeros(gradx.shape[1], device=p.device, dtype=p.dtype) if gradient else None
        for block, (allowed, t, logt) in enumerate(self.teacher):
            start = block*self.chunk
            qx = self.q[start:start+self.chunk, active]
            qa = qx.square() @ p
            if bool((qa <= 1e-20).any()):
                raise ValueError('zero sparse query rows')
            den = qa.sqrt()[:, None]*rb.sqrt()[None, :]
            cosine = (qx*p) @ rx.T/den
            logits = (cosine/self.temperature).masked_fill(~allowed, -torch.inf)
            logs = F.log_softmax(logits, dim=1).masked_fill(~allowed, 0)
            weights = self.weights[start:start+self.chunk]
            loss += (weights*(t*(logt-logs)).sum(1)).sum()/len(self.q)
            nearest_cos, nearest = cosine.masked_fill(~allowed, -torch.inf).max(1)
            error = nearest_cos-self.max_target[start:start+self.chunk]
            score_loss += (weights*error.square()).sum()/(len(self.q)*self.score_scale)
            if gradient:
                coeff = self.distribution_weight*weights[:, None]*(logits.softmax(dim=1)-t)/(
                    len(self.q)*self.scale*self.temperature)
                coeff[torch.arange(len(qx), device=p.device), nearest] += (
                    self.score_weight*2*weights*error/(len(self.q)*self.score_scale))
                qgrad = self.q[start:start+self.chunk] if full_gradient else qx
                products = coeff*cosine
                grad += (qgrad*((coeff/den) @ gradx)).sum(0)
                grad -= .5*(qgrad.square()*(products.sum(1)/qa)[:, None]).sum(0)
                grad -= .5*(gradx.square()*(products.sum(0)/rb)[:, None]).sum(0)
        neighborhood = loss/self.scale
        var = p.new_zeros(())
        if self.va is not None:
            if full_gradient:
                pp = torch.zeros(self.q.shape[1], device=p.device, dtype=p.dtype)
                pp[active] = p
                va, vb = self.va, self.vb
            else:
                pp, va, vb = p, self.va[:, active], self.vb[:, active]
            try:
                cosine, jac = LocalObjective.cosine(va, vb, pp, gradient)
            except ValueError as error:
                raise ValueError('zero sparse nuisance rows') from error
            var = (1-cosine).mean()/self.vscale
            if gradient:
                grad -= self.beta*jac.mean(0)/self.vscale
        total = self.distribution_weight*neighborhood+self.score_weight*score_loss+self.beta*var
        self.evaluations += 1
        return float(total), grad, {
            'normal_neighbor_distribution_kl_relative':float(neighborhood),
            'normal_match_score_relative_mse':float(score_loss),
            'variability_term':float(self.beta*var), 'relative_variability':float(var)}

    def refit(self, active, initial, maximum=150, tol=1e-5):
        p = self.project(initial.double())
        value, grad, terms = self.evaluate(active, p, True)
        history, step, prev_p, prev_g = [value], 1., None, None
        for _ in range(maximum):
            if float((p-self.project(p-grad)).abs().max()) <= tol:
                break
            if prev_p is not None:
                s, y = p-prev_p, grad-prev_g
                if float(s @ y) > 1e-16:
                    step = min(10., max(1e-7, float((s @ s)/(s @ y))))
            oldp, oldg, accepted = p.clone(), grad.clone(), False
            for _ in range(18):
                proposal = self.project(p-step*grad)
                try:
                    new_value, _, new_terms = self.evaluate(active, proposal)
                except ValueError as error:
                    if 'zero sparse' not in str(error):
                        raise
                    new_value = math.inf
                if new_value <= value+1e-4*float(grad @ (proposal-p)):
                    p, value, terms, accepted = proposal, new_value, new_terms, True
                    break
                step *= .5
            if not accepted:
                break
            prev_p, prev_g = oldp, oldg
            value, grad, terms = self.evaluate(active, p, True)
            history.append(value)
        residual = float((p-self.project(p-grad)).abs().max())
        return p, value, terms, {'residual':residual, 'iterations':len(history)-1,
                                'training_history':history}


def _starts(x, groups, paired, beta, k, rank):
    near, far = neighbor_pairs(x, x, groups, groups, rank=rank)
    distance = (1-F.cosine_similarity(x, x[near])).clamp_min(0)
    weights = (1+distance/distance.mean().clamp_min(1e-4)).clamp_max(4)
    weights /= weights.mean()
    energy = x.double().square().mean(0)
    leverage = (((x.double()-x[near].double()).square()+
                 (x.double()-x[far].double()).square())*weights[:, None]).mean(0)/2
    stability = torch.zeros_like(energy)
    if beta and paired is not None:
        clean, pert = paired
        stability = (pert.double()-clean.double()[None]).square().mean((0, 1))/(energy+1e-6)
    score = leverage/leverage.mean().clamp_min(1e-8)-beta*stability/stability.mean().clamp_min(1e-8)
    rankings = [('local_contribution',score), ('normal_energy',energy),
                ('stable_local',leverage/(1+stability))]
    return [(name, ranking.argsort(descending=True, stable=True)[:k].sort().values)
            for name, ranking in rankings]


@torch.no_grad()
@precise_matmul()
def generate_fit_only_pool(x, groups, perturbations, cfg, label='uvarfs:matching'):
    """This API has no holdout argument: generation and continuation are fit-only."""
    beta = float(cfg.get('beta',.002))
    k = min(int(cfg['max_features']), x.shape[1])
    maximum = int(cfg.get('matching_refit_max_iter',150))
    rounds = int(cfg.get('matching_exchange_max_steps',2))
    proposals = int(cfg.get('matching_exchange_candidates',4))
    chunk = int(cfg.get('matching_chunk',256))
    rank = int(cfg.get('matching_temperature_rank',8))
    tol = float(cfg.get('tol',1e-5))
    if k < 1 or maximum < 1 or rounds < 0 or proposals < 1 or not math.isfinite(tol) or tol <= 0:
        raise ValueError('invalid matching solver budget')
    tau = normal_temperature(x, groups, rank, chunk)
    engine = MatchingObjective(x, x, groups, groups, tau, perturbations if beta else None,
        beta, chunk, float(cfg.get('matching_score_weight',1.)),
        float(cfg.get('matching_distribution_weight',1.)),
        float(cfg.get('cosine_weight_floor_mass',1e-4)),
        float(cfg.get('performance_weight_cap_factor',4.)))
    pool, accepted, started = [], [], time.perf_counter()

    def record(active, p, source, diag=None):
        value, grad, terms = engine.evaluate(active, p, True)
        pool.append({'candidate_id':len(pool), 'source':source, 'active':active.cpu().tolist(),
            'weights':p.cpu().tolist(), 'training_objective':value, 'training_terms':terms,
            'fixed_support_first_order_residual':float((p-engine.project(p-grad)).abs().max()),
            'solver_iterations':0 if diag is None else diag['iterations'], 'diagnostics':diag})

    for name, active in _starts(x, groups, perturbations if beta else None, beta, k, rank):
        uniform = torch.full((k,),1/k,device=x.device,dtype=torch.float64)
        try:
            record(active, uniform, name+':uniform')
            p, value, _, diag = engine.refit(active, uniform, maximum, tol)
            record(active, p, name+':refit', diag)
        except ValueError as error:
            if 'zero sparse' not in str(error):
                raise
            continue
        for iteration in range(rounds):
            _, grad, _ = engine.evaluate(active, p, True, True)
            outside = torch.ones(x.shape[1], device=x.device, dtype=torch.bool)
            outside[active] = False
            all_ids = torch.arange(x.shape[1], device=x.device)
            additions = all_ids[outside][grad[outside].argsort(stable=True)[:proposals]]
            donor = int((p*grad[active]).argmax())
            drop, best = int(active[donor]), None
            for add in additions.tolist():
                trial = active.clone()
                trial[donor] = add
                trial, order = trial.sort()
                try:
                    record(trial, uniform, name+f':exchange{iteration+1}:uniform')
                    q, v, _, d = engine.refit(trial, p[order], maximum, tol)
                    d.update(drop=drop, add=add, accepted_for_continuing_search=False)
                    record(trial, q, name+f':exchange{iteration+1}:refit', d)
                except ValueError as error:
                    if 'zero sparse' not in str(error):
                        raise
                    continue
                if v < value-1e-8 and (best is None or v < best[2]):
                    best = trial, q, v, len(pool)-1
            if best is None:
                break
            accepted.append({'start':name, 'iteration':iteration+1, 'before':value,
                             'after':best[2], 'candidate_id':best[3]})
            pool[best[3]]['diagnostics']['accepted_for_continuing_search'] = True
            active, p, value, _ = best
        print(f'[{label}] start={name} candidates={len(pool)} elapsed={time.perf_counter()-started:.1f}s',flush=True)
    if not pool:
        raise ValueError('no valid fixed-K matching support')
    return engine, pool, accepted


@torch.no_grad()
@precise_matmul()
def fit_matching_uvarfs(x, heldout, groups, heldout_groups, perturbations, cfg,
                        label='uvarfs:matching'):
    started = time.perf_counter()
    if set(torch.as_tensor(groups).cpu().tolist()) & set(torch.as_tensor(heldout_groups).cpu().tolist()):
        raise ValueError('normal fit and holdout image groups must be disjoint')
    engine, pool, accepted = generate_fit_only_pool(x, groups, perturbations, cfg, label)
    # Hash the complete training pool before any holdout loss can affect selection.
    digest = hashlib.sha256(json.dumps(pool,sort_keys=True,allow_nan=False).encode()).hexdigest()
    validation = MatchingObjective(heldout, x, heldout_groups, groups, engine.temperature,
        beta=0., chunk=engine.chunk, score_weight=engine.score_weight,
        distribution_weight=engine.distribution_weight, floor_mass=engine.floor_mass,
        cap_factor=engine.cap_factor)
    for candidate in pool:
        active = torch.tensor(candidate['active'], device=x.device)
        p = torch.tensor(candidate['weights'],device=x.device,dtype=torch.float64)
        try:
            value, _, terms = validation.evaluate(active, p)
        except ValueError as error:
            if 'zero sparse' not in str(error):
                raise
            candidate.update(validation_valid=False,validation_error=str(error))
            continue
        candidate.update(validation_valid=True, validation_objective=value, validation_terms=terms,
            selection_score=value+candidate['training_terms']['variability_term'])
    valid = [c for c in pool if c['validation_valid']]
    if not valid:
        raise ValueError('no valid matching candidate on normal holdout')
    selected = min(valid,key=lambda c:(c['selection_score'],c['candidate_id']))
    active = np.array(selected['active'],dtype=np.int64)
    p = np.array(selected['weights'],dtype=np.float64)
    weights = np.zeros(x.shape[1],np.float64)
    weights[active] = p
    residual = selected['fixed_support_first_order_residual']
    return {'objective_definition':MATCHING, 'active':active, 'scales':np.sqrt(p),
        'budgeted_weights':weights, 'objective':selected['training_objective'],
        'validation_objective':selected['validation_objective'], 'selection_score':selected['selection_score'],
        'candidate_id':selected['candidate_id'], 'candidate_pool':pool,
        'candidate_pool_fit_only_sha256':digest, 'support_generation_uses_holdout':False,
        'accepted_exchange_trace':accepted, 'temperature_fit_only':engine.temperature,
        'temperature_policy':'max_0.01_fit_median_nearest_to_rank_gap',
        'temperature_rank':int(cfg.get('matching_temperature_rank',8)),
        'score_weight':engine.score_weight, 'distribution_weight':engine.distribution_weight,
        'beta':engine.beta, 'fixed_feature_budget':len(active), 'weight_cap_factor':engine.cap_factor,
        'weight_floor_mass':engine.floor_mass, 'effective_weight_dimension':float(1/np.square(p).sum()),
        'maximum_relative_weight':float(p.max()), 'simplex_residual':abs(float(p.sum())-1),
        'fixed_support_first_order_residual':residual, 'solver_converged':residual<=float(cfg.get('tol',1e-5)),
        'solver_iterations':selected['solver_iterations'], 'fit_seconds':time.perf_counter()-started,
        'selection_policy':'minimum_normal_holdout_matching_loss_plus_training_nuisance',
        'geometry_metric':'actual_normal_1nn_score_relative_rmse',
        'geometry_error':max(0.,selected['training_terms']['normal_match_score_relative_mse'])**.5,
        'feasible':None, 'global_optimality_claimed':False,
        'objective_certificate_scope':'fixed_support_current_nearest_branch_projected_residual_only',
        'global_geometry_is_diagnostic_only':True, 'objective_dtype':'float64',
        'reference_policy':'all_normal_fit_patches_excluding_same_image',
        'fit_score_scale':float(engine.score_scale), 'fit_distribution_scale':float(engine.scale),
        'fit_nuisance_scale':float(engine.vscale), 'validation_score_scale':float(validation.score_scale),
        'validation_distribution_scale':float(validation.scale),
        'evaluation_requests':engine.evaluations+validation.evaluations,
        **selected['training_terms']}
