"""Normal-only local cosine selection. No abnormal examples enter this module."""
from __future__ import annotations

import itertools
import time
import numpy as np
import torch
import torch.nn.functional as F

from .numerics import precise_matmul
from .performance_uvarfs import project_bounded_simplex

LOCAL = 'normal_local_fixed_budget'


@torch.no_grad()
@precise_matmul()
def neighbor_pairs(query, reference, query_groups, reference_groups, rank=8, chunk=256):
    """Dense teacher edges; exclude every patch from the query's own image."""
    q, r = F.normalize(query.float(), dim=1), F.normalize(reference.float(), dim=1)
    qg = torch.as_tensor(query_groups, device=q.device)
    rg = torch.as_tensor(reference_groups, device=q.device)
    if len(qg) != len(q) or len(rg) != len(r) or rank < 2:
        raise ValueError('invalid image groups or neighbor rank')
    near, far = [], []
    for start in range(0, len(q), chunk):
        score = q[start:start+chunk] @ r.T
        allowed = qg[start:start+chunk, None] != rg[None, :]
        counts = allowed.sum(1)
        if int(counts.min()) < 2:
            raise ValueError('local fitting needs at least two reference patches from other images')
        score.masked_fill_(~allowed, -torch.inf)
        k = min(rank, int(counts.min()))
        ids = score.topk(k, dim=1, sorted=True).indices
        near.append(ids[:, 0]); far.append(ids[:, -1])
    return torch.cat(near), torch.cat(far)


class LocalObjective:
    """Analytic pair gradient, O(edges * dimensions), without an N by N loss.

    The teacher is the declared dense input. The hard neighbor is still normal:
    its separation is a representation proxy, not anomaly supervision.
    """
    def __init__(self, query, reference, query_groups, reference_groups,
                 perturbations=None, rank=8, rank_weight=1., beta=.002):
        self.query, self.reference = query.float(), reference.float()
        if query.ndim != 2 or reference.ndim != 2 or query.shape[1] != reference.shape[1]:
            raise ValueError('local features must be aligned matrices')
        if not torch.isfinite(query).all() or not torch.isfinite(reference).all():
            raise ValueError('nonfinite local features')
        if bool((query.float().norm(dim=1)<=1e-12).any() or (reference.float().norm(dim=1)<=1e-12).any()):
            raise ValueError('zero dense normal rows cannot define local cosine teacher')
        self.near, self.far = neighbor_pairs(query, reference, query_groups, reference_groups, rank)
        self.a = torch.cat([self.query, self.query])
        self.b = torch.cat([self.reference[self.near], self.reference[self.far]])
        self.target = F.cosine_similarity(self.a, self.b)
        n = len(query)
        distance = (1-self.target[:n]).clamp_min(0)
        self.weights = (1 + distance / distance.mean().clamp_min(1e-4)).clamp_max(4)
        self.weights /= self.weights.mean()
        self.scale = (1-self.target).square().mean().clamp_min(1e-4)
        self.rank_weight, self.beta = float(rank_weight), float(beta)
        self.va = self.vb = None
        if perturbations is not None:
            clean, perturbed = perturbations
            if clean.ndim != 2 or perturbed.ndim != 3 or perturbed.shape[1:] != clean.shape:
                raise ValueError('paired nuisance patches must have shape V,N,M')
            if clean.shape[1] != query.shape[1] or not torch.isfinite(clean).all() or not torch.isfinite(perturbed).all():
                raise ValueError('invalid paired nuisance features')
            self.va = clean.float().repeat(perturbed.shape[0], 1)
            self.vb = perturbed.float().reshape(-1, clean.shape[1])
            self.var_scale = (1-F.cosine_similarity(self.va, self.vb)).clamp_min(0).mean().clamp_min(1e-4)
        self.evaluations = 0

    @staticmethod
    def cosine(a, b, p, gradient=False):
        ha, hb = a.square() @ p, b.square() @ p
        if bool((ha <= 1e-20).any() or (hb <= 1e-20).any()):
            raise ValueError('support produces zero normal or perturbed rows')
        den = (ha * hb).sqrt()
        c = (a * b) @ p / den
        if not gradient:
            return c, None
        jac = a*b / den[:, None] - .5*c[:, None]*(a.square()/ha[:, None]+b.square()/hb[:, None])
        return c, jac

    def evaluate(self, active, p, gradient=False, full_gradient=False):
        self.evaluations += 1
        ids = torch.as_tensor(active, device=self.a.device, dtype=torch.long)
        p = p.to(self.a)
        if full_gradient:
            weights = torch.zeros(self.a.shape[1], device=p.device, dtype=p.dtype)
            weights[ids] = p
            a, b = self.a, self.b
        else:
            weights = p
            a, b = self.a[:, ids], self.b[:, ids]
        c, jac = self.cosine(a, b, weights, gradient)
        n = len(self.query)
        delta = c-self.target
        w = self.weights.repeat(2)
        rep = (w*delta.square()).mean()/self.scale
        violation = (self.target[:n]-self.target[n:] - c[:n]+c[n:]).clamp_min(0)
        rank = (self.weights*violation.square()).mean()/self.scale
        value = rep + self.rank_weight*rank
        grad = None
        if gradient:
            coeff = 2*w*delta/(2*n*self.scale)
            rank_coeff = 2*self.rank_weight*self.weights*violation/(n*self.scale)
            coeff[:n] -= rank_coeff; coeff[n:] += rank_coeff
            grad = coeff @ jac
        var = torch.zeros((), device=p.device)
        if self.va is not None and self.beta:
            va, vb = (self.va, self.vb) if full_gradient else (self.va[:, ids], self.vb[:, ids])
            vc, vjac = self.cosine(va, vb, weights, gradient)
            var = (1-vc).mean()/self.var_scale
            value = value+self.beta*var
            if gradient:
                grad -= self.beta*vjac.mean(0)/self.var_scale
        terms = {'local_representation_term':float(rep), 'rank_term':float(rank),
                 'relative_variability':float(var), 'variability_term':float(self.beta*var),
                 'geometry_error':float(rep.sqrt()),
                 'normal_rank_violation_fraction':float((violation>1e-6).float().mean())}
        return float(value), grad, terms


def _refit(engine, active, cfg, initial=None):
    k = len(active)
    project = lambda p: project_bounded_simplex(p, float(cfg.get('cosine_weight_floor_mass',1e-4)),
                                                float(cfg.get('performance_weight_cap_factor',4.)))
    p = torch.full((k,),1/k,device=engine.a.device) if initial is None else project(initial)
    iterations = int(cfg.get('local_refit_max_iter',150))
    if iterations < 1:
        raise ValueError('local refit iteration budget must be positive')
    value, grad, terms = engine.evaluate(active,p,True)
    steps = 0
    for iteration in range(iterations):
        residual = float((p-project(p-grad)).abs().max())
        if residual <= float(cfg.get('tol',1e-5)):
            break
        step = 1.
        accepted = False
        for _ in range(20):
            proposal = project(p-step*grad)
            new_value, _, new_terms = engine.evaluate(active,proposal)
            if new_value <= value + 1e-4*float(grad @ (proposal-p)):
                p, value, terms = proposal, new_value, new_terms
                accepted = True
                break
            step *= .5
        if not accepted:
            break
        value, grad, terms = engine.evaluate(active,p,True)
        steps = iteration+1
    residual = float((p-project(p-grad)).abs().max())
    return p, value, terms, residual, steps


@torch.no_grad()
@precise_matmul()
def fit_local_uvarfs(x, heldout, groups, heldout_groups, perturbations, cfg, label='uvarfs:local'):
    """Fit on train edges; select a finite fixed-K pool using normal holdout."""
    t0 = time.perf_counter()
    beta = float(cfg.get('beta',.002))
    engine = LocalObjective(x,x,groups,groups,perturbations,int(cfg.get('local_neighbor_rank',8)),
                            float(cfg.get('local_rank_weight',1.)), beta)
    validation = LocalObjective(heldout,x,heldout_groups,groups,rank=int(cfg.get('local_neighbor_rank',8)),
                                rank_weight=float(cfg.get('local_rank_weight',1.)),beta=0.)
    m = x.shape[1]; k = min(m,int(cfg['max_features']))
    if k < 1 or beta < 0 or float(cfg.get('local_rank_weight',1.)) < 0:
        raise ValueError('invalid local objective budget or coefficients')
    all_ids = torch.arange(m,device=x.device)
    uniform = torch.full((m,),1/m,device=x.device)
    # Dense representation gradients vanish at the teacher. Local coordinate
    # contributions provide finite support starts; these are heuristics.
    energy = x.float().square().mean(0)
    edge_leverage = ((engine.a-engine.b).square()*engine.weights.repeat(2)[:,None]).mean(0)
    stability = torch.zeros_like(energy)
    if perturbations is not None and beta:
        clean, pert = perturbations
        stability = (pert-clean[None]).square().mean((0,1))/(energy+1e-6)
    score = edge_leverage/(edge_leverage.mean()+1e-8) - beta*stability/(stability.mean()+1e-8)
    starts = [('local_contribution',score), ('normal_energy',energy),
              ('stable_local_contribution',edge_leverage/(stability+1.))]
    candidates, states = [], []
    def record(active,p,source,residual=None,iterations=0):
        value,_,terms = engine.evaluate(active,p)
        val,_,vterms = validation.evaluate(active,p)
        selection = val + terms['variability_term']
        row = {'candidate_id':len(candidates),'source':source,'active':active.cpu().tolist(),
               'weights':p.cpu().tolist(),'training_objective':value,'validation_objective':val,
               'selection_score':selection,'fixed_support_first_order_residual':residual,
               'solver_iterations':iterations,'training_terms':terms,'validation_terms':vterms}
        candidates.append(row); states.append((active.clone(),p.clone()))
    for name, ranking in starts:
        active = ranking.argsort(descending=True,stable=True)[:k].sort().values
        p = torch.full((k,),1/k,device=x.device)
        try:
            record(active,p,name+':uniform')
            p, value, terms, residual, steps = _refit(engine,active,cfg,p)
            record(active,p,name+':refit',residual,steps)
        except ValueError as error:
            if 'zero' not in str(error):
                raise
            continue
        for _ in range(int(cfg.get('local_exchange_max_steps',2))):
            _,full_grad,_ = engine.evaluate(active,p,True,True)
            outside = torch.ones(m,device=x.device,dtype=torch.bool); outside[active]=False
            add = all_ids[outside][full_grad[outside].argsort(stable=True)[:int(cfg.get('local_exchange_candidates',4))]]
            drop = int(active[(p*full_grad[active]).argmax()])
            best = None
            for j in add.tolist():
                proposal = active.clone(); proposal[proposal==drop]=j; proposal=proposal.sort().values
                try:
                    rp, rv, rt, rr, rs = _refit(engine,proposal,cfg)
                    record(proposal,rp,name+':exchange',rr,rs)
                except ValueError as error:
                    if 'zero' not in str(error):
                        raise
                    continue
                if rv < value-1e-8 and (best is None or rv < best[1]):
                    best = (proposal,rv,rp,rt)
            if best is None:
                break
            active,value,p,terms = best
        print(f'[{label}] start={name} candidates={len(candidates)} elapsed={time.perf_counter()-t0:.1f}s',flush=True)
    if not candidates:
        raise ValueError('no valid fixed-K local support: normal or perturbed rows vanish')
    selected = min(candidates,key=lambda c:(c['selection_score'],c['candidate_id']))
    active,p = states[selected['candidate_id']]
    # Recompute the certificate even when a uniform reference wins validation.
    _,grad,terms = engine.evaluate(active,p,True)
    projected = project_bounded_simplex(p-grad,float(cfg.get('cosine_weight_floor_mass',1e-4)),
                                       float(cfg.get('performance_weight_cap_factor',4.)))
    residual = float((p-projected).abs().max())
    weights = np.zeros(m,np.float32); weights[active.cpu().numpy()] = p.cpu().numpy()
    return {'objective_definition':LOCAL,'active':active.cpu().numpy(),
            'scales':p.sqrt().cpu().numpy(),'budgeted_weights':weights,
            'objective':selected['training_objective'],'validation_objective':selected['validation_objective'],
            'selection_score':selected['selection_score'],'candidate_id':selected['candidate_id'],
            'candidate_pool':candidates,'beta':beta,'rank_weight':engine.rank_weight,
            'fixed_feature_budget':k,'weight_cap_factor':float(cfg.get('performance_weight_cap_factor',4.)),
            'weight_floor_mass':float(cfg.get('cosine_weight_floor_mass',1e-4)),
            'effective_weight_dimension':float(1/p.square().sum()),
            'maximum_relative_weight':float(p.max()),'simplex_residual':abs(float(p.sum())-1),
            'fixed_support_first_order_residual':residual,'solver_converged':residual<=float(cfg.get('tol',1e-5)),
            'solver_iterations':selected['solver_iterations'],'fit_seconds':time.perf_counter()-t0,
            'selection_policy':'minimum_normal_holdout_local_rank_loss_plus_training_nuisance',
            'geometry_metric':'normal_local_pair_relative_rmse','geometry_error':terms['geometry_error'],
            'feasible':None,'global_optimality_claimed':False,
            'objective_certificate_scope':'fixed_support_capped_simplex_first_order_only',
            'global_geometry_is_diagnostic_only':True,'evaluation_requests':engine.evaluations+validation.evaluations,
            **terms}


@torch.no_grad()
@precise_matmul()
def fit_local_asls(patches, heldout, groups, heldout_groups, paired, cfg):
    """Compare all subsets in the original 2--6 layer budget on normal holdout."""
    t0=time.perf_counter(); layers=sorted(patches)
    full=torch.cat([patches[l] for l in layers],1)
    val=torch.cat([heldout[l] for l in layers],1)
    teacher=LocalObjective(val,full,heldout_groups,groups,rank=int(cfg.get('local_neighbor_rank',8)),beta=0.)
    a,b={},{}
    for l in layers:
        a[l]=torch.cat([heldout[l],heldout[l]]).float()
        b[l]=torch.cat([patches[l][teacher.near],patches[l][teacher.far]]).float()
    ab=torch.stack([(a[l]*b[l]).sum(1) for l in layers])
    aa=torch.stack([a[l].square().sum(1) for l in layers])
    bb=torch.stack([b[l].square().sum(1) for l in layers])
    vab=vaa=vbb=None
    if paired is not None:
        clean,pert=paired
        va={l:clean[l].repeat(pert[l].shape[0],1).float() for l in layers}
        vb={l:pert[l].reshape(-1,pert[l].shape[-1]).float() for l in layers}
        vab=torch.stack([(va[l]*vb[l]).sum(1) for l in layers])
        vaa=torch.stack([va[l].square().sum(1) for l in layers])
        vbb=torch.stack([vb[l].square().sum(1) for l in layers])
    lo,hi=int(cfg.get('min_layers',2)),min(int(cfg.get('max_layers',6)),len(layers))
    if not 1<=lo<=hi:
        raise ValueError('invalid local ASLS layer budget')
    path=[]; n=len(val)
    for k in range(lo,hi+1):
        for subset in itertools.combinations(range(len(layers)),k):
            ids=list(subset); ha=aa[ids].sum(0); hb=bb[ids].sum(0)
            if bool((ha<=1e-20).any() or (hb<=1e-20).any()):
                continue
            c=ab[ids].sum(0)/(ha*hb).sqrt()
            rep=(teacher.weights.repeat(2)*(c-teacher.target).square()).mean()/teacher.scale
            violation=(teacher.target[:n]-teacher.target[n:]-c[:n]+c[n:]).clamp_min(0)
            rank=(teacher.weights*violation.square()).mean()/teacher.scale
            var=0.
            if vab is not None:
                vh=vaa[ids].sum(0)*vbb[ids].sum(0)
                if bool((vh<=1e-20).any()):
                    continue
                var=float((1-vab[ids].sum(0)/vh.sqrt()).mean().clamp_min(0))
            score=float(rep)+float(cfg.get('local_rank_weight',1.))*float(rank)+float(cfg.get('variability_weight',.15))*var
            path.append({'layers':[layers[i] for i in ids],'layer_count':k,'selection_score':score,
                         'local_representation_term':float(rep),'rank_term':float(rank),'nuisance_cosine_distance':var})
        print(f'[asls:local] evaluated through {k} layers; candidates={len(path)}',flush=True)
    if not path:
        raise ValueError('no valid local ASLS subset')
    best=min(path,key=lambda c:(c['selection_score'],c['layers']))
    return {'selected_layers':best['layers'],'geometry_representation':'patch',
            'discrete_selection':'normal_local','geometry_error':best['local_representation_term']**.5,
            'geometry_metric':'normal_local_pair_relative_rmse','feasible':None,
            'selection_score':best['selection_score'],'selection_path':path,
            'selection_policy':'minimum_normal_holdout_local_rank_loss_plus_training_nuisance',
            'compression_is_not_selection_priority':True,'probabilities':{},'variability':{},
            'fit_seconds':time.perf_counter()-t0,'global_optimality_claimed':False}
