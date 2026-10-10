"""Bounded reuse of the last exact cosine state; no approximate geometry.

The v18 line search evaluates an accepted trial without a gradient, then
rebuilds the same N-by-N cosine matrix for its gradient. Retain only that
last matrix, under an explicit byte limit. Different supports or weights
invalidate it. The original objective and all solver decisions are kept.
"""
from __future__ import annotations
import math
import torch
from .cosine_uvarfs import CosineObjective


class CachedCosineObjective(CosineObjective):
    def __init__(self, x, variability, beta=.002, chunk=512, cache_mib=128):
        if not math.isfinite(cache_mib) or cache_mib<0:
            raise ValueError('invalid cosine state cache limit')
        super().__init__(x,variability,beta,chunk)
        self.cache_limit_bytes=int(cache_mib*1024**2)
        self._state=None
        self.geometry_evaluations=0; self.geometry_reuses=0
        self.identical_result_reuses=0; self.peak_cache_bytes=0

    def cache_diagnostics(self):
        return {'policy':'last exact support and weights only; no approximate reuse',
                'cache_limit_bytes':self.cache_limit_bytes,
                'peak_cache_bytes':self.peak_cache_bytes,
                'objective_requests':self.evaluations,
                'geometry_evaluations':self.geometry_evaluations,
                'geometry_reuses':self.geometry_reuses,
                'identical_result_reuses':self.identical_result_reuses}

    @torch.no_grad()
    def evaluate(self, active, p, gradient=False, full_gradient=False):
        n,m=self.x.shape; k=len(active); size=self.x.element_size()
        # Reserve the support tensors, row normalizers, matrix blocks, and
        # BOTH possible gradients before deciding to keep any state.
        required=size*(n*n+2*n*k+2*n+2*m)+p.element_size()*k+active.element_size()*k+8
        if required>self.cache_limit_bytes:
            self._state=None
            result=super().evaluate(active,p,gradient,full_gradient)
            self.geometry_evaluations+=1
            return result
        cached=self._state
        same=(cached is not None and active.dtype==cached['active'].dtype and
              active.device==cached['active'].device and p.dtype==cached['p'].dtype and
              p.device==cached['p'].device and torch.equal(active,cached['active']) and torch.equal(p,cached['p']))
        if same:
            self.geometry_reuses+=1
            if not gradient or full_gradient in cached['gradients']:
                self.evaluations+=1; self.identical_result_reuses+=1
                g=cached['gradients'][full_gradient].clone() if gradient else None
                return cached['value'].clone(),g,dict(cached['terms'])
            xs=cached['xs']; h=cached['h']; d=cached['d']; normalized=cached['normalized']
        else:
            # Discard the preceding state before allocating a new one.
            self._state=None; cached=None
            xs=self.x[:,active]
            h=xs.square()@p
            if not torch.isfinite(p).all() or (p<0).any() or not torch.isfinite(h).all() or (h<=0).any():
                raise ValueError('invalid candidate weights or zero cosine row')
            d=h.rsqrt()
            normalized=(xs*torch.sqrt(p))*d[:,None]
        targets=self.x if full_gradient else xs
        if gradient:
            t=targets*d[:,None]
            g=targets.new_zeros(targets.shape[1],dtype=torch.float64)
        squared=self.denominator.new_zeros(())
        blocks=[] if cached is None else cached['blocks']
        for block,start in enumerate(range(0,len(h),self.chunk)):
            end=min(start+self.chunk,len(h))
            actual=normalized[start:end]@normalized.T if cached is None else blocks[block]
            if cached is None:
                blocks.append(actual)
            delta=actual-self.reference[start:end]
            squared+=delta.double().square().sum()
            if gradient:
                first=(t[start:end]*(delta@t)).sum(0,dtype=torch.float64)
                row=(delta*actual).sum(1)
                second=(targets[start:end].square()*(row/h[start:end])[:,None]).sum(0,dtype=torch.float64)
                g+=first-second
        projected=self.P[active].T@p
        var=projected.square().sum()/self.var_base if self.var_base>0 else squared.new_zeros(())
        rep=.5*squared/self.denominator
        value=rep+self.beta*var
        if gradient:
            pv=self.P if full_gradient else self.P[active]
            vg=2*self.beta*(pv@projected)/self.var_base if self.var_base>0 else torch.zeros_like(g)
            g=(g/self.denominator+vg).to(p.dtype)
        if not torch.isfinite(value) or (gradient and not torch.isfinite(g).all()):
            self._state=None
            raise ValueError('nonfinite cosine objective or gradient')
        self.evaluations+=1
        terms={'representation_term':float(rep),'variability_term':float(self.beta*var),
               'relative_variability':float(var),'geometry_error':float(torch.sqrt(squared/self.denominator))}
        if cached is None:
            self.geometry_evaluations+=1
            cached={'active':active.clone(),'p':p.clone(),'xs':xs,'h':h,'d':d,
                    'normalized':normalized,'blocks':blocks,'value':value.clone(),
                    'terms':dict(terms),'gradients':{}}
        if gradient:
            cached['gradients'][full_gradient]=g.clone()
        self._state=cached
        tensors=[cached[name] for name in ['active','p','xs','h','d','normalized','value']]+blocks+list(cached['gradients'].values())
        stored=sum(t.numel()*t.element_size() for t in tensors)
        self.peak_cache_bytes=max(self.peak_cache_bytes,stored)
        return value,g if gradient else None,terms
