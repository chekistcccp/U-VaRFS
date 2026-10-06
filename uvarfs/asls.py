from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
import time
from scipy.stats import rankdata
from .numerics import precise_matmul


def cosine_gram(x: torch.Tensor) -> torch.Tensor:
    x = F.normalize(x.float(), dim=-1)
    return x @ x.T


class LayerGramGeometry:
    """Exact <K_l,K_m> = ||F_l.T F_m||_F^2 without storing patch Gram matrices."""
    def __init__(self,layer_features,layers):
        features=[F.normalize(layer_features[layer].float(),dim=1) for layer in layers]
        shapes={tuple(x.shape) for x in features}
        if len(shapes)!=1 or features[0].ndim!=2:
            raise ValueError('ASLS layers must contain corresponding normal samples and equal hidden dimensions')
        n,d=features[0].shape
        count=len(features)
        joined=torch.cat(features,dim=1)
        products=(joined.T @ joined).reshape(count,d,count,d)
        self.inner_products=products.square().sum(dim=(1,3)).double()
        self.reference=torch.full((count,),1/count,device=joined.device,dtype=torch.float64)
        self.denominator=torch.sqrt(self.reference @ self.inner_products @ self.reference).clamp_min(1e-8)
        diagonal=torch.stack([x.square().sum(1).double() for x in features]).mean(0)
        total=sum(x.sum(0).double().square().sum() for x in features)/count
        off_count=n*(n-1)
        self.off_mean=float((total-diagonal.sum())/off_count) if off_count else None
        second=(self.denominator.square()-diagonal.square().sum())/max(off_count,1)
        self.off_std=float(torch.sqrt((second-self.off_mean**2).clamp_min(0))) if off_count else None
        self.sample_count=n

    def error(self,difference):
        difference=difference.to(dtype=self.inner_products.dtype)
        return torch.sqrt((difference @ self.inner_products @ difference).clamp_min(1e-24))/self.denominator

    def subset_error(self,indices):
        difference=-self.reference.clone()
        difference[indices]+=1/len(indices)
        return self.error(difference)


@precise_matmul()
def fit_asls(layer_features: dict[int, torch.Tensor], layer_variability: dict[int, float], cfg: dict) -> dict:
    layers = sorted(layer_features)
    steps=int(cfg.get('steps',300))
    if not layers or steps<1 or not 1 <= int(cfg.get('min_layers',2)) <= min(int(cfg.get('max_layers',6)),len(layers)):
        raise ValueError('invalid ASLS layer budget or optimizer steps')
    representation=cfg.get('geometry_representation','patch')
    if representation not in {'pooled','patch'}:
        raise ValueError('ASLS geometry_representation must be pooled or patch')
    geometry=LayerGramGeometry(layer_features,layers) if representation=='patch' else None
    if geometry is None:
        grams = torch.stack([cosine_gram(layer_features[l]) for l in layers], dim=0)
        consensus = grams.mean(dim=0)
        denom = consensus.norm().clamp_min(1e-8)
    device=next(iter(layer_features.values())).device
    v = torch.tensor([layer_variability.get(l, 0.0) for l in layers], device=device, dtype=torch.float32)
    if v.numel() and v.max() > v.min():
        v = (v - v.min()) / (v.max() - v.min() + 1e-8)
    logits = torch.zeros(len(layers), device=device, requires_grad=True)
    opt = torch.optim.Adam([logits], lr=float(cfg.get("lr", 0.05)))
    gamma = float(cfg.get("variability_weight", 0.15))
    rho = float(cfg.get("sparsity_weight", 0.02))
    history=[]
    t0=time.perf_counter()
    log_every=max(1,int(cfg.get('log_every',50)))
    for step in range(1,steps+1):
        p = torch.sigmoid(logits)
        if geometry is None:
            weighted = (p[:, None, None] * grams).sum(0) / len(layers)
            geom = (weighted - consensus).norm() / denom
        else:
            geom=geometry.error((p-1)/len(layers))
        var = (p * v).sum() / p.sum().clamp_min(1e-6)
        sparse = p.mean()
        loss = geom + gamma * var + rho * sparse
        log_step=step==1 or step%log_every==0 or step==steps
        term_gradients={}
        if log_step:
            for name,term in [('geometry',geom),('variability',gamma*var),('sparsity',rho*sparse)]:
                term_gradients[name]=torch.autograd.grad(term,logits,retain_graph=True)[0].detach().cpu().tolist()
        opt.zero_grad(set_to_none=True); loss.backward()
        if log_step:
            history.append({'step':step,'loss':float(loss.detach()),'geometry':float(geom.detach()),
                            'variability':float(var.detach()),'sparsity':float(sparse.detach()),
                            'gradient_norm':float(logits.grad.norm().detach()),
                            'term_gradients':term_gradients})
            print(f'[asls] step={step}/{steps} loss={history[-1]["loss"]:.5f} '
                  f'geometry={history[-1]["geometry"]:.5f} probability_mean={history[-1]["sparsity"]:.5f}',flush=True)
        opt.step()
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    order = np.argsort(-probs)
    tol = float(cfg.get("geometry_tolerance", 0.05))
    min_layers = int(cfg.get("min_layers", 2)); max_layers = int(cfg.get("max_layers", 6))
    best = list(order[:min_layers])
    if geometry is None:
        cons = consensus.detach()
    selection_path=[]
    for k in range(min_layers, min(max_layers, len(layers)) + 1):
        ids = order[:k]
        err = float(((grams[ids].mean(0) - cons).norm() / denom).cpu()) if geometry is None else float(geometry.subset_error(ids))
        selection_path.append({'layer_count':k,'layers':[layers[i] for i in ids],
                               'geometry_error':err,'feasible':err<=tol})
        best = list(ids)
        if err <= tol:
            break
    selected = [layers[i] for i in best]
    final_err = float(((grams[best].mean(dim=0) - consensus).norm() / denom).cpu()) if geometry is None else float(geometry.subset_error(best))
    if geometry is None:
        off_diagonal=~torch.eye(cons.shape[0],device=cons.device,dtype=torch.bool)
        off_mean=float(cons[off_diagonal].mean()) if off_diagonal.any() else None
        off_std=float(cons[off_diagonal].std(unbiased=False)) if off_diagonal.any() else None
        layer_errors={str(l):float((grams[i]-cons).norm()/denom) for i,l in enumerate(layers)}
    else:
        off_mean,off_std=geometry.off_mean,geometry.off_std
        layer_errors={str(l):float(geometry.subset_error([i])) for i,l in enumerate(layers)}
    raw_v=np.array([layer_variability.get(l,0.0) for l in layers])
    ranks_p=rankdata(probs)
    ranks_v=rankdata(raw_v)
    correlation=float(np.corrcoef(ranks_p,ranks_v)[0,1]) if np.ptp(raw_v)>0 and np.ptp(probs)>0 else None
    saturated=bool(np.min(probs)>0.98)
    if saturated:
        print('[asls] diagnostic: gates near all-open; preserve objective and inspect normal geometry/gradient trace',flush=True)
    return {
        "selected_layers": selected,
        "geometry_representation":representation,
        "probabilities": {str(l): float(probs[i]) for i,l in enumerate(layers)},
        "variability": {str(l): float(layer_variability.get(l, 0.0)) for l in layers},
        "geometry_error": final_err,
        "feasible":final_err<=tol,
        "selection_path":selection_path,
        "optimizer_trace":history,
        "fit_seconds":time.perf_counter()-t0,
        "diagnostics":{'gates_near_all_open':saturated,'probability_span':float(np.ptp(probs)),
                       'gate_variability_rank_correlation':correlation,
                       'consensus_off_diagonal_mean':off_mean,
                       'consensus_off_diagonal_std':off_std,
                       'layer_geometry_errors':layer_errors,
                       'normal_geometry_samples':int(next(iter(layer_features.values())).shape[0]),
                       'representation':'normal_patch_cosine_gram' if representation=='patch' else 'normal_mean_pooled_cosine_gram',
                       'matmul_precision':'highest'},
    }
