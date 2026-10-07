"""Budgeted support search for the unchanged U-VaRFS quadratic objective.

No labels enter these routines. H, P, beta, rscale and lambda are frozen
throughout pruning and refitting. This is not inverse-Hessian OBS pruning.
"""
from __future__ import annotations

import torch


def top_weight_budget(weights, cfg):
    m=weights.shape[0]
    minimum=min(int(cfg.get('min_features',32)),m)
    maximum=min(int(cfg.get('max_features',256)),m)
    if not 1<=minimum<=maximum:
        raise ValueError('feature budgets must satisfy 1 <= min_features <= max_features')
    threshold=float(cfg.get('active_threshold',1e-4))
    retained=torch.zeros_like(weights)
    for j in range(weights.shape[1]):
        active=torch.where(weights[:,j]>threshold)[0]
        if len(active)>maximum or len(active)<minimum:
            count=maximum if len(active)>maximum else minimum
            active=torch.argsort(weights[:,j],descending=True,stable=True)[:count]
        retained[active,j]=weights[active,j]
    return retained


def objective_prune(weights, H, P, beta, rscale, lambdas, cfg, label):
    """Remove the dimension with the smallest exact one-coordinate loss change.

    Delta F_j = .5*A_jj*w_j**2 - gradient_j*w_j, A=H+2*beta*R.
    Update the full gradient after each deletion; do not construct dense R.
    """
    maximum=min(int(cfg.get('max_features',256)),weights.shape[0])
    threshold=float(cfg.get('active_threshold',1e-4))
    current=torch.where(weights>threshold,weights,0).clone()
    counts=(current>0).sum(0)
    deletions=(counts-maximum).clamp_min(0)
    steps=int(deletions.max().item())
    columns=torch.arange(weights.shape[1],device=weights.device)
    linear=H.sum(1,keepdim=True)
    lams=torch.as_tensor(lambdas,device=H.device,dtype=H.dtype)[None,:]
    diagonal=H.diagonal()+2*beta*rscale*P.square().sum(1)

    def gradient():
        return H@current+2*beta*rscale*(P@(P.T@current))-linear+lams

    grad=gradient()
    for step in range(steps):
        needed=counts>maximum
        change=.5*diagonal[:,None]*current.square()-grad*current
        change=change.masked_fill(current<=0,float('inf'))
        ids=change.argmin(0)
        removed=current[ids,columns]*needed
        current[ids,columns]-=removed
        a_columns=H[:,ids]+2*beta*rscale*(P@P[ids].T)
        grad-=a_columns*removed[None,:]
        counts-=needed.to(counts.dtype)
        if (step+1)%64==0:
            grad=gradient()  # Bound accumulated FP32 incremental-update error.
        if step==0 or (step+1)%200==0 or step+1==steps:
            print(f'[{label}:support] prune={step+1}/{steps}',flush=True)
    return current,deletions.cpu().tolist()


def refit_fixed_support(seed, original_weights, H, P, beta, rscale, lambdas, cfg, label):
    """Batched box solves on fixed supports of the ORIGINAL full objective.

    Critically, the linear target is (H*1)_S, not H_SS*1. The omitted
    feature cross terms, original P normalization and rscale are retained.
    """
    minimum=min(int(cfg.get('min_features',32)),H.shape[0])
    threshold=float(cfg.get('active_threshold',1e-4))
    supports=[]
    for j in range(seed.shape[1]):
        active=torch.where(seed[:,j]>threshold)[0]
        if len(active)<minimum:
            order=torch.argsort(original_weights[:,j],descending=True,stable=True)
            padding=order[~torch.isin(order,active)][:minimum-len(active)]
            active=torch.cat([active,padding])
        supports.append(active.sort().values)
    width=max(map(len,supports))
    ids=torch.zeros((len(supports),width),device=H.device,dtype=torch.long)
    valid=torch.zeros_like(ids,dtype=torch.bool)
    for j,active in enumerate(supports):
        ids[j,:len(active)]=active
        valid[j,:len(active)]=True
    hp=H[ids[:,:,None],ids[:,None,:]]
    pp=P[ids]*valid[:,:,None]
    A=(hp+2*beta*rscale*(pp@pp.transpose(1,2)))*valid[:,:,None]*valid[:,None,:]
    target=H.sum(1)[ids]*valid
    lams=torch.as_tensor(lambdas,device=H.device,dtype=H.dtype)[:,None]
    lipschitz=(A.abs().sum(2).amax(1,keepdim=True)*(1+1e-6)).clamp_min(1e-12)
    z=seed.T.gather(1,ids)*valid
    y=z.clone(); t=torch.ones((len(supports),1),device=H.device,dtype=H.dtype)

    def apply(value):
        return torch.bmm(A,value[:,:,None]).squeeze(2)

    def value(value):
        return (.5*value*apply(value)-target*value+lams*value).sum(1)

    best=z.clone(); best_value=value(z)
    iterations=int(cfg.get('support_refit_max_iter',cfg.get('max_iter',400)))
    tolerance=float(cfg.get('tol',1e-5))
    interval=max(1,int(cfg.get('check_every',25)))
    if iterations<1:
        raise ValueError('support_refit_max_iter must be positive')
    for step in range(1,iterations+1):
        zn=(y-(apply(y)-target+lams)/lipschitz).clamp(0,1)*valid
        tn=.5*(1+torch.sqrt(1+4*t*t))
        restart=((y-zn)*(zn-z)).sum(1,keepdim=True)>0
        momentum=torch.where(restart,0.,(t-1)/tn)
        y=zn+momentum*(zn-z)
        z=zn; t=torch.where(restart,1.,tn)
        if step%interval==0 or step==iterations:
            current_value=value(z)
            improved=current_value<best_value
            best=torch.where(improved[:,None],z,best)
            best_value=torch.minimum(best_value,current_value)
            grad=apply(best)-target+lams
            gap=(grad.clamp_min(0)*best-grad.clamp_max(0)*(1-best)*valid).sum(1)
            if bool((gap<=tolerance).all()):
                break
    # Keep detector weights and diagnostics consistent after thresholding.
    best=torch.where(best>threshold,best,0)
    grad=apply(best)-target+lams
    gap=(grad.clamp_min(0)*best-grad.clamp_max(0)*(1-best)*valid).sum(1)
    out=torch.zeros_like(seed)
    # Invalid padded indices may repeat row 0; scatter_add avoids overwriting it.
    out.scatter_add_(0,ids.T,(best*valid).T)
    print(f'[{label}:support] refit iterations={step} max_fixed_support_gap={float(gap.max()):.3e}',flush=True)
    return out,{'iterations':step,'fixed_support_box_gap':gap.cpu().tolist(),
                'fixed_support_converged':(gap<=tolerance).cpu().tolist()}


def refine_budget(weights,H,P,beta,rscale,lambdas,cfg,label):
    """Keep only changes that improve the original objective and preserve geometry.

    The legacy candidate is included independently for every lambda, so support
    heuristics cannot worsen its budgeted objective/geometry (at working precision).
    This safeguard uses normal training quantities, never anomaly performance.
    """
    from .u_varfs import _objective_certificate
    legacy=top_weight_budget(weights,cfg)
    pruned,deletions=objective_prune(weights,H,P,beta,rscale,lambdas,cfg,label)
    seeds=torch.cat([legacy,pruned],1)
    fitted,refit=refit_fixed_support(seeds,torch.cat([weights,weights],1),H,P,beta,rscale,
                                   list(lambdas)*2,cfg,label)
    paths=weights.shape[1]
    candidates=torch.cat([legacy,pruned,fitted],1)
    certificate=_objective_certificate(candidates,H,P,beta,rscale,list(lambdas)*4)
    objective=certificate['objective'].reshape(4,paths)
    difference=1-candidates
    error=torch.sqrt(((difference*(H@difference)).sum(0)/H.sum().clamp_min(1e-12)).clamp_min(0)).reshape(4,paths)
    nonzero=(candidates>0).sum(0).reshape(4,paths)
    minimum=min(int(cfg.get('min_features',32)),H.shape[0])
    required=torch.minimum(nonzero[0],torch.full_like(nonzero[0],minimum))
    allowed=(objective<=objective[0])&(error<=error[0])&(nonzero>=required)
    choice=objective.masked_fill(~allowed,float('inf')).argmin(0)
    columns=choice*paths+torch.arange(paths,device=weights.device)
    retained=candidates[:,columns]
    sources=['top_weights','objective_prune','top_weights_refit','objective_prune_refit']
    diagnostics=[]
    for j,index in enumerate(choice.cpu().tolist()):
        diagnostics.append({'budget_source':sources[index],
            'legacy_budgeted_objective':float(objective[0,j]),
            'legacy_geometry_error':float(error[0,j]),
            'objective_improvement':float(objective[0,j]-objective[index,j]),
            'geometry_improvement':float(error[0,j]-error[index,j]),
            'pruned_dimensions':deletions[j],
            'fixed_support_box_gap':refit['fixed_support_box_gap'][(index-2)*paths+j] if index>=2 else None,
            'fixed_support_converged':refit['fixed_support_converged'][(index-2)*paths+j] if index>=2 else None,
            'support_refit_iterations':refit['iterations'],
            'support_optimality_scope':'fixed_support_only'})
    return retained,diagnostics


def objective_exchange(seed, H, P, beta, rscale, lambdas, cfg, label):
    """Exchange one retained dimension with one omitted dimension per lambda.

    Minimize the exact two-coordinate loss change over the new coefficient in
    [0,1]. Unlike deletion alone, this can revisit omitted dimensions. Each
    accepted exchange preserves cardinality and decreases both the ORIGINAL
    objective and its representation term. No support-global optimality claim.
    """
    current=seed.clone()
    steps=int(cfg.get('support_exchange_max_steps',8))
    chunk=int(cfg.get('support_exchange_chunk',256))
    minimum_gain=float(cfg.get('support_exchange_min_improvement',1e-8))
    if steps<0 or chunk<1 or not 0<minimum_gain<float('inf'):
        raise ValueError('invalid support exchange numerical budget')
    paths=current.shape[1]; m=current.shape[0]
    columns=torch.arange(paths,device=current.device)
    count=(current>0).sum(0)
    width=max(int(count.max().item()),1)
    target=H.sum(1,keepdim=True)
    lams=torch.as_tensor(lambdas,device=H.device,dtype=H.dtype)[None,:]
    hdiag=H.diagonal()
    adiag=hdiag+2*beta*rscale*P.square().sum(1)
    exchanges=torch.zeros(paths,device=H.device,dtype=torch.long)
    for step in range(steps):
        # Stable order resolves equal-loss candidates without RNG consumption.
        ids=torch.argsort(current.T,descending=True,stable=True)[:,:width]
        removed=current.T.gather(1,ids)
        valid=removed>0
        rep_grad=H@current-target
        gradient=rep_grad+2*beta*rscale*(P@(P.T@current))+lams
        remove_loss=.5*adiag[ids]*removed.square()-gradient.T.gather(1,ids)*removed
        remove_rep=.5*hdiag[ids]*removed.square()-rep_grad.T.gather(1,ids)*removed
        best=torch.full((paths,),-minimum_gain,device=H.device,dtype=H.dtype)
        best_out=ids[:,0].clone(); best_in=best_out.clone()
        best_weight=torch.zeros(paths,device=H.device,dtype=H.dtype)
        for start in range(0,m,chunk):
            end=min(start+chunk,m)
            cross=H[start:end,ids].permute(1,0,2)
            a_cross=cross+2*beta*rscale*torch.einsum('mv,ckv->cmk',P[start:end],P[ids])
            conditional=gradient[start:end].T[:,:,None]-a_cross*removed[:,None,:]
            coefficient=(-conditional/adiag[start:end].clamp_min(1e-30)[None,:,None]).clamp(0,1)
            loss=remove_loss[:,None,:]+conditional*coefficient+.5*adiag[start:end][None,:,None]*coefficient.square()
            rep=remove_rep[:,None,:]+(rep_grad[start:end].T[:,:,None]-cross*removed[:,None,:])*coefficient+.5*hdiag[start:end][None,:,None]*coefficient.square()
            allowed=(current[start:end].T[:,:,None]==0)&valid[:,None,:]
            allowed &= (coefficient>float(cfg.get('active_threshold',1e-4)))&(rep<=0)
            values,flat=loss.masked_fill(~allowed,float('inf')).flatten(1).min(1)
            better=values<best
            outgoing=ids.gather(1,(flat%width)[:,None]).squeeze(1)
            incoming=start+flat//width
            new_weight=coefficient.flatten(1).gather(1,flat[:,None]).squeeze(1)
            best=torch.where(better,values,best)
            best_out=torch.where(better,outgoing,best_out)
            best_in=torch.where(better,incoming,best_in)
            best_weight=torch.where(better,new_weight,best_weight)
        changed=best_weight>0
        if not bool(changed.any()):
            break
        current[best_out[changed],columns[changed]]=0
        current[best_in[changed],columns[changed]]=best_weight[changed]
        exchanges+=changed
        print(f'[{label}:support] exchange={step+1}/{steps} changed_paths={int(changed.sum())}',flush=True)
    return current,exchanges.cpu().tolist()


def exchange_budget(seed, original_weights, H, P, beta, rscale, lambdas, cfg, label):
    """Protect the v8 candidate independently for every lambda, including refit."""
    from .u_varfs import _objective_certificate
    exchanged,counts=objective_exchange(seed,H,P,beta,rscale,lambdas,cfg,label)
    fitted,info=refit_fixed_support(exchanged,original_weights,H,P,beta,rscale,lambdas,cfg,label)
    paths=seed.shape[1]
    candidates=torch.cat([seed,exchanged,fitted],1)
    objective=_objective_certificate(candidates,H,P,beta,rscale,list(lambdas)*3)['objective'].reshape(3,paths)
    missing=1-candidates
    error=torch.sqrt(((missing*(H@missing)).sum(0)/H.sum().clamp_min(1e-12)).clamp_min(0)).reshape(3,paths)
    nonzero=(candidates>0).sum(0).reshape(3,paths)
    required=torch.minimum(nonzero[0],torch.full_like(nonzero[0],min(int(cfg.get('min_features',32)),H.shape[0])))
    allowed=(objective<=objective[0])&(error<=error[0])&(nonzero>=required)
    choice=objective.masked_fill(~allowed,float('inf')).argmin(0)
    chosen=candidates[:,choice*paths+torch.arange(paths,device=seed.device)]
    diagnostics=[]
    for j,index in enumerate(choice.cpu().tolist()):
        diagnostics.append({'exchange_source':['prune_refit','objective_exchange','objective_exchange_refit'][index],
            'exchange_steps':counts[j],
            'exchange_support_refit_iterations':info['iterations'],
            'prune_refit_budgeted_objective':float(objective[0,j]),
            'prune_refit_geometry_error':float(error[0,j]),
            'exchange_objective_improvement':float(objective[0,j]-objective[index,j]),
            'exchange_geometry_improvement':float(error[0,j]-error[index,j]),
            'exchange_fixed_support_box_gap':info['fixed_support_box_gap'][j] if index==2 else None,
            'exchange_fixed_support_converged':info['fixed_support_converged'][j] if index==2 else None})
    return chosen,diagnostics
