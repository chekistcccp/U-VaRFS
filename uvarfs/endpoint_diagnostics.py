"""Read-only feasible directional probes of the unchanged normal matching loss."""
from __future__ import annotations
import math
import torch
from .numerics import precise_matmul

VERSION='normal-endpoint-diagnostics-v1'
METHODS=('precision_sparse','precision_beta_zero','precision_random_weighted_seed42',
         'precision_dense_weighted','precision_dense_beta_zero')
STEPS=(1.,1e-2,1e-4,1e-6,1e-8,1e-10,1e-12)
DIRECTIONS=('projected_gradient','steepest_feasible_pair','toward_parent','uniform_reference','fixed_coordinate_pair')
SETTINGS={'evaluation_version':VERSION,'source_seed':42,'methods':list(METHODS),'step_fractions':list(STEPS),
          'directions':list(DIRECTIONS),'fit_images':144,'selection_images_used':0,'audit_images_used':0,
          'nuisance_images':96,'patches_per_image':16,'feature_budget':256,'roundoff_multiplier':64,
          'optimizer_or_detector_enabled':False}

def feasible_directions(engine,p,g,parent):
    k=len(p);lower=engine.floor_mass/k;upper=min(1.,engine.cap_factor/k)
    def pair(donors,receivers):
        d=torch.zeros_like(p)
        for i in donors:
            for j in receivers:
                if i!=j:
                    mass=min(float(p[i])-lower,upper-float(p[j]))
                    if mass>1e-15:d[i]=-mass;d[j]=mass;return d
        return d
    donors=[i for i in range(k) if float(p[i])>lower+1e-15]
    receivers=[i for i in range(k) if float(p[i])<upper-1e-15]
    result={'projected_gradient':engine.project(p-g)-p,
      'steepest_feasible_pair':pair(sorted(donors,key=lambda i:-float(g[i])),sorted(receivers,key=lambda i:float(g[i]))),
      'toward_parent':parent-p,'uniform_reference':torch.ones_like(p)/k-p,
      'fixed_coordinate_pair':pair(donors,list(reversed(receivers)))}
    for d in result.values():
        if abs(float(d.sum()))>1e-10 or float((p+d).min())<lower-1e-12 or float((p+d).max())>upper+1e-12:
            raise ValueError('diagnostic direction violates original feasible segment')
    return result

@torch.no_grad()
@precise_matmul()
def geometry(engine,active,p):
    ids=torch.as_tensor(active,device=p.device,dtype=torch.long);r=engine.r[:,ids];rb=r.square()@p
    if bool((rb<=1e-20).any()):raise ValueError('zero reference rows')
    blocks=[];nearest=[];second=[];gaps=[];ties=[];near=[]
    for b,(allowed,_,_) in enumerate(engine.teacher):
        q=engine.q[b*engine.chunk:(b+1)*engine.chunk,ids];qa=q.square()@p
        if bool((qa<=1e-20).any()) or bool((allowed.sum(1)<2).any()):raise ValueError('two valid normal neighbors required')
        den=qa.sqrt()[:,None]*rb.sqrt()[None,:];cosine=(q*p)@r.T/den
        values,index=cosine.masked_fill(~allowed,-torch.inf).topk(2,dim=1)
        # Match the original max's deterministic branch, including exact ties.
        maximum,winner=cosine.masked_fill(~allowed,-torch.inf).max(1)
        gap=values[:,0]-values[:,1];exact=(cosine==maximum[:,None])&allowed
        threshold=64*torch.finfo(p.dtype).eps*(1+maximum.abs())
        nearest.append(winner);second.append(index[:,1]);gaps.append(gap);ties.append(exact.sum(1));near.append(((maximum[:,None]-cosine<=threshold[:,None])&allowed).sum(1))
        blocks.append((q,qa,den,cosine,maximum,winner,allowed,exact))
    arrays={'nearest':torch.cat(nearest),'second':torch.cat(second),'gap':torch.cat(gaps),
            'exact_tie_count':torch.cat(ties),'roundoff_near_tie_count':torch.cat(near)}
    return r,rb,blocks,arrays

@torch.no_grad()
@precise_matmul()
def independent_slopes(engine,p,d,geo,active):
    """Analytic directional derivative, independent of evaluate's gradient assembly.

    The exact-tie derivative uses the maximum directional cosine derivative of
    all observed *exactly* tied neighbors. Near ties are never promoted to ties.
    This floating point observation is not a Clarke/global optimality certificate.
    """
    r,rb,blocks,_=geo;distribution=p.new_zeros(());branch=p.new_zeros(());tied=p.new_zeros(())
    first_crossing=[]
    for b,(q,qa,den,cosine,maximum,winner,allowed,exact) in enumerate(blocks):
        derivative=(q*d)@r.T/den-.5*cosine*((q.square()@d/qa)[:,None]+(r.square()@d/rb)[None,:])
        weights=engine.weights[b*engine.chunk:(b+1)*engine.chunk];target=engine.teacher[b][1]
        logits=(cosine/engine.temperature).masked_fill(~allowed,-torch.inf)
        distribution+=(weights[:,None]*(logits.softmax(1)-target)*derivative).sum()/(len(engine.q)*engine.scale*engine.temperature)
        coefficient=2*weights*(maximum-engine.max_target[b*engine.chunk:(b+1)*engine.chunk])/(len(engine.q)*engine.score_scale)
        chosen=derivative.gather(1,winner[:,None]).squeeze(1)
        branch+=(coefficient*chosen).sum()
        tied+=(coefficient*derivative.masked_fill(~exact,-torch.inf).max(1).values).sum()
        closing=derivative-chosen[:,None];gap=maximum[:,None]-cosine
        valid=allowed&~exact&(closing>0)
        ratios=(gap/closing.clamp_min(torch.finfo(p.dtype).tiny)).masked_fill(~valid,torch.inf)
        first_crossing.append(float(ratios.min()))
    nuisance=p.new_zeros(())
    if engine.va is not None:
        ids=torch.as_tensor(active,device=p.device,dtype=torch.long)
        va,vb=engine.va[:,ids],engine.vb[:,ids];a=va.square()@p;b=vb.square()@p
        den=(a*b).sqrt();cosine=(va*vb)@p/den
        derivative=(va*vb)@d/den-.5*cosine*((va.square()@d/a)+(vb.square()@d/b))
        nuisance=-engine.beta*derivative.mean()/engine.vscale
    base=engine.distribution_weight*distribution+nuisance
    crossing=min(first_crossing)
    return {'independent_current_branch_slope':float(base+engine.score_weight*branch),
            'observed_exact_tie_directional_slope':float(base+engine.score_weight*tied),
            'linearized_first_neighbor_crossing':crossing if math.isfinite(crossing) else None}

@torch.no_grad()
@precise_matmul()
def diagnose_endpoint(engine,active,weights,parent_weights):
    p=torch.as_tensor(weights,device=engine.q.device,dtype=torch.float64).clone()
    parent=torch.as_tensor(parent_weights,device=p.device,dtype=p.dtype)
    if p.shape!=parent.shape or len(p)!=len(active) or not torch.isfinite(p).all() or not torch.isfinite(parent).all():raise ValueError('endpoint shape differs')
    if float((p-engine.project(p)).abs().max())>1e-10:raise ValueError('infeasible source endpoint')
    if float((parent-engine.project(parent)).abs().max())>1e-10:raise ValueError('infeasible frozen parent')
    initial=p.clone();loss,g,terms=engine.evaluate(active,p,True)
    geo=geometry(engine,active,p);arrays={k:v.cpu().numpy() for k,v in geo[3].items()}
    dirs=feasible_directions(engine,p,g,parent);rows=[];directions=[];calls=1
    for di,(name,d) in enumerate(dirs.items()):
        slope=float(g@d);slopes=independent_slopes(engine,p,d,geo,active)
        mismatch=abs(slope-slopes['independent_current_branch_slope'])
        if mismatch>1e-8*(1+abs(slope)):raise ValueError('independent current-branch derivative mismatch')
        zero=float(d.abs().max())<1e-18
        directions.append({'name':name,'delta_p':d.cpu().tolist(),'gradient_dot_direction':slope,'independent_derivative_error':mismatch,'zero_direction':zero,**slopes})
        if zero:continue
        for si,h in enumerate(STEPS):
            candidate=p+h*d;delta=candidate-p
            value,_,_=engine.evaluate(active,candidate,False);calls+=1
            observed=geometry(engine,active,candidate)[3];change=value-loss
            precision_scale=64*torch.finfo(p.dtype).eps*(1+abs(value)+abs(loss))
            arrays[f'd{di}_h{si}_nearest']=observed['nearest'].cpu().numpy()
            arrays[f'd{di}_h{si}_gap']=observed['gap'].cpu().numpy()
            rows.append({'direction':name,'step_fraction':h,'actual_loss':value,'loss_difference':change,'one_sided_quotient':change/h,
              'gradient_dot_actual_delta':float(g@delta),'branch_linear_prediction':h*slope,
              'observed_exact_tie_linear_prediction':h*slopes['observed_exact_tie_directional_slope'],
              'changed_neighbor_queries':int((observed['nearest']!=geo[3]['nearest']).sum()),
              'max_abs_p_delta':float(delta.abs().max()),'max_abs_mass_delta':float(delta.abs().max())*len(p),
              'sum_error':abs(float(candidate.sum())-1),'min_weight':float(candidate.min()),'max_weight':float(candidate.max()),
              'roundoff_scale':precision_scale,'difference_within_roundoff_scale':abs(change)<=precision_scale,
              'strict_armijo_pass':bool(change<0 and float(g@delta)<0 and value<=loss+1e-4*float(g@delta)),
              'candidate_equals_endpoint':bool(torch.equal(candidate,p))})
        print(f'[endpoint] {name}: {len(STEPS)} fixed probes',flush=True)
    repeat,rg,_=engine.evaluate(active,p,True);calls+=1
    if not torch.equal(initial,p) or abs(repeat-loss)>1e-12 or not torch.allclose(g,rg,rtol=1e-10,atol=1e-12):raise ValueError('read-only endpoint reproduction failed')
    gap=geo[3]['gap'];report={'version':VERSION,'active':list(active),'weights':p.cpu().tolist(),'gradient_p':g.cpu().tolist(),'loss':loss,'terms':terms,
      'projected_residual':float((p-engine.project(p-g)).abs().max()),'projection_displacement':float((p-engine.project(p)).abs().max()),
      'normal_queries':len(engine.q),'normal_references':len(engine.r),'gap_quantiles':torch.quantile(gap,torch.tensor([0.,.01,.1,.5,1.],device=p.device,dtype=p.dtype)).cpu().tolist(),
      'exact_tied_queries':int((geo[3]['exact_tie_count']>1).sum()),'roundoff_near_tied_queries':int((geo[3]['roundoff_near_tie_count']>1).sum()),
      'directions':directions,'probes':rows,'actual_objective_calls':calls,'geometry_passes':1+len(rows),
      'source_weights_modified':False,'optimizer_run':False,'test_data_used':False,'nonsmooth_stationarity_claimed':False}
    return report,arrays
