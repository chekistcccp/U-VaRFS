"""Clean normal relation selection; inference retains original coordinates only.

This is a new objective, not an equivalent solver for the frozen v22 objective.
No fitting API accepts labels, masks or outer audit features.
"""
from __future__ import annotations
import copy,time
import numpy as np
import torch
import torch.nn.functional as F
from .matching_uvarfs import MatchingObjective,_features,normal_temperature
from .local_uvarfs import LocalObjective
from .mechanism_validation import RANDOM_SEEDS,ROLES,jsonable,artifact_digest,common_refit,best_supports
from .pursuit_support import pursue_support
from .numerics import precise_matmul

VERSION='clean-relation-sparse-validation-v1'
SETTINGS=dict(evaluation_version=VERSION,feature_budget=256,random_seeds=RANDOM_SEEDS,normal_roles=ROLES,
    graph=dict(neighbors=8,resamples=3,coordinate_fraction=.5,minimum_votes=2,seed_offset=61019),
    common_weight_refit=dict(maximum=800,tol=1e-5),
    support_search=dict(starts=2,rounds=4,breakpoint_quantiles=[0.,.25,.5,.75],
                        breakpoint_multiplier=1.05,refit_max_iter=150,tol=1e-5,final_supports=2))

def scientific_digest(value):
    """Hash numerical choices, excluding runtime duration from repeatability."""
    def clean(v):
        if isinstance(v,dict):return {k:clean(x) for k,x in v.items() if k!='seconds'}
        if isinstance(v,(tuple,list)):return [clean(x) for x in v]
        return v
    return artifact_digest(clean(value))

def expected_methods():
    names={'relation_sparse','relation_sparse_uniform','relation_beta_zero','relation_beta_zero_uniform',
        'same_support_beta_zero','dense_weighted','dense_beta_zero','asls_raw','asls_pca','fixed4_raw',
        'all_raw','last_raw','previous_sparse_anchor','previous_dense_anchor','previous_beta_zero_anchor'}
    for seed in RANDOM_SEEDS:
        names.update(f'{p}_seed{seed}' for p in ['random_weighted','random_uniform','random_beta_zero','randomk_raw'])
    for baseline in ['laplacian','repeatability']:
        names.update(baseline+'_'+p for p in ['uniform','weighted','beta_zero'])
    return names

@torch.no_grad()
@precise_matmul()
def clean_graph(query,reference,qg,rg,settings,seed,chunk=256):
    """Cross-image neighbors reproducible under clean coordinate subsampling.

    Coordinate subsampling is a graph stability heuristic, not patient bootstrap.
    Rows without a reliable positive are excluded rather than invented.
    """
    qg,rg=_features(query,reference,qg,rg)
    n,k=query.shape[1],int(settings['neighbors'])
    if not 1<=settings['minimum_votes']<=settings['resamples'] or k<1 or not 0<settings['coordinate_fraction']<=1:
        raise ValueError('invalid clean graph settings')
    gen=np.random.default_rng(int(seed)+int(settings['seed_offset']))
    subsets=[np.sort(gen.choice(n,max(1,int(n*settings['coordinate_fraction'])),replace=False)).tolist()
             for _ in range(settings['resamples'])]
    q,r=F.normalize(query.double(),dim=1),F.normalize(reference.double(),dim=1)
    blocks=[]
    for start in range(0,len(q),chunk):
        allowed=qg[start:start+chunk,None]!=rg[None,:]
        kk=min(k,int(allowed.sum(1).min()))
        if kk<1:raise ValueError('graph needs other-image references')
        base=((q[start:start+chunk]@r.T).masked_fill(~allowed,-torch.inf)).argsort(dim=1,descending=True,stable=True)[:,:kk]
        votes=torch.zeros_like(base)
        for coords in subsets:
            qq=F.normalize(query[start:start+chunk,coords].double(),dim=1)
            rr=F.normalize(reference[:,coords].double(),dim=1)
            ids=((qq@rr.T).masked_fill(~allowed,-torch.inf)).argsort(dim=1,descending=True,stable=True)[:,:kk]
            votes+=(base[:,:,None]==ids[:,None,:]).any(2)
        positive=torch.zeros_like(allowed,dtype=torch.float64)
        weights=torch.where(votes>=settings['minimum_votes'],votes.double()/settings['resamples'],0.)
        positive.scatter_(1,base,weights)
        blocks.append(positive)
    weights=torch.cat(blocks);valid=weights.sum(1)>0
    if not bool(valid.any()):raise ValueError('no reproducible clean graph positives')
    info=dict(settings=copy.deepcopy(settings),coordinate_subsets=subsets,valid_query_indices=valid.nonzero().flatten().tolist(),
              valid_queries=int(valid.sum()),total_queries=len(query),positive_edges=int((weights>0).sum()),
              same_image_edges=int(((weights>0)&(qg[:,None]==rg[None,:])).sum()),
              positive_weights_sha256=artifact_digest(weights),nuisance_used=False)
    return weights,info

class RelationObjective(MatchingObjective):
    """Smooth multi-positive normal relation loss and actual paired nuisance.

    Graph is fixed before fitting. Positive weights also enter the denominator,
    giving a probability ratio <=1. Every other-image patch is in its denominator.
    """
    def __init__(self,q,r,qg,rg,temperature,graph,paired=None,beta=.002,chunk=256,floor_mass=1e-4,cap_factor=4.):
        self.qg,self.rg=_features(q,r,qg,rg)
        if not np.isfinite([temperature,beta,floor_mass,cap_factor]).all() or temperature<=0 or beta<0 or chunk<1 or not 0<floor_mass<=1 or cap_factor<1:
            raise ValueError('invalid relation parameters')
        self.q,self.r=q.double(),r.double();self.temperature=float(temperature);self.beta=float(beta)
        self.chunk=int(chunk);self.floor_mass=float(floor_mass);self.cap_factor=float(cap_factor)
        allowed=self.qg[:,None]!=self.rg[None,:]
        self.graph=torch.as_tensor(graph,device=q.device,dtype=torch.float64).clone()
        if self.graph.shape!=allowed.shape or not torch.isfinite(self.graph).all() or (self.graph<0).any() or (self.graph>1).any() or (self.graph[~allowed]!=0).any():
            raise ValueError('invalid clean relation graph')
        self.valid=self.graph.sum(1)>0
        if not self.valid.any():raise ValueError('no valid clean positives')
        self.va=self.vb=None;self.vscale=q.new_tensor(1.,dtype=torch.float64)
        # beta0 never indexes or validates paired nuisance, including starts/graph.
        if beta:
            if paired is None:raise ValueError('actual nuisance pairs required')
            clean,pert=paired
            if clean.ndim!=2 or pert.ndim!=3 or pert.shape[1:]!=clean.shape or clean.shape[1]!=q.shape[1] or not len(clean) or not len(pert) or not torch.isfinite(clean).all() or not torch.isfinite(pert).all():
                raise ValueError('invalid actual nuisance pairs')
            self.va=clean.double().repeat(len(pert),1);self.vb=pert.double().reshape(-1,q.shape[1])
            if (self.va.norm(dim=1)<=1e-12).any() or (self.vb.norm(dim=1)<=1e-12).any():raise ValueError('zero dense nuisance rows')
            self.vscale=(1-F.cosine_similarity(self.va,self.vb)).mean().clamp_min(1e-4)
        self.evaluations=0

    @torch.no_grad()
    @precise_matmul()
    def evaluate(self,active,p,gradient=False,full_gradient=False):
        active=torch.as_tensor(active,device=self.q.device,dtype=torch.long)
        p=torch.as_tensor(p,device=self.q.device,dtype=torch.float64)
        if active.ndim!=1 or p.shape!=active.shape or not len(active) or len(active.unique())!=len(active) or active.min()<0 or active.max()>=self.q.shape[1] or not torch.isfinite(p).all() or (p<0).any():
            raise ValueError('invalid relation support/weights')
        rx=self.r[:,active];rb=rx.square()@p
        if (rb<=1e-20).any():raise ValueError('zero sparse reference rows')
        if ((self.q[:,active].square()@p)<=1e-20).any():raise ValueError('zero sparse query rows')
        gx=self.r if full_gradient else rx
        grad=torch.zeros(gx.shape[1],device=p.device,dtype=p.dtype) if gradient else None
        loss=p.new_zeros(());count=int(self.valid.sum())
        for start in range(0,len(self.q),self.chunk):
            valid=self.valid[start:start+self.chunk]
            if not valid.any():continue
            q0=self.q[start:start+self.chunk][valid];qx=q0[:,active];qa=qx.square()@p
            if (qa<=1e-20).any():raise ValueError('zero sparse query rows')
            den=qa.sqrt()[:,None]*rb.sqrt()[None,:];cosine=(qx*p)@rx.T/den
            graph=self.graph[start:start+self.chunk][valid];positive=graph>0
            allowed=self.qg[start:start+self.chunk][valid,None]!=self.rg[None,:]
            logweight=torch.where(positive,graph.clamp_min(1e-300).log(),torch.zeros_like(graph))
            logits=(cosine/self.temperature+logweight).masked_fill(~allowed,-torch.inf)
            pos=logits.masked_fill(~positive,-torch.inf)
            loss+=(torch.logsumexp(logits,1)-torch.logsumexp(pos,1)).sum()/count
            if gradient:
                coeff=(logits.softmax(1)-pos.softmax(1))/(count*self.temperature)
                qgrad=q0 if full_gradient else qx;products=coeff*cosine
                grad+=(qgrad*((coeff/den)@gx)).sum(0)
                grad-=.5*(qgrad.square()*(products.sum(1)/qa)[:,None]).sum(0)
                grad-=.5*(gx.square()*(products.sum(0)/rb)[:,None]).sum(0)
        var=p.new_zeros(())
        if self.va is not None:
            if full_gradient:
                pp=torch.zeros(self.q.shape[1],device=p.device,dtype=p.dtype);pp[active]=p;va,vb=self.va,self.vb
            else:pp,va,vb=p,self.va[:,active],self.vb[:,active]
            cosine,jac=LocalObjective.cosine(va,vb,pp,gradient);var=(1-cosine).mean()/self.vscale
            if gradient:grad-=self.beta*jac.mean(0)/self.vscale
        self.evaluations+=1
        return float(loss+self.beta*var),grad,dict(normal_relation_loss=float(loss),variability_term=float(self.beta*var),relative_variability=float(var))

    def refit(self,active,initial,maximum=150,tol=1e-5):
        before=self.evaluations;started=time.perf_counter()
        p,value,terms,diag=super().refit(active,initial,maximum,tol)
        diag.update(loss_gradient_and_value_calls=self.evaluations-before,seconds=time.perf_counter()-started,
                    stop_status='residual_tolerance' if diag['residual']<=tol else
                        ('iteration_budget' if diag['iterations']>=maximum else 'line_search_stalled'),
                    residual_scope='smooth_fixed_graph_fixed_support_projected_gradient_not_global')
        return p,value,terms,diag

@torch.no_grad()
def rankings(x,groups,graph):
    """Classic Laplacian score and clean split variance repeatability controls."""
    w=(graph+graph.T)/2;degree=w.sum(1);total=degree.sum()
    if total<=0:raise ValueError('no clean relation edges')
    xx=x.double();center=xx-(degree[:,None]*xx).sum(0)/total
    variance=(degree[:,None]*center.square()).sum(0)
    smooth=(degree[:,None]*xx.square()).sum(0)-(xx*(w@xx)).sum(0)
    lap=smooth/variance.clamp_min(1e-12);lap[variance<=1e-12]=torch.inf
    ids=groups.unique(sorted=True);a=xx[torch.isin(groups,ids[::2])].var(0,unbiased=False)
    b=xx[torch.isin(groups,ids[1::2])].var(0,unbiased=False)
    repeat=(a+b)/((a-b).abs()+.05*(a+b).mean().clamp_min(1e-12))
    return lap,repeat

@torch.no_grad()
def complementary_support(x,score,k):
    """Fixed relevance minus maximum squared clean coordinate correlation."""
    z=x.double()-x.double().mean(0);z=F.normalize(z,dim=0)
    order=score.argsort(descending=True,stable=True);relevance=torch.empty_like(score)
    relevance[order]=torch.linspace(1.,0.,len(score),device=x.device,dtype=torch.float64)
    redundancy=torch.zeros_like(score);chosen=[];remaining=torch.ones(len(score),device=x.device,dtype=torch.bool)
    for _ in range(k):
        j=int((relevance-.5*redundancy).masked_fill(~remaining,-torch.inf).argmax())
        chosen.append(j);remaining[j]=False;redundancy=torch.maximum(redundancy,(z.T@z[:,j]).square())
    return sorted(chosen)

def fit_branch(x,val,g,vg,paired,settings,seed,beta):
    tau=normal_temperature(x,g);graph,graphinfo=clean_graph(x,x,g,g,settings['graph'],seed)
    engine=RelationObjective(x,x,g,g,tau,graph,paired if beta else None,beta)
    k=min(settings['feature_budget'],x.shape[1]);lap,repeat=rankings(x,g,graph)
    starts={'laplacian':lap.argsort(stable=True)[:k].sort().values.tolist(),
            'repeatability':repeat.argsort(descending=True,stable=True)[:k].sort().values.tolist(),
            'complementary':complementary_support(x,-lap,k)}
    pool=[]
    def record(active,p,name,diag=None):
        value,_,terms=engine.evaluate(active,p);i=len(pool)
        pool.append(dict(candidate_id=i,active=jsonable(active),weights=jsonable(p),source=name,
                         training_objective=value,training_terms=terms,diagnostics=jsonable(diag)))
        return i
    common=settings['common_weight_refit']
    for name,ids in starts.items():
        c=common_refit(engine,ids,common);record(c['active'],c['weights'],name,c['diagnostics'])
    search=settings['support_search'];attempts=[]
    for parent in best_supports(pool,int(search['starts'])):
        _,_,info=pursue_support(engine,parent['active'],parent['weights'],search,record,print)
        attempts.append(dict(parent_candidate_id=parent['candidate_id'],**info))
    learned=[common_refit(engine,c['active'],common) for c in best_supports(pool,search['final_supports'])]
    random={s:common_refit(engine,np.sort(np.random.default_rng(s).choice(x.shape[1],k,replace=False)),common) for s in RANDOM_SEEDS}
    baseline={name:common_refit(engine,starts[name],common) for name in ['laplacian','repeatability']}
    dense=common_refit(engine,range(x.shape[1]),common)
    frozen=scientific_digest(dict(pool=pool,learned=learned,random=random,baseline=baseline,dense=dense))
    # Selection values become accessible only after all training candidates freeze.
    vgraph,vinfo=clean_graph(val,x,vg,g,settings['graph'],seed)
    validator=RelationObjective(val,x,vg,g,tau,vgraph,beta=0.)
    for c in [*learned,*random.values(),*baseline.values(),dense]:
        value,_,terms=validator.evaluate(c['active'],c['weights'])
        c.update(selection_objective=value,selection_terms=terms,selection_score=value+c['training_terms']['variability_term'])
    winner=min(range(len(learned)),key=lambda i:(learned[i]['selection_score'],i))
    return dict(beta=beta,temperature_fit_only=tau,graph=graphinfo,selection_graph=vinfo,starts=starts,support_pool=pool,
        search_attempts=attempts,learned=learned,random=random,baseline=baseline,dense=dense,winner_index=winner,
        fit_only_sha256=frozen,graph_weights_sha256=artifact_digest(graph),fit_evaluations=engine.evaluations,
        selection_used_only_after_full_fit=True,beta_zero_reads_nuisance=False),engine

def fit_design(training,selection,g,vg,paired,layers,oldspecs,settings,seed):
    started=time.perf_counter();join=lambda data:torch.cat([data[l] for l in layers],-1)
    x,val=join(training),join(selection);branches={};specs={}
    def spec(c,uniform=False):
        return dict(layers=layers,layer_normalization='l2',kind='selected_raw' if uniform else 'uvarfs',
                    obj=dict(active=c['active']) if uniform else dict(active=c['active'],scales=np.sqrt(c['weights']).tolist()))
    for label,beta in [('nuisance',.002),('beta_zero',0.)]:
        # Defer nuisance feature access until the beta>0 branch.
        pairs=(join(paired[0]),join(paired[1])) if beta else None
        b,engine=fit_branch(x,val,g,vg,pairs,settings,seed,beta);branches[label]=b
        c=b['learned'][b['winner_index']];name='relation_sparse' if beta else 'relation_beta_zero'
        specs[name]=spec(c);specs[name+'_uniform']=spec(c,True)
        specs['dense_weighted' if beta else 'dense_beta_zero']=spec(b['dense'])
        for s,c in b['random'].items():
            specs[f'{"random_weighted" if beta else "random_beta_zero"}_seed{s}']=spec(c)
            if beta:specs[f'random_uniform_seed{s}']=spec(c,True)
        for n,c in b['baseline'].items():
            specs[n+('_weighted' if beta else '_beta_zero')]=spec(c)
            if beta:specs[n+'_uniform']=spec(c,True)
        if not beta:
            active=specs['relation_sparse']['obj']['active']
            c=common_refit(engine,active,settings['common_weight_refit'])
            branches['same_support_beta_zero']=c;specs['same_support_beta_zero']=spec(c)
    if branches['nuisance']['graph_weights_sha256']!=branches['beta_zero']['graph_weights_sha256']:
        raise ValueError('beta branches must share exact clean graph')
    for name in ['asls_raw','asls_pca','fixed4_raw','all_raw','last_raw',*[f'randomk_raw_seed{s}' for s in RANDOM_SEEDS]]:
        specs[name]=copy.deepcopy(oldspecs[name])
    for new,old in [('previous_sparse_anchor','mechanism_sparse'),('previous_dense_anchor','dense_weighted'),('previous_beta_zero_anchor','mechanism_beta_zero')]:
        specs[new]=copy.deepcopy(oldspecs[old])
    if set(specs)!=expected_methods():raise ValueError('fixed relation method roster differs')
    return jsonable(dict(specs=specs,branches=branches,asls_layers_frozen=layers,new_fit_seconds=time.perf_counter()-started))
