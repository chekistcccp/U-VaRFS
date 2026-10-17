"""Normal-only factorial controls, with no audit input in the fitting API."""
from __future__ import annotations
import copy,hashlib,json,time
import numpy as np
import torch
from .local_uvarfs import fit_local_asls
from .matching_uvarfs import generate_fit_only_pool,MatchingObjective
from .pursuit_support import pursue_support,digest
from .transforms import fit_pca

VERSION='mechanism-factorial-audit-v1'
RANDOM_SEEDS=[42,123,3407,2026,2027]
ROLES={'fit_images':144,'selection_images':48,'audit_images':64,'nuisance_images':96,'patches_per_image':16}
SETTINGS={'evaluation_version':VERSION,'feature_budget':256,'random_seeds':RANDOM_SEEDS,'normal_roles':ROLES,
          'common_weight_refit':{'maximum':800,'tol':1e-5},
          'support_search':{'starts':2,'rounds':4,'breakpoint_quantiles':[0.,.25,.5,.75],
                            'breakpoint_multiplier':1.05,'refit_max_iter':150,'tol':1e-5,'final_supports':2}}

def jsonable(x):
    if isinstance(x,(torch.Tensor,np.ndarray)):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {str(k):jsonable(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [jsonable(v) for v in x]
    return x

def artifact_digest(value):
    return hashlib.sha256(json.dumps(jsonable(value),sort_keys=True,allow_nan=False).encode()).hexdigest()

def expected_methods():
    result={'mechanism_sparse','mechanism_sparse_uniform','mechanism_beta_zero','mechanism_beta_zero_uniform',
            'dense_weighted','dense_beta_zero','asls_raw','asls_pca','fixed4_raw','all_raw','last_raw',
            'v22_anchor','v22_beta_zero_anchor','pursuit_anchor','pursuit_beta_zero_anchor'}
    for s in RANDOM_SEEDS:
        result.update(f'{prefix}_seed{s}' for prefix in ['random_weighted','random_uniform','random_beta_zero','randomk_raw'])
    return result

def partition(source,seed,roles=None):
    """Only image identities determine roles; audit feature values are not read."""
    roles=dict(ROLES if roles is None else roles);g=source['groups'];ag=source['heldout_groups']
    ids=g.unique(sorted=True).cpu().tolist();aids=ag.unique(sorted=True).cpu().tolist()
    nfit,nsel,nvar,ppi=[roles[k] for k in ['fit_images','selection_images','nuisance_images','patches_per_image']]
    if len(ids)!=nfit+nsel or len(aids)!=roles['audit_images'] or set(ids)&set(aids) or not 1<nvar<=nfit:
        raise ValueError('normal image-role coverage differs')
    if any(int((g==i).sum())!=ppi for i in ids) or any(int((ag==i).sum())!=ppi for i in aids):
        raise ValueError('normal image patch budget differs')
    paired_rows=len(next(iter(source['paired_clean'].values())))
    if paired_rows!=nvar*ppi or g[:paired_rows].unique(sorted=True).tolist()!=ids[:nvar]:
        raise ValueError('paired nuisance rows must belong to initial fitting images')
    remaining=np.asarray(ids[nvar:]);np.random.default_rng(int(seed)+27017).shuffle(remaining)
    selected=sorted(remaining[:nsel].tolist());fitted=sorted(set(ids)-set(selected))
    if len(fitted)!=nfit or not set(ids[:nvar])<=set(fitted):raise ValueError('nuisance selection leakage')
    train=torch.tensor([int(v) in set(fitted) for v in g.cpu().tolist()],device=g.device)
    select=~train
    info={'fit_image_positions':fitted,'selection_image_positions':selected,'audit_image_positions':aids,
          'nuisance_image_positions':ids[:nvar], 'roles':roles,'recipe':'all original nuisance images fit; remaining shuffle seed+27017',
          'audit_used_for_asls_or_feature_selection':False,'historically_exposed_normal_images':True,'patient_disjoint_claimed':False}
    return train,select,info

@torch.no_grad()
def common_refit(engine,active,settings):
    """Identical weight solver and uniform initialization for every support family."""
    active=torch.as_tensor(active,device=engine.q.device,dtype=torch.long)
    uniform=torch.full((len(active),),1/len(active),device=engine.q.device,dtype=torch.float64)
    initial=engine.evaluate(active,uniform)[0];started=time.perf_counter()
    print(f'[mechanism:common] uniform -> refit K={len(active)}',flush=True)
    p,value,terms,diag=engine.refit(active,uniform,int(settings['maximum']),float(settings['tol']))
    if value>initial+1e-10:raise ValueError('common refit increased actual training loss')
    print(f'[mechanism:common] done K={len(active)} objective={value:.8g} seconds={time.perf_counter()-started:.1f}',flush=True)
    return {'active':active.cpu().tolist(),'weights':p.cpu().tolist(),'training_objective':value,'training_terms':terms,
            'diagnostics':jsonable(diag),'initial_uniform_objective':initial,'seconds':time.perf_counter()-started,
            'initialization':'uniform','weight_solver':'original_bounded_simplex_BB_Armijo'}

def best_supports(pool,count):
    unique={}
    for c in pool:
        key=tuple(c['active'])
        if key not in unique or (c['training_objective'],c['candidate_id'])<(unique[key]['training_objective'],unique[key]['candidate_id']):unique[key]=c
    return sorted(unique.values(),key=lambda c:(c['training_objective'],c['candidate_id']))[:count]

@torch.no_grad()
def fit_branch(x,selection,g,sg,paired,cfg,settings,seed):
    cfg=copy.deepcopy(cfg);cfg['matching_polish_max_iter']=0
    engine,pool,trace=generate_fit_only_pool(x,g,paired if cfg['beta'] else None,cfg,'mechanism:support')
    base_count=len(pool);base_hash=digest(pool);search=settings['support_search'];attempts=[]
    def record(active,p,name,diag=None):
        value,_,terms=engine.evaluate(active,p)
        pool.append(dict(candidate_id=len(pool),source=name,active=active.cpu().tolist(),weights=p.cpu().tolist(),
                         training_objective=value,training_terms=terms,diagnostics=jsonable(diag)))
        return len(pool)-1
    for parent in best_supports(pool,int(search['starts'])):
        _,_,result=pursue_support(engine,parent['active'],parent['weights'],search,record,print)
        attempts.append(dict(parent_candidate_id=parent['candidate_id'],**result))
    if digest(pool[:base_count])!=base_hash:raise ValueError('fresh original support pool changed')
    supports=best_supports(pool,int(search['final_supports']))
    learned=[common_refit(engine,c['active'],settings['common_weight_refit']) for c in supports]
    k=min(int(cfg['max_features']),x.shape[1])
    random={s:common_refit(engine,np.sort(np.random.default_rng(s).choice(x.shape[1],k,replace=False)),settings['common_weight_refit']) for s in RANDOM_SEEDS}
    dense=common_refit(engine,range(x.shape[1]),settings['common_weight_refit'])
    # Nothing above has access to selection feature values through the engine.
    frozen=artifact_digest({'support_pool':pool,'learned':learned,'random':random,'dense':dense})
    validation=MatchingObjective(selection,x,sg,g,engine.temperature,beta=0.,chunk=engine.chunk,
                                floor_mass=engine.floor_mass,cap_factor=engine.cap_factor)
    for c in [*learned,*random.values(),dense]:
        value,_,terms=validation.evaluate(c['active'],c['weights'])
        c.update(selection_objective=value,selection_terms=terms,selection_score=value+c['training_terms']['variability_term'])
    winner=min(range(len(learned)),key=lambda i:(learned[i]['selection_score'],i))
    return dict(beta=engine.beta,temperature_fit_only=engine.temperature,support_pool=pool,
                base_pool_sha256=base_hash,base_pool_count=base_count,search_attempts=attempts,
                fit_only_artifacts_sha256=frozen,support_pool_fit_only_sha256=digest(pool),learned=learned,random=random,dense=dense,winner_index=winner,
                floor_mass=engine.floor_mass,cap_factor=engine.cap_factor,
                all_weight_refits_from_uniform=True,selection_used_only_after_full_fit=True,
                audit_used_for_fitting=False,global_optimality_claimed=False)

def fit_design(training,selection,g,sg,paired,base_cfg,settings,seed):
    """No outer audit, old ASLS, old support solution, labels or masks argument."""
    started=time.perf_counter()
    asls=fit_local_asls(training,selection,g,sg,paired,base_cfg['asls']);layers=asls['selected_layers']
    join=lambda data:torch.cat([data[l] for l in layers],-1)
    x,val=join(training),join(selection);pairs=(join(paired[0]),join(paired[1]))
    branches={};specs={}
    def spec(c,uniform=False):
        return dict(layers=layers,layer_normalization='l2',kind='selected_raw' if uniform else 'uvarfs',
                    obj={'active':c['active']} if uniform else {'active':c['active'],'scales':np.sqrt(c['weights']).tolist()})
    for label,beta in [('nuisance',.002),('beta_zero',0.)]:
        cfg=copy.deepcopy(base_cfg['uvarfs']);cfg['beta']=beta
        branch=fit_branch(x,val,g,sg,pairs if beta else None,cfg,settings,seed);branches[label]=branch
        c=branch['learned'][branch['winner_index']];name='mechanism_sparse' if beta else 'mechanism_beta_zero'
        specs[name]=spec(c);specs[name+'_uniform']=spec(c,True)
        specs['dense_weighted' if beta else 'dense_beta_zero']=spec(branch['dense'])
        for s,c in branch['random'].items():
            specs[f'{"random_weighted" if beta else "random_beta_zero"}_seed{s}']=spec(c)
            if beta:specs[f'random_uniform_seed{s}']=spec(c,True)
    all_layers=sorted(training)
    for name,ls in [('asls_raw',layers),('all_raw',all_layers),('fixed4_raw',[3,6,9,12]),('last_raw',[all_layers[-1]])]:
        if not set(ls)<=set(all_layers):raise ValueError('expected DINO hierarchy missing')
        specs[name]=dict(layers=ls,layer_normalization='l2',kind='raw')
    for s in RANDOM_SEEDS:
        ls=sorted(np.random.default_rng(s).choice(all_layers,len(layers),replace=False).tolist())
        specs[f'randomk_raw_seed{s}']=dict(layers=ls,layer_normalization='l2',kind='raw')
    specs['asls_pca']=dict(layers=layers,layer_normalization='l2',kind='pca',obj=jsonable(fit_pca(x,min(256,x.shape[1]),seed)))
    return jsonable(dict(asls=asls,branches=branches,specs=specs,new_fit_seconds=time.perf_counter()-started))
