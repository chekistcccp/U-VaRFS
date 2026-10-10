from __future__ import annotations
import hashlib
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from .data import BMADDataset
from .perturb import perturb_batch, PERTURBATIONS
from .asls import fit_asls, reselect_asls
from .u_varfs import fit_uvarfs
from .cosine_uvarfs import fit_cosine_uvarfs
from .performance_uvarfs import fit_performance_uvarfs, fit_fixed_budget_reference
from .objectives import COSINE, QUADRATIC, PERFORMANCE, LOCAL, objective_modes, experiment_branches, methods_for_objective
from .local_uvarfs import fit_local_uvarfs, fit_local_asls
from .transforms import fit_pca
from .utils import save_json
from .variability import NormalVariability
from .geometry_audit import audit_normal_geometry, describe_norms
from .representation import normalization_modes, normalization_prefix, normalize_layers


def _sample_indices(n, k, seed):
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=min(n, k), replace=False))


def _patch_ids(p, k, seed):
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(p, size=min(p, k), replace=False))


def perturbation_seed(seed, dataset, image_offset):
    text=f'uvarfs-normal-noise-v1:{int(seed)}:{dataset}:{int(image_offset)}'
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8],'little') % (2**63)


def collect_normal_inputs(extractor, train_samples, cfg):
    if not train_samples or any(s.label != 0 or s.split != 'train' for s in train_samples):
        raise ValueError('ASLS/U-VaRFS fit requires the official normal-reference training split')
    seed = int(cfg["seed"])
    fit_n = int(cfg["data"]["fit_images"])
    var_n = int(cfg["data"]["variability_images"])
    ppi = int(cfg["data"]["patches_per_image"])
    ids = _sample_indices(len(train_samples), fit_n, seed)
    local_protocol = cfg['asls'].get('discrete_selection') == 'normal_local'
    if LOCAL in objective_modes(cfg)[1] and not local_protocol:
        raise ValueError('normal-local U-VaRFS requires the image-disjoint normal-local ASLS protocol')
    holdout_n = 0
    if local_protocol:
        # Randomize image order before reserving holdout. All existing nuisance
        # forwards stay in the fitting partition, within the original 256 images.
        np.random.default_rng(seed+1907).shuffle(ids)
        fraction=float(cfg['data'].get('normal_holdout_fraction',.25))
        if not 0<fraction<1:
            raise ValueError('normal holdout fraction must lie strictly between zero and one')
        holdout_n = max(1, int(round(len(ids)*fraction)))
        if len(ids)-holdout_n < max(2,min(var_n,len(ids))):
            raise ValueError('normal holdout leaves too few fit/variability images')
    fit_count = len(ids)-holdout_n
    ds = BMADDataset(train_samples, int(cfg["model"]["input_size"]), False)
    dl = DataLoader(
        Subset(ds, ids.tolist()),
        batch_size=int(cfg["model"]["batch_size"]), shuffle=False,
        num_workers=int(cfg["model"]["num_workers"]), pin_memory=torch.cuda.is_available(),
    )
    layers = range(1, extractor.num_layers + 1)
    _,modes=normalization_modes(cfg)
    pooled = {mode:{i: [] for i in layers} for mode in modes}
    patchbuf = {i: [] for i in layers}
    perturbbuf = {kind:{i:[] for i in layers} for kind in PERTURBATIONS}
    patch_groups=[]
    statistics = {mode:{i: NormalVariability(extractor.hidden_dim, PERTURBATIONS,
                    float(cfg['data'].get('variability_epsilon',1e-6))) for i in layers} for mode in modes}
    patch_manifest = []
    noise_manifest=[]
    global_pos = 0
    for batch in tqdm(dl, desc="fit-features", leave=False):
        x = batch["image"]
        feats = extractor(x)
        views={mode:normalize_layers(feats,mode) for mode in modes}
        b = x.shape[0]
        p = next(iter(feats.values())).shape[1]
        patch_ids = _patch_ids(p, ppi, seed + global_pos)
        patch_groups.extend(np.repeat(np.arange(global_pos,global_pos+b),len(patch_ids)).tolist())
        patch_manifest.append({'image_offset':global_pos,'batch_images':b,'patch_indices':patch_ids.tolist()})
        for l in layers:
            patchbuf[l].append(feats[l][:, patch_ids, :].reshape(-1, extractor.hidden_dim).cpu())
        for mode,features in views.items():
            pf=extractor.pooled(features)
            for l in layers:
                pooled[mode][l].append(pf[l].cpu())
        if global_pos < var_n:
            take = min(b, var_n - global_pos)
            xvar = x[:take].cuda(non_blocking=True) if torch.cuda.is_available() else x[:take]
            base = {mode:{l:features[l][:take] for l in layers} for mode,features in views.items()}
            noise_seed=perturbation_seed(seed,train_samples[0].dataset,global_pos)
            generator=torch.Generator(device=xvar.device).manual_seed(noise_seed)
            noise_manifest.append({'image_offset':global_pos,'batch_images':take,'seed':noise_seed})
            for mode in modes:
                for l in layers:
                    statistics[mode][l].add_normal(base[mode][l])
            for kind in PERTURBATIONS:
                pert = extractor(perturb_batch(xvar, kind, generator=generator))
                if local_protocol:
                    for l in layers:
                        perturbbuf[kind][l].append(pert[l][:,patch_ids,:].reshape(-1,extractor.hidden_dim).cpu())
                for mode in modes:
                    view=normalize_layers(pert,mode)
                    for l in layers:
                        statistics[mode][l].add_perturbation(kind,base[mode][l],view[l])
        global_pos += b
    device = "cuda" if torch.cuda.is_available() else "cpu"
    raw_patches = {l: torch.cat(v).to(device) for l, v in patchbuf.items()}
    manifest={'normal_train_indices':ids.tolist(),
               'normal_train_paths':[str(train_samples[int(i)].image) for i in ids],
               'fit_images':len(ids),'variability_images':min(var_n,len(ids)),
               'patches_per_image':ppi,'sampled_patches':patch_manifest,
               'variability_definition':'global_mean_squared_delta/(global_population_variance+epsilon)',
               'variability_epsilon':float(cfg['data'].get('variability_epsilon',1e-6)),
               'asls_geometry_representation':cfg['asls'].get('geometry_representation','patch'),
               'perturbations':list(PERTURBATIONS),
               'backbone_raw_patch_norms':{str(l):describe_norms(v) for l,v in raw_patches.items()},
               'perturbation_rng':{'recipe':'sha256:uvarfs-normal-noise-v1:seed:dataset:image_offset',
                                   'base_seed':seed,'dataset':train_samples[0].dataset,'noise_batches':noise_manifest}}
    collected={}
    groups=torch.tensor(patch_groups,device=device)
    fitting=groups<fit_count
    if local_protocol:
        manifest['normal_selection_split']={
            'unit':'image','patient_disjoint_claimed':False,
            'recipe':'shuffle sampled image indices with seed+1907; reserve final fraction',
            'fit_image_positions':list(range(fit_count)),
            'holdout_image_positions':list(range(fit_count,len(ids))),
            'fit_patch_groups':groups[fitting].cpu().tolist(),
            'holdout_patch_groups':groups[~fitting].cpu().tolist(),
            'fit_patch_rows':int(fitting.sum()),'holdout_patch_rows':int((~fitting).sum()),
            'holdout_used_for_gradients':False,'holdout_used_for_selection_reference':False,
            'final_memory_uses_official_normal_train_after_selection':True}
    for mode in modes:
        normalized=normalize_layers(raw_patches,mode)
        collected[mode]={'patches':{l:v[fitting] if local_protocol else v for l,v in normalized.items()},
                         'pooled':{l:torch.cat(v).to(device)[:fit_count] for l,v in pooled[mode].items()},
                         'variabilities':{l:statistics[mode][l].vectors().to(device) for l in layers},
                         'manifest':manifest}
        if local_protocol:
            pert={l:torch.stack([normalize_layers({l:torch.cat(perturbbuf[kind][l]).to(device)},mode)[l]
                                for kind in PERTURBATIONS]) for l in layers}
            paired_count=next(iter(pert.values())).shape[1]
            collected[mode].update(heldout={l:v[~fitting] for l,v in normalized.items()},
                groups=groups[fitting],heldout_groups=groups[~fitting],
                paired_clean={l:v[:paired_count] for l,v in normalized.items()},paired_perturbations=pert)
            manifest['normal_selection_split']['paired_patch_rows']=paired_count
    return collected


def fit_representation(extractor, train_samples, cfg, dataset_out, collected=None):
    if not train_samples or any(s.label != 0 or s.split != 'train' for s in train_samples):
        raise ValueError('ASLS/U-VaRFS fit requires the official normal-reference training split')
    mode,_=normalization_modes(cfg)
    collected=collect_normal_inputs(extractor,train_samples,cfg) if collected is None else collected
    inputs=collected[mode]
    patches,pooled,var_vectors=inputs['patches'],inputs['pooled'],inputs['variabilities']
    save_json({**inputs['manifest'],'layer_normalization':mode,
               'variability_feature_space':mode,
               'variability_vectors':{str(l):v.cpu().tolist() for l,v in var_vectors.items()}},dataset_out/'fit_manifest.json')
    layer_var = {l: float(var_vectors[l].mean().cpu()) for l in pooled}
    comparison_methods=[m for m in cfg['methods'] if any(m.startswith('asls_'+name+'_') for name in ['pooled','patch','geometry_search','gate_prefix'])]
    cache_key=json.dumps([cfg['asls'],comparison_methods],sort_keys=True)
    if inputs.get('asls_cache_key')==cache_key:
        for name,result in inputs['asls_artifacts'].items():
            save_json(result,dataset_out/name)
        return patches,var_vectors,inputs['asls_artifacts']['asls.json']
    representation=cfg['asls'].get('geometry_representation','patch')
    inputs={'pooled':pooled,'patch':patches}
    print(f"[fit] ASLS start: geometry={representation} layers={len(pooled)} samples={next(iter(inputs[representation].values())).shape[0]}", flush=True)
    legacy_cfg={**cfg['asls'],'discrete_selection':'geometry_search'}
    local_protocol=cfg['asls'].get('discrete_selection')=='normal_local'
    legacy_asls=fit_asls(inputs[representation],layer_var,legacy_cfg if local_protocol else cfg['asls'])
    if local_protocol:
        source=collected[mode]
        asls=fit_local_asls(patches,source['heldout'],source['groups'],source['heldout_groups'],
            (source['paired_clean'],source['paired_perturbations']),cfg['asls'])
        save_json(legacy_asls,dataset_out/'asls_previous.json')
    else:
        asls=legacy_asls
    asls['layer_normalization']=mode
    comparisons={representation:asls}
    for other in ['pooled','patch']:
        if other!=representation and any(method.startswith(f'asls_{other}_') for method in cfg['methods']):
            print(f'[fit] ASLS {other} geometry ablation',flush=True)
            comparisons[other]=fit_asls(inputs[other],layer_var,{**legacy_cfg,'geometry_representation':other})
    for geometry_mode,result in comparisons.items():
        save_json(result,dataset_out/f'asls_{geometry_mode}.json')
    # Keep only selection metadata here; optimizer traces live in separate files.
    asls['geometry_comparisons']={mode:{k:result[k] for k in ['selected_layers','geometry_error','feasible','geometry_representation','discrete_selection']}
                                  for mode,result in comparisons.items()}
    asls['selection_comparisons']={}
    for rule in ['gate_prefix','geometry_search']:
        if any(method.startswith(f'asls_{rule}_') for method in cfg['methods']):
            selection=reselect_asls(legacy_asls,legacy_cfg,rule)
            print(f'[fit] ASLS {rule} ablation: evaluated={selection["evaluated_subsets"]} '
                  f'layers={selection["selected_layers"]} feasible={selection["feasible"]}',flush=True)
            save_json(selection,dataset_out/f'asls_{rule}.json')
            asls['selection_comparisons'][rule]={k:selection[k] for k in
                ['selected_layers','geometry_error','feasible','geometry_representation','discrete_selection']}
    print(f"[fit] ASLS done: selected_layers={asls['selected_layers']} geometry_error={asls['geometry_error']:.4f}", flush=True)
    save_json(asls, dataset_out / "asls.json")
    collected[mode]['asls_cache_key']=cache_key
    collected[mode]['asls_artifacts']={p.name:json.loads(p.read_text(encoding='utf-8'))
        for p in dataset_out.glob('asls*.json')}
    return patches, var_vectors, asls


def fit_method_specs(extractor, train_samples, cfg, dataset_out, collected=None, gram_cache=None, method_prefix=''):
    if collected is None:
        collected=collect_normal_inputs(extractor,train_samples,cfg)
    patches, variabilities, asls = fit_representation(extractor, train_samples, cfg, dataset_out,collected)
    input_mode,_=normalization_modes(cfg)
    selected = asls["selected_layers"]
    fixed = [3, 6, 9, 12]
    all_layers = list(range(1, extractor.num_layers + 1))
    last = [extractor.num_layers]
    seed = int(cfg["seed"])
    enabled = set(cfg["methods"])
    uv_cache={}
    gram_cache={} if gram_cache is None else gram_cache
    objective,_=objective_modes(cfg)

    def serializable(value):
        if isinstance(value,np.ndarray):
            return value.tolist()
        if isinstance(value,dict):
            return {k:serializable(v) for k,v in value.items()}
        if isinstance(value,list):
            return [serializable(v) for v in value]
        return value

    def save_uv(name,layers,result):
        audit=result.get('budget_geometry_audit')
        mass=audit['coordinate_reference_alignment'] if audit is not None else []
        by_layer={str(l):sum(mass[i*extractor.hidden_dim:(i+1)*extractor.hidden_dim]) for i,l in enumerate(layers)}
        relative=result.get('budgeted_weights',[]) if result.get('objective_definition') in {COSINE,PERFORMANCE,LOCAL} else []
        relative_by_layer={str(l):float(sum(relative[i*extractor.hidden_dim:(i+1)*extractor.hidden_dim])) for i,l in enumerate(layers)} if len(relative) else None
        save_json({'layers':layers,'layer_normalization':input_mode,
                   'normal_reference_alignment_by_layer':by_layer if mass else None,
                   'relative_weight_by_layer':relative_by_layer,
                   **serializable(result)},dataset_out/f'uvarfs_{name}.json')

    def fit_gram(name,layers):
        key=tuple(layers)
        if key not in gram_cache:
            x = torch.cat([patches[l] for l in layers], dim=-1)
            v = torch.cat([variabilities[l] for l in layers], dim=1)
            gram_cache[key]=fit_uvarfs(x,v,cfg['uvarfs'],label=f'uvarfs:{name}:gram-control')
            gram_cache[key]['objective_definition']=QUADRATIC
            gram_cache[key]['geometry_metric']='unrenormalized_weighted_gram_relative_frobenius'
        return gram_cache[key]

    def fit_uv(name, layers):
        key=tuple(layers)
        if key not in uv_cache:
            if objective==LOCAL:
                source=collected[input_mode]
                join=lambda values:torch.cat([values[l] for l in layers],dim=-1)
                uv_cache[key]=fit_local_uvarfs(join(patches),join(source['heldout']),source['groups'],source['heldout_groups'],
                    (join(source['paired_clean']),join(source['paired_perturbations'])),cfg['uvarfs'],label=f'uvarfs:{name}:local')
            elif objective in {COSINE,PERFORMANCE}:
                old=fit_gram(name,layers)
                x=torch.cat([patches[l] for l in layers],dim=-1)
                v=torch.cat([variabilities[l] for l in layers],dim=1)
                if objective==PERFORMANCE:
                    controls=gram_cache.setdefault('fixed_budget_controls',{})
                    if key not in controls:
                        controls[key]=fit_fixed_budget_reference(x,v,cfg['uvarfs'],old,label=f'uvarfs:{name}:v17-control')
                    uv_cache[key]=fit_performance_uvarfs(x,v,cfg['uvarfs'],old,
                        label=f'uvarfs:{name}:{objective}',reference=controls[key])
                else:
                    uv_cache[key]=fit_cosine_uvarfs(x,v,cfg['uvarfs'],old,label=f'uvarfs:{name}:{objective}')
            else:
                uv_cache[key]=fit_gram(name,layers)
        r=uv_cache[key]
        save_uv(name,layers,r)
        return r

    main_uv = fit_uv("main", selected)
    main_gram=fit_gram('main',selected)
    fixed_uv = fit_uv("fixed4", fixed) if 'fixed4_uvarfs' in enabled else None
    all_uv = fit_uv("all", all_layers) if 'all_uvarfs' in enabled else None
    xsel = torch.cat([patches[l] for l in selected], dim=-1)
    target = max(1, len(main_uv["active"]))
    pca = fit_pca(xsel, target, seed) if 'asls_pca' in enabled else None
    specs = {
        "main": {"layers": selected, "kind": "uvarfs", "obj": main_uv},
        "last_raw": {"layers": last, "kind": "raw"},
        "fixed4_raw": {"layers": fixed, "kind": "raw"},
        "all_raw": {"layers": all_layers, "kind": "raw"},
        "asls_raw": {"layers": selected, "kind": "raw"},
        "asls_pca": {"layers": selected, "kind": "pca", "obj": pca},
        "fixed4_uvarfs": {"layers": fixed, "kind": "uvarfs", "obj": fixed_uv},
        "all_uvarfs": {"layers": all_layers, "kind": "uvarfs", "obj": all_uv},
    }
    if objective==LOCAL:
        source=collected[input_mode]
        for name,beta,rank_weight in [('asls_no_variability_uvarfs',0.,float(cfg['uvarfs'].get('local_rank_weight',1.))),
                                      ('asls_no_rank_uvarfs',float(cfg['uvarfs']['beta']),0.)]:
            if name in enabled:
                join=lambda values:torch.cat([values[l] for l in selected],dim=-1)
                obj=fit_local_uvarfs(xsel,join(source['heldout']),source['groups'],source['heldout_groups'],
                    (join(source['paired_clean']),join(source['paired_perturbations'])),
                    {**cfg['uvarfs'],'beta':beta,'local_rank_weight':rank_weight},label='uvarfs:'+name)
                save_uv(name.removesuffix('_uvarfs'),selected,obj)
                specs[name]={'layers':selected,'kind':'uvarfs','obj':obj}
    if 'previous_main' in enabled:
        previous=json.loads((dataset_out/'asls_previous.json').read_text(encoding='utf-8'))
        pl=previous['selected_layers']; px=torch.cat([patches[l] for l in pl],1)
        pv=torch.cat([variabilities[l] for l in pl],1)
        old=fit_gram('previous_main',pl)
        obj=fit_performance_uvarfs(px,pv,cfg['uvarfs'],old,label='uvarfs:previous_main')
        save_uv('previous_main',pl,obj)
        specs['previous_main']={'layers':pl,'kind':'uvarfs','obj':obj,'asls_geometry':previous}
    if 'asls_fixed_budget_uvarfs' in enabled:
        cache=gram_cache.setdefault('fixed_budget_controls',{})
        key=tuple(selected)
        if key not in cache:
            v=torch.cat([variabilities[l] for l in selected],dim=1)
            cache[key]=fit_fixed_budget_reference(xsel,v,cfg['uvarfs'],main_gram,label='uvarfs:v17-control')
        save_uv('asls_fixed_budget',selected,cache[key])
        specs['asls_fixed_budget_uvarfs']={'layers':selected,'kind':'uvarfs','obj':cache[key]}
    if 'asls_compression_uvarfs' in enabled:
        cache=gram_cache.setdefault('compression_cosine_controls',{})
        key=tuple(selected)
        if key not in cache:
            v=torch.cat([variabilities[l] for l in selected],dim=1)
            cache[key]=fit_cosine_uvarfs(xsel,v,cfg['uvarfs'],main_gram,label='uvarfs:compression-control')
        save_uv('asls_compression',selected,cache[key])
        specs['asls_compression_uvarfs']={'layers':selected,'kind':'uvarfs','obj':cache[key]}
    if 'asls_selected_raw' in enabled:
        # Copy the fitted Main support, never rerank, refit or use test data.
        control={'layers':list(selected),'active':[int(j) for j in main_uv['active']],
                 'source_main':method_prefix+'main','source_objective':objective,
                 'layer_normalization':input_mode,'feature_dim':len(main_uv['active']),
                 'kind':'main_support_without_relative_weights',
                 'representation_weighting':'uniform_on_main_support',
                 'extra_fit':False,'selection_candidate':False}
        save_json(control,dataset_out/'selected_support_control.json')
        specs['asls_selected_raw']={'layers':list(selected),'kind':'selected_raw',
            'obj':control,'support_source_method':control['source_main'],
            'representation_weighting':control['representation_weighting']}
    for mode,selection in asls['geometry_comparisons'].items():
        layers=selection['selected_layers']
        if f'asls_{mode}_raw' in enabled:
            specs[f'asls_{mode}_raw']={'layers':layers,'kind':'raw'}
        if f'asls_{mode}_uvarfs' in enabled:
            obj=main_uv if layers==selected else fit_uv(f'asls_{mode}',layers)
            specs[f'asls_{mode}_uvarfs']={'layers':layers,'kind':'uvarfs','obj':obj}
        for name in [f'asls_{mode}_raw',f'asls_{mode}_uvarfs']:
            if name in specs:
                specs[name]['asls_geometry']=selection
    for rule,selection in asls['selection_comparisons'].items():
        layers=selection['selected_layers']
        for kind in ['raw','uvarfs']:
            name=f'asls_{rule}_{kind}'
            if name in enabled:
                spec={'layers':layers,'kind':kind,'asls_geometry':selection}
                if kind=='uvarfs':
                    spec['obj']=main_uv if layers==selected else fit_uv(f'asls_{rule}',layers)
                specs[name]=spec
    if 'asls_top_weights_uvarfs' in enabled:
        obj=main_gram.get('legacy_top_weights',main_gram)
        save_uv('asls_top_weights',selected,obj)
        specs['asls_top_weights_uvarfs']={'layers':selected,'kind':'uvarfs','obj':obj,
            'asls_geometry':asls['geometry_comparisons'][asls['geometry_representation']]}
    if 'asls_prune_refit_uvarfs' in enabled:
        obj=main_gram.get('prune_refit',main_gram)
        save_uv('asls_prune_refit',selected,obj)
        specs['asls_prune_refit_uvarfs']={'layers':selected,'kind':'uvarfs','obj':obj,
            'asls_geometry':asls['geometry_comparisons'][asls['geometry_representation']]}
    if 'asls_exchange_refit_uvarfs' in enabled:
        obj=main_gram.get('exchange_refit',main_gram)
        save_uv('asls_exchange_refit',selected,obj)
        specs['asls_exchange_refit_uvarfs']={'layers':selected,'kind':'uvarfs','obj':obj,
            'asls_geometry':asls['geometry_comparisons'][asls['geometry_representation']]}
    if 'legacy_main' in enabled:
        prefix=reselect_asls(json.loads((dataset_out/'asls_previous.json').read_text(encoding='utf-8')) if cfg['asls'].get('discrete_selection')=='normal_local' else asls,cfg['asls'],'gate_prefix')
        prefix_uv=fit_gram('asls_gate_prefix',prefix['selected_layers'])
        obj=prefix_uv.get('legacy_top_weights',prefix_uv)
        save_uv('legacy_main',prefix['selected_layers'],obj)
        specs['legacy_main']={'layers':prefix['selected_layers'],'kind':'uvarfs','obj':obj,'asls_geometry':prefix}
    seeds = cfg.get("random_baseline_seeds", [seed])
    if "asls_random" in cfg["methods"]:
        for rs in seeds:
            rr = np.random.default_rng(int(rs))
            ridx = np.sort(rr.choice(xsel.shape[1], size=min(target, xsel.shape[1]), replace=False))
            specs[f"asls_random_seed{rs}"] = {"layers": selected, "kind": "random", "obj": ridx}
    if "random4_uvarfs" in cfg["methods"]:
        for rs in seeds:
            rr = np.random.default_rng(int(rs))
            rl = sorted(rr.choice(all_layers, min(4, len(all_layers)), replace=False).tolist())
            specs[f"random4_uvarfs_seed{rs}"] = {"layers": rl, "kind": "uvarfs", "obj": fit_uv(f"random4_seed{rs}", rl)}
    # Match ASLS's chosen layer count, using every configured seed.
    if enabled & {'randomk_raw','randomk_uvarfs'}:
        for rs in seeds:
            rr=np.random.default_rng(int(rs))
            rl=sorted(rr.choice(all_layers,len(selected),replace=False).tolist())
            if 'randomk_raw' in enabled:
                specs[f'randomk_raw_seed{rs}']={'layers':rl,'kind':'raw'}
            if 'randomk_uvarfs' in enabled:
                specs[f'randomk_uvarfs_seed{rs}']={'layers':rl,'kind':'uvarfs',
                    'obj':fit_uv(f'randomk_seed{rs}',rl)}
    keep = {}
    for k, v in specs.items():
        base = "asls_random" if k.startswith("asls_random_seed") else "random4_uvarfs" if k.startswith("random4_uvarfs_seed") else k
        if k.startswith('randomk_raw_seed'):
            base='randomk_raw'
        elif k.startswith('randomk_uvarfs_seed'):
            base='randomk_uvarfs'
        if base in enabled:
            v['layer_normalization']=input_mode
            v['objective_branch']=objective
            if v['kind']=='uvarfs':
                v['uvarfs_objective']=v['obj'].get('objective_definition',QUADRATIC)
            if k in {'main','asls_raw','asls_pca','asls_selected_raw','asls_compression_uvarfs','asls_fixed_budget_uvarfs'} or base=='asls_random':
                v['asls_geometry']=asls['geometry_comparisons'][asls['geometry_representation']]
            keep[k] = v
    print('[fit] normal cosine geometry audit (diagnostic only)',flush=True)
    audit=audit_normal_geometry(patches,keep,input_mode)
    save_json(audit,dataset_out/'normal_geometry_audit.json')
    return keep


def fit_all_method_specs(extractor, train_samples, cfg, dataset_out):
    """Paired input ablation using the same images, patches and DINO forwards."""
    primary,modes=normalization_modes(cfg)
    collected=collect_normal_inputs(extractor,train_samples,cfg)
    specs={}; caches={mode:{} for mode in modes}
    for branch in experiment_branches(cfg):
        mode=branch['layer_normalization']; prefix=branch['prefix']
        settings={**cfg,'representation':{'layer_normalization':mode,'ablation_layer_normalization':None},
                  'methods':methods_for_objective(cfg,branch['objective']),
                  'uvarfs':{**cfg['uvarfs'],'objective':branch['objective'],'ablation_objective':None}}
        output=dataset_out/branch['folder']
        fitted=fit_method_specs(extractor,train_samples,settings,output,collected,caches[mode],method_prefix=prefix)
        specs.update({prefix+name:spec for name,spec in fitted.items()})
    save_json({'primary_layer_normalization':primary,
               'ablation_layer_normalization':modes[1] if len(modes)>1 else None,
               'shared_fit_images_and_patches':True,'shared_backbone_forwards':True,
               'shared_asls_by_input':True,
               'objective_branches':experiment_branches(cfg),
               'objective_branch_by_method':{name:spec['objective_branch'] for name,spec in specs.items()},
               'uvarfs_objective_by_method':{name:spec.get('uvarfs_objective') for name,spec in specs.items()},
               'normalization_by_method':{name:spec['layer_normalization'] for name,spec in specs.items()}},
              dataset_out/'representation_manifest.json')
    return specs
