from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from .data import BMADDataset
from .perturb import perturb_batch, PERTURBATIONS
from .asls import fit_asls
from .u_varfs import fit_uvarfs
from .transforms import fit_pca
from .utils import save_json
from .variability import NormalVariability


def _sample_indices(n, k, seed):
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=min(n, k), replace=False))


def _patch_ids(p, k, seed):
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(p, size=min(p, k), replace=False))


def fit_representation(extractor, train_samples, cfg, dataset_out):
    if not train_samples or any(s.label != 0 or s.split != 'train' for s in train_samples):
        raise ValueError('ASLS/U-VaRFS fit requires the official normal-reference training split')
    seed = int(cfg["seed"])
    fit_n = int(cfg["data"]["fit_images"])
    var_n = int(cfg["data"]["variability_images"])
    ppi = int(cfg["data"]["patches_per_image"])
    ids = _sample_indices(len(train_samples), fit_n, seed)
    ds = BMADDataset(train_samples, int(cfg["model"]["input_size"]), False)
    dl = DataLoader(
        Subset(ds, ids.tolist()),
        batch_size=int(cfg["model"]["batch_size"]), shuffle=False,
        num_workers=int(cfg["model"]["num_workers"]), pin_memory=torch.cuda.is_available(),
    )
    layers = range(1, extractor.num_layers + 1)
    pooled = {i: [] for i in layers}
    patchbuf = {i: [] for i in layers}
    statistics = {i: NormalVariability(extractor.hidden_dim, PERTURBATIONS,
                    float(cfg['data'].get('variability_epsilon',1e-6))) for i in layers}
    patch_manifest = []
    global_pos = 0
    for batch in tqdm(dl, desc="fit-features", leave=False):
        x = batch["image"]
        feats = extractor(x)
        pf = extractor.pooled(feats)
        b = x.shape[0]
        p = next(iter(feats.values())).shape[1]
        patch_ids = _patch_ids(p, ppi, seed + global_pos)
        patch_manifest.append({'image_offset':global_pos,'batch_images':b,'patch_indices':patch_ids.tolist()})
        for l in layers:
            pooled[l].append(pf[l].cpu())
            patchbuf[l].append(feats[l][:, patch_ids, :].reshape(-1, extractor.hidden_dim).cpu())
        if global_pos < var_n:
            take = min(b, var_n - global_pos)
            xvar = x[:take].cuda(non_blocking=True) if torch.cuda.is_available() else x[:take]
            base = {l: feats[l][:take] for l in layers}
            for l in layers:
                statistics[l].add_normal(base[l])
            for kind in PERTURBATIONS:
                pert = extractor(perturb_batch(xvar, kind))
                for l in layers:
                    statistics[l].add_perturbation(kind,base[l],pert[l])
        global_pos += b
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pooled = {l: torch.cat(v).to(device) for l, v in pooled.items()}
    patches = {l: torch.cat(v).to(device) for l, v in patchbuf.items()}
    var_vectors = {}
    for l in pooled:
        var_vectors[l] = statistics[l].vectors().to(device)
    save_json({'normal_train_indices':ids.tolist(),
               'normal_train_paths':[str(train_samples[int(i)].image) for i in ids],
               'fit_images':len(ids),'variability_images':min(var_n,len(ids)),
               'patches_per_image':ppi,'sampled_patches':patch_manifest,
               'variability_definition':'global_mean_squared_delta/(global_population_variance+epsilon)',
               'variability_epsilon':float(cfg['data'].get('variability_epsilon',1e-6)),
               'asls_geometry_representation':cfg['asls'].get('geometry_representation','patch'),
               'perturbations':list(PERTURBATIONS),
               'variability_vectors':{str(l):v.cpu().tolist() for l,v in var_vectors.items()}},
              dataset_out/'fit_manifest.json')
    layer_var = {l: float(var_vectors[l].mean().cpu()) for l in pooled}
    representation=cfg['asls'].get('geometry_representation','patch')
    inputs={'pooled':pooled,'patch':patches}
    print(f"[fit] ASLS start: geometry={representation} layers={len(pooled)} samples={next(iter(inputs[representation].values())).shape[0]}", flush=True)
    asls = fit_asls(inputs[representation], layer_var, cfg["asls"])
    comparisons={representation:asls}
    for other in ['pooled','patch']:
        if other!=representation and any(method.startswith(f'asls_{other}_') for method in cfg['methods']):
            print(f'[fit] ASLS {other} geometry ablation',flush=True)
            comparisons[other]=fit_asls(inputs[other],layer_var,{**cfg['asls'],'geometry_representation':other})
    for mode,result in comparisons.items():
        save_json(result,dataset_out/f'asls_{mode}.json')
    # Keep only selection metadata here; optimizer traces live in separate files.
    asls['geometry_comparisons']={mode:{k:result[k] for k in ['selected_layers','geometry_error','feasible','geometry_representation']}
                                  for mode,result in comparisons.items()}
    print(f"[fit] ASLS done: selected_layers={asls['selected_layers']} geometry_error={asls['geometry_error']:.4f}", flush=True)
    save_json(asls, dataset_out / "asls.json")
    return patches, var_vectors, asls


def fit_method_specs(extractor, train_samples, cfg, dataset_out):
    patches, variabilities, asls = fit_representation(extractor, train_samples, cfg, dataset_out)
    selected = asls["selected_layers"]
    fixed = [3, 6, 9, 12]
    all_layers = list(range(1, extractor.num_layers + 1))
    last = [extractor.num_layers]
    seed = int(cfg["seed"])
    enabled = set(cfg["methods"])

    def fit_uv(name, layers):
        x = torch.cat([patches[l] for l in layers], dim=-1)
        v = torch.cat([variabilities[l] for l in layers], dim=1)
        print(f"[fit] U-VaRFS {name}: layers={layers} shape={tuple(x.shape)}", flush=True)
        r = fit_uvarfs(x, v, cfg["uvarfs"], label=f"uvarfs:{name}")
        serializable={k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in r.items()}
        save_json({"layers":layers,**serializable},
                  dataset_out / f"uvarfs_{name}.json")
        return r

    main_uv = fit_uv("main", selected)
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
            if k in {'main','asls_raw','asls_pca'} or base=='asls_random':
                v['asls_geometry']=asls['geometry_comparisons'][asls['geometry_representation']]
            keep[k] = v
    return keep
