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


def _sample_indices(n, k, seed):
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=min(n, k), replace=False))


def _patch_ids(p, k, seed):
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(p, size=min(p, k), replace=False))


def fit_representation(extractor, train_samples, cfg, dataset_out):
    seed = int(cfg["seed"])
    fit_n = int(cfg["data"]["fit_images"])
    var_n = int(cfg["data"]["variability_images"])
    ppi = int(cfg["data"]["patches_per_image"])
    ids = _sample_indices(len(train_samples), fit_n, seed)
    ds = BMADDataset(train_samples, int(cfg["model"]["input_size"]), False)
    dl = DataLoader(
        Subset(ds, ids.tolist()),
        batch_size=int(cfg["model"]["batch_size"]), shuffle=False,
        num_workers=int(cfg["model"]["num_workers"]), pin_memory=True,
    )
    layers = range(1, extractor.num_layers + 1)
    pooled = {i: [] for i in layers}
    patchbuf = {i: [] for i in layers}
    var_sum = {k: {i: torch.zeros(extractor.hidden_dim) for i in layers} for k in PERTURBATIONS}
    var_count = {k: 0 for k in PERTURBATIONS}
    global_pos = 0
    for batch in tqdm(dl, desc="fit-features", leave=False):
        x = batch["image"]
        feats = extractor(x)
        pf = extractor.pooled(feats)
        b = x.shape[0]
        p = next(iter(feats.values())).shape[1]
        patch_ids = _patch_ids(p, ppi, seed + global_pos)
        for l in layers:
            pooled[l].append(pf[l].cpu())
            patchbuf[l].append(feats[l][:, patch_ids, :].reshape(-1, extractor.hidden_dim).cpu())
        if global_pos < var_n:
            take = min(b, var_n - global_pos)
            xvar = x[:take].cuda(non_blocking=True) if torch.cuda.is_available() else x[:take]
            base = {l: feats[l][:take] for l in layers}
            for kind in PERTURBATIONS:
                pert = extractor(perturb_batch(xvar, kind))
                for l in layers:
                    aa, bb = base[l].float(), pert[l].float()
                    delta = (aa - bb).pow(2).mean(dim=(0, 1)).cpu()
                    denom = aa.reshape(-1, aa.shape[-1]).var(dim=0, unbiased=False).clamp_min(1e-6).cpu()
                    var_sum[kind][l] += (delta / denom) * take
                var_count[kind] += take
        global_pos += b
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pooled = {l: torch.cat(v).to(device) for l, v in pooled.items()}
    patches = {l: torch.cat(v).to(device) for l, v in patchbuf.items()}
    var_vectors = {}
    for l in pooled:
        rows = [var_sum[k][l] / max(var_count[k], 1) for k in PERTURBATIONS]
        var_vectors[l] = torch.stack(rows, dim=0).to(device)
    layer_var = {l: float(var_vectors[l].mean().cpu()) for l in pooled}
    asls = fit_asls(pooled, layer_var, cfg["asls"])
    save_json(asls, dataset_out / "asls.json")
    return patches, var_vectors, asls


def fit_method_specs(extractor, train_samples, cfg, dataset_out):
    patches, variabilities, asls = fit_representation(extractor, train_samples, cfg, dataset_out)
    selected = asls["selected_layers"]
    fixed = [3, 6, 9, 12]
    all_layers = list(range(1, extractor.num_layers + 1))
    last = [extractor.num_layers]
    seed = int(cfg["seed"])

    def fit_uv(name, layers):
        x = torch.cat([patches[l] for l in layers], dim=-1)
        v = torch.cat([variabilities[l] for l in layers], dim=1)
        r = fit_uvarfs(x, v, cfg["uvarfs"])
        save_json({"layers": layers, "lambda": r["lambda"], "active": r["active"].tolist(),
                   "scales": r["scales"].tolist(), "geometry_error": r["geometry_error"]},
                  dataset_out / f"uvarfs_{name}.json")
        return r

    main_uv = fit_uv("main", selected)
    fixed_uv = fit_uv("fixed4", fixed)
    all_uv = fit_uv("all", all_layers)
    xsel = torch.cat([patches[l] for l in selected], dim=-1)
    target = max(1, len(main_uv["active"]))
    pca = fit_pca(xsel, target, seed)
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
    enabled = set(cfg["methods"])
    keep = {}
    for k, v in specs.items():
        base = "asls_random" if k.startswith("asls_random_seed") else "random4_uvarfs" if k.startswith("random4_uvarfs_seed") else k
        if base in enabled:
            keep[k] = v
    return keep
