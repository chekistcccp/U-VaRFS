#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from uvarfs.utils import load_config, set_seed, ensure_dir
from uvarfs.data import discover_bmad_roots, scan_split
from uvarfs.dinov3 import DINOv3Extractor
from uvarfs.pipeline_fit import fit_method_specs
from uvarfs.pipeline_eval import build_memories, evaluate, write_summaries


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="configs/default.yaml"); ap.add_argument("--dataset", default="all")
    args = ap.parse_args(); cfg = load_config(args.config); set_seed(int(cfg["seed"]))
    exp = Path(cfg["experiment_dir"]); res = ensure_dir(cfg["results_dir"])
    roots = discover_bmad_roots(exp / "data" / "BMAD")
    if not roots: raise SystemExit("No BMAD datasets found. Run scripts/download_assets.py first.")
    extractor = DINOv3Extractor(exp / "models" / "dinov3_vitsplus", int(cfg["model"]["input_size"]), cfg["model"]["amp"])
    all_rows, layer_rows = [], []
    candidate_dim = extractor.num_layers * extractor.hidden_dim
    for dname, droot in sorted(roots.items()):
        if args.dataset != "all" and args.dataset.lower() != dname.lower(): continue
        out = ensure_dir(res / dname); train = scan_split(droot, "train", dname); test = scan_split(droot, "test", dname)
        if not train or not test:
            print(f"[skip] {dname}: incomplete split"); continue
        print(f"\n=== {dname}: train={len(train)} test={len(test)} ===")
        if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
        t0 = time.time(); t = time.time(); specs = fit_method_specs(extractor, train, cfg, out); fit_s = time.time() - t
        t = time.time(); memories = build_memories(extractor, train, specs, cfg); mem_s = time.time() - t
        t = time.time(); rows = evaluate(extractor, test, specs, memories, cfg); eval_s = time.time() - t
        peak = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
        for r in rows:
            spec = specs[r["method"]]; dim = int(memories[r["method"]].shape[1])
            r.update(dataset=dname, seconds=time.time()-t0, fit_seconds=fit_s, memory_seconds=mem_s, eval_seconds=eval_s,
                     peak_gpu_gb=peak, selected_layers=";".join(map(str, spec["layers"])), selected_layer_count=len(spec["layers"]),
                     feature_dim=dim, candidate_dim=candidate_dim, compression_ratio=1.0-dim/max(candidate_dim,1), memory_size=len(memories[r["method"]]))
        pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False); all_rows += rows
        apath = out / "asls.json"
        if apath.exists():
            obj = json.loads(apath.read_text(encoding="utf-8"))
            for l in range(1, extractor.num_layers+1):
                layer_rows.append({"dataset": dname, "layer": l, "selected": int(l in obj.get("selected_layers", [])),
                                   "probability": obj.get("probabilities", {}).get(str(l), np.nan),
                                   "variability": obj.get("variability", {}).get(str(l), np.nan),
                                   "geometry_error": obj.get("geometry_error", np.nan)})
    df = pd.DataFrame(all_rows); df.to_csv(res / "all_metrics.csv", index=False); write_summaries(df, res)
    if layer_rows: pd.DataFrame(layer_rows).to_csv(res / "layer_selection.csv", index=False)
    if not df.empty:
        cols = [c for c in ["dataset","method","image_auroc","image_auprc","pixel_auroc","aupro","feature_dim","compression_ratio"] if c in df.columns]
        print(df[cols].to_string(index=False)); print(f"\nSaved: {res/'all_metrics.csv'}"); print(f"Saved: {res/'summary_metrics.csv'}")

if __name__ == "__main__": main()
