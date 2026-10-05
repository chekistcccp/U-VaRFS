#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, shutil, subprocess, sys
from pathlib import Path
from urllib.parse import urlencode


def _download(kind: str, repo_id: str, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    try:
        from modelscope import snapshot_download, dataset_snapshot_download
        if kind == "model": snapshot_download(repo_id, local_dir=str(out))
        else: dataset_snapshot_download(repo_id, local_dir=str(out))
        return
    except Exception as sdk_error:
        exe=shutil.which("modelscope") or shutil.which("ms")
        if exe:
            cmd=[exe,"download",f"--{kind}",repo_id,"--local_dir",str(out)]
            p=subprocess.run(cmd)
            if p.returncode == 0: return
        raise RuntimeError(f"ModelScope download failed: {repo_id}; SDK error={sdk_error}")


def _unpack_archives(root: Path):
    import tarfile, zipfile
    for p in list(root.rglob("*")):
        if not p.is_file(): continue
        low=p.name.lower()
        try:
            if low.endswith(".zip"):
                out=p.with_suffix(""); out.mkdir(exist_ok=True); zipfile.ZipFile(p).extractall(out)
            elif low.endswith((".tar.gz",".tgz",".tar")):
                out=p.parent/(p.name.split(".tar")[0]); out.mkdir(exist_ok=True); tarfile.open(p).extractall(out)
            elif low.endswith(".7z"):
                import py7zr
                out=p.with_suffix(""); out.mkdir(exist_ok=True)
                with py7zr.SevenZipFile(p,"r") as z: z.extractall(out)
        except Exception as e:
            print(f"[warn] cannot unpack {p}: {e}")

def _valid_bmad(root: Path):
    return len([p for p in root.rglob("train") if p.is_dir() and (p.parent/"test").is_dir()]) >= 6


def _search_bmad_ids():
    # Public OpenAPI; if authentication is required, MODELSCOPE_API_TOKEN is used.
    import requests
    url="https://modelscope.cn/openapi/v1/datasets?"+urlencode({"search":"BMAD","sort":"downloads","page_size":30})
    headers={}
    tok=os.environ.get("MODELSCOPE_API_TOKEN")
    if tok: headers["Authorization"]="Bearer "+tok
    try:
        r=requests.get(url,headers=headers,timeout=30); r.raise_for_status(); obj=r.json()
        data=obj.get("data",obj); rows=data.get("datasets",[]) if isinstance(data,dict) else []
        ids=[]
        for x in rows:
            rid=x.get("id") or x.get("name") or x.get("path")
            if rid and "/" in rid: ids.append(rid)
        return ids
    except Exception:
        return []


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--root",default="experiment"); ap.add_argument("--model-id",default=os.getenv("DINOV3_MODELSCOPE_ID","keras/dinov3_vit_small_plus_lvd1689m")); ap.add_argument("--dataset-id",default=os.getenv("BMAD_MODELSCOPE_ID","auto")); args=ap.parse_args()
    root=Path(args.root); model_dir=root/"models"/"dinov3_vitsplus"; data_dir=root/"data"/"BMAD"
    if not (model_dir/"config.json").exists() and not any(model_dir.glob("*.json")):
        print(f"[download] DINOv3 from ModelScope: {args.model_id}")
        _download("model",args.model_id,model_dir)
    else: print("[download] model already present")
    if _valid_bmad(data_dir):
        print("[download] BMAD already present"); return
    candidates=[]
    if args.dataset_id != "auto": candidates.append(args.dataset_id)
    candidates += _search_bmad_ids()
    candidates += ["OpenDataLab/BMAD","AI-ModelScope/BMAD","DorisBao/BMAD","modelscope/BMAD"]
    seen=set(); candidates=[x for x in candidates if not (x in seen or seen.add(x))]
    errors=[]
    for rid in candidates:
        print(f"[download] trying BMAD ModelScope repo: {rid}")
        tmp=root/"data"/("_bmad_"+rid.replace("/","__"))
        try:
            if tmp.exists(): shutil.rmtree(tmp)
            _download("dataset",rid,tmp)
            _unpack_archives(tmp)
            if _valid_bmad(tmp):
                if data_dir.exists(): shutil.rmtree(data_dir)
                tmp.rename(data_dir); print(f"[download] BMAD ready from {rid}"); return
            errors.append(f"{rid}: downloaded but not recognized as all-6 BMAD layout")
        except Exception as e: errors.append(f"{rid}: {e}")
        finally:
            if tmp.exists(): shutil.rmtree(tmp,ignore_errors=True)
    raise SystemExit("\nNo verified all-6 BMAD mirror was found on ModelScope.\nSet BMAD_MODELSCOPE_ID=<namespace/repo> to a ModelScope mirror containing the reorganized BMAD data.\nTried:\n  - "+"\n  - ".join(errors))

if __name__=="__main__": main()
