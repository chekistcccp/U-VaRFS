from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

ALIASES = {
    "brain": ["brain", "brats"],
    "liver": ["liver", "lits", "btcv"],
    "resc": ["resc"],
    "oct2017": ["oct2017", "oct_2017", "kermany"],
    "xray": ["xray", "x-ray", "rsna", "chest"],
    "camelyon16": ["camelyon16", "camelyon"],
}

@dataclass
class Sample:
    image: Path
    label: int
    mask: Optional[Path]
    split: str
    dataset: str


def _images_under(path: Path):
    if not path.exists():
        return []
    return sorted([p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS])


def discover_bmad_roots(root: str | Path) -> dict[str, Path]:
    root = Path(root)
    candidates = [p for p in root.rglob("*") if p.is_dir() and (p / "train").is_dir() and (p / "test").is_dir()]
    found = {}
    for p in candidates:
        name = p.name.lower()
        key = None
        for canonical, aliases in ALIASES.items():
            if any(a in name for a in aliases):
                key = canonical
                break
        if key is None:
            key = p.name
        found.setdefault(key, p)
    return found


def scan_split(dataset_root: Path, split: str, dataset_name: str) -> list[Sample]:
    split_root = dataset_root / split
    out: list[Sample] = []
    for cls_name, label in [("good", 0), ("Ungood", 1), ("ungood", 1), ("bad", 1)]:
        croot = split_root / cls_name
        if not croot.exists():
            continue
        img_root = croot / "img" if (croot / "img").is_dir() else croot
        mask_root = croot / "anomaly_mask"
        for img in _images_under(img_root):
            if "anomaly_mask" in img.parts:
                continue
            mask = None
            if label == 1 and mask_root.is_dir():
                rel = img.relative_to(img_root)
                m = mask_root / rel
                if m.exists():
                    mask = m
                else:
                    same = list(mask_root.rglob(img.name))
                    mask = same[0] if same else None
            out.append(Sample(img, label, mask, split, dataset_name))
    return out


def letterbox_rgb(path: Path, size: int) -> tuple[np.ndarray, tuple[int,int,int,int]]:
    im = Image.open(path).convert("RGB")
    w, h = im.size
    scale = min(size / max(w, 1), size / max(h, 1))
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    im = im.resize((nw, nh), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (size, size), color=(0, 0, 0))
    left, top = (size - nw) // 2, (size - nh) // 2
    canvas.paste(im, (left, top))
    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    return arr, (left, top, nw, nh)


def load_mask(path: Optional[Path], size: int, box: tuple[int,int,int,int]) -> np.ndarray:
    if path is None:
        return np.zeros((size, size), dtype=np.uint8)
    m = Image.open(path).convert("L")
    left, top, nw, nh = box
    m = m.resize((nw, nh), Image.Resampling.NEAREST)
    canvas = Image.new("L", (size, size), 0)
    canvas.paste(m, (left, top))
    return (np.asarray(canvas) > 0).astype(np.uint8)


class BMADDataset(Dataset):
    def __init__(self, samples: list[Sample], image_size: int, with_mask: bool = False):
        self.samples = samples
        self.image_size = image_size
        self.with_mask = with_mask
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        s = self.samples[idx]
        arr, box = letterbox_rgb(s.image, self.image_size)
        item = {
            "image": torch.from_numpy(arr),  # HWC for Keras torch backend
            "label": torch.tensor(s.label, dtype=torch.long),
            "path": str(s.image),
        }
        if self.with_mask:
            item["mask"] = torch.from_numpy(load_mask(s.mask, self.image_size, box))
            item["has_mask"] = torch.tensor(s.mask is not None, dtype=torch.bool)
        return item
