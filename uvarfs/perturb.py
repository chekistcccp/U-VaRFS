from __future__ import annotations
import torch
import torch.nn.functional as F

_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)

def _denorm(x):
    mean=_MEAN.to(x.device,x.dtype).view(1,1,1,3); std=_STD.to(x.device,x.dtype).view(1,1,1,3)
    return (x*std+mean).clamp(0,1)
def _norm(x):
    mean=_MEAN.to(x.device,x.dtype).view(1,1,1,3); std=_STD.to(x.device,x.dtype).view(1,1,1,3)
    return (x-mean)/std

def perturb_batch(x: torch.Tensor, kind: str, generator: torch.Generator | None = None) -> torch.Tensor:
    y=_denorm(x)
    if kind == "noise":
        noise=torch.randn(y.shape,device=y.device,dtype=y.dtype,generator=generator)
        y=(y + 0.015 * noise).clamp(0,1)
    elif kind == "blur":
        z=y.permute(0,3,1,2)
        z=F.avg_pool2d(z,3,stride=1,padding=1)
        y=z.permute(0,2,3,1)
    elif kind == "gamma":
        y=y.clamp_min(1e-6).pow(1.08)
    elif kind == "resolution":
        z=y.permute(0,3,1,2); h,w=z.shape[-2:]
        z=F.interpolate(z,scale_factor=0.75,mode="bilinear",align_corners=False,antialias=True)
        z=F.interpolate(z,size=(h,w),mode="bilinear",align_corners=False,antialias=True)
        y=z.permute(0,2,3,1)
    else:
        raise ValueError(kind)
    return _norm(y)

PERTURBATIONS=("noise","blur","gamma","resolution")
