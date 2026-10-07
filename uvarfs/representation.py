"""Shared feature input conventions for fit, variability, memory and queries."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def normalization_modes(cfg):
    settings=cfg.get('representation',{})
    primary=settings.get('layer_normalization','none')
    ablation=settings.get('ablation_layer_normalization')
    if primary not in {'none','l2'} or ablation not in {None,'none','l2'}:
        raise ValueError('layer normalization must be none or l2')
    if ablation==primary:
        raise ValueError('normalization ablation must differ from the primary input')
    return primary,[primary]+([ablation] if ablation is not None else [])


def normalization_prefix(mode):
    if mode not in {'none','l2'}:
        raise ValueError('layer normalization must be none or l2')
    return 'raw_input_' if mode=='none' else 'layer_l2_'


def normalize_layer(features: torch.Tensor, mode: str):
    if mode=='none':
        return features
    if mode=='l2':
        # Protocol/result auditing needs only the mode names, not PyTorch.
        import torch.nn.functional as F
        return F.normalize(features.float(),dim=-1)
    raise ValueError('layer normalization must be none or l2')


def normalize_layers(features, mode):
    return {layer:normalize_layer(value,mode) for layer,value in features.items()}
