from __future__ import annotations

import torch


class NormalVariability:
    """Streaming E[(f-f_perturbed)^2] / (Var[f] + eps), normal data only.

    Welford merging includes variance between batches. Averaging batch-wise
    ratios loses that variance and makes the estimate depend on batch size.
    """

    def __init__(self, dimensions: int, perturbations, epsilon: float = 1e-6):
        if epsilon <= 0:
            raise ValueError("variability epsilon must be positive")
        self.epsilon = float(epsilon)
        self.count = 0
        self.mean = torch.zeros(dimensions, dtype=torch.float64)
        self.m2 = torch.zeros_like(self.mean)
        self.delta_sum = {kind: torch.zeros_like(self.mean) for kind in perturbations}
        self.delta_count = {kind: 0 for kind in perturbations}

    def add_normal(self, features: torch.Tensor):
        x = features.detach().reshape(-1, features.shape[-1]).double()
        n = len(x)
        if not n:
            return
        batch_mean = x.mean(0)
        mean = batch_mean.cpu()
        m2 = ((x - batch_mean).square().sum(0)).cpu()
        difference = mean - self.mean
        total = self.count + n
        self.m2 += m2 + difference.square() * (self.count * n / total)
        self.mean += difference * (n / total)
        self.count = total

    def add_perturbation(self, kind: str, normal: torch.Tensor, perturbed: torch.Tensor):
        if normal.shape != perturbed.shape:
            raise ValueError("perturbations must preserve patch correspondence")
        delta = (normal.detach().double() - perturbed.detach().double()).reshape(-1, normal.shape[-1])
        self.delta_sum[kind] += delta.square().sum(0).cpu()
        self.delta_count[kind] += len(delta)

    def vectors(self) -> torch.Tensor:
        if not self.count or any(n != self.count for n in self.delta_count.values()):
            raise ValueError("each perturbation must cover the same normal-reference patches")
        variance = self.m2 / self.count
        return torch.stack([
            (self.delta_sum[k] / self.count) / (variance + self.epsilon)
            for k in self.delta_sum
        ]).float()
