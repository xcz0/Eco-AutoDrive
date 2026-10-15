"""Detached, update-scoped aggregation of PPO diagnostics."""

from collections.abc import Mapping, Sequence

import torch
from torchmetrics import MaxMetric, MeanMetric


class PPOMetrics:
    def __init__(self, names: Sequence[str], device: torch.device) -> None:
        self.means = {
            name: MeanMetric(nan_strategy="error").set_dtype(torch.float64).to(device)
            for name in names
        }
        self.maximum_gradient = MaxMetric(nan_strategy="error").set_dtype(torch.float64).to(device)
        # Preserve the defined zero result when KL stops before any optimizer step.
        self.maximum_gradient.update(torch.zeros((), dtype=torch.float64, device=device))

    def update(self, values: Mapping[str, torch.Tensor]) -> None:
        if values.keys() != self.means.keys():
            raise ValueError("PPO metric values must match the configured names")
        for name, metric in self.means.items():
            metric.update(values[name].detach().mean().to(dtype=torch.float64))

    def gradient(self, value: torch.Tensor) -> None:
        self.maximum_gradient.update(value.detach().to(dtype=torch.float64))

    def compute(self) -> dict[str, float]:
        values = {name: metric.compute() for name, metric in self.means.items()}
        values["maximum_pre_clip_gradient_norm"] = self.maximum_gradient.compute()
        host = torch.stack(tuple(values.values())).cpu()
        if not torch.isfinite(host).all():
            raise FloatingPointError("PPO update diagnostics must be finite")
        return {name: float(value) for name, value in zip(values, host, strict=True)}
