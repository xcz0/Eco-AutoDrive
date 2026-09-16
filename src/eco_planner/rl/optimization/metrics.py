"""Detached, update-scoped aggregation of PPO diagnostics."""

from collections.abc import Sequence

import torch
from torchmetrics import MaxMetric, MeanMetric


class PPOMetrics:
    def __init__(self, count: int, device: torch.device) -> None:
        self.means = [
            MeanMetric(nan_strategy="error").set_dtype(torch.float64).to(device)
            for _ in range(count)
        ]
        self.maximum_gradient = MaxMetric(nan_strategy="error").set_dtype(torch.float64).to(device)
        # Preserve the defined zero result when KL stops before any optimizer step.
        self.maximum_gradient.update(torch.zeros((), dtype=torch.float64, device=device))

    def update(self, values: Sequence[torch.Tensor]) -> None:
        for metric, value in zip(self.means, values, strict=True):
            metric.update(value.detach().mean().to(dtype=torch.float64))

    def gradient(self, value: torch.Tensor) -> None:
        self.maximum_gradient.update(value.detach().to(dtype=torch.float64))

    def compute(self) -> torch.Tensor:
        return torch.stack(
            [metric.compute() for metric in self.means] + [self.maximum_gradient.compute()]
        )
