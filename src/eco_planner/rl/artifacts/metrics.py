"""Transition-weighted rollout aggregations, independent of PPO mathematics."""

from tensordict import TensorDictBase
from torchmetrics import MaxMetric, MeanMetric, SumMetric


class RolloutMetrics:
    def __init__(self, trajectory: TensorDictBase) -> None:
        self.trajectory = trajectory

    def _compute(self, name: str, metric: MeanMetric | SumMetric | MaxMetric) -> float:
        value = self.trajectory[name].detach()
        metric.set_dtype(value.dtype)
        metric.to(value.device)
        metric.update(value)
        return float(metric.compute())

    def mean(self, name: str) -> float:
        return self._compute(name, MeanMetric(nan_strategy="error"))

    def sum(self, name: str) -> float:
        return self._compute(name, SumMetric(nan_strategy="error"))

    def maximum(self, name: str) -> float:
        return self._compute(name, MaxMetric(nan_strategy="error"))

    def stopped_fraction(self) -> float:
        metric = MeanMetric(nan_strategy="error").to(self.trajectory.device)
        metric.update(self.trajectory["stopped"].detach().float())
        return float(metric.compute())
