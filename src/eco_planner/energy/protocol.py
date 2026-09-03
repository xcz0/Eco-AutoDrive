"""Shared interface for trace-level energy measurement."""

from __future__ import annotations

from typing import Protocol

from eco_planner.energy.metrics import EnergyMetrics, EnergyTrace


class EnergyMetricProvider(Protocol):
    """Measure energy over an executed time/speed trace."""

    def measure(self, trace: EnergyTrace) -> EnergyMetrics: ...


__all__ = ["EnergyMetricProvider"]
