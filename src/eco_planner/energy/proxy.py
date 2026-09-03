"""Lightweight MetaDrive-compatible fuel proxy provider."""

from __future__ import annotations

import numpy as np

from eco_planner.energy.metrics import EnergyMetrics, EnergyTrace


class MetaDriveFuelProxyProvider:
    """Recompute MetaDrive's fuel proxy over actual executed distances."""

    def measure(self, trace: EnergyTrace) -> EnergyMetrics:
        speed_kmh = trace.speed_mps[1:] * 3.6
        step_fuel_ml = (
            3.25 * np.exp(0.01 * speed_kmh) * (trace.step_distance_m / 1_000.0) / 100.0 * 1_000.0
        )
        return EnergyMetrics(
            metric="metadrive_fuel_proxy",
            distance_m=trace.distance_m,
            energy_j=None,
            fuel_ml=float(step_fuel_ml.sum(dtype=np.float64)),
        )


__all__ = ["MetaDriveFuelProxyProvider"]
