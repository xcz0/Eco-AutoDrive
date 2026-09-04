"""Framework-independent energy facts and providers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Protocol, TypeAlias

import numpy as np

EnergyMetricName: TypeAlias = Literal["metadrive_fuel_proxy", "fastsim_fuel_energy"]


@dataclass(frozen=True, slots=True)
class EnergyTrace:
    """One executed speed trace with one distance value per transition."""

    time_s: np.ndarray
    speed_mps: np.ndarray
    step_distance_m: np.ndarray

    def __post_init__(self) -> None:
        time_s = _readonly_vector(self.time_s, "energy trace time")
        speed_mps = _readonly_vector(self.speed_mps, "energy trace speed")
        step_distance_m = _readonly_vector(self.step_distance_m, "energy trace step distance")
        if time_s.size < 2:
            raise ValueError("energy trace requires at least two time/speed points")
        if speed_mps.shape != time_s.shape:
            raise ValueError("energy trace time and speed must have identical shape")
        if step_distance_m.shape != (time_s.size - 1,):
            raise ValueError("energy trace step distance must align with transitions")
        if time_s[0] != 0.0 or np.any(np.diff(time_s) <= 0.0):
            raise ValueError("energy trace time must start at zero and strictly increase")
        if np.any(speed_mps < 0.0) or np.any(step_distance_m < 0.0):
            raise ValueError("energy trace speed and distance must be non-negative")
        object.__setattr__(self, "time_s", time_s)
        object.__setattr__(self, "speed_mps", speed_mps)
        object.__setattr__(self, "step_distance_m", step_distance_m)

    @property
    def distance_m(self) -> float:
        return float(self.step_distance_m.sum(dtype=np.float64))


@dataclass(frozen=True, slots=True)
class EnergyMetrics:
    """Energy consumed over one trace window in provider-native physical units."""

    metric: EnergyMetricName
    distance_m: float
    energy_j: float | None
    fuel_ml: float | None

    def __post_init__(self) -> None:
        if self.metric not in {"metadrive_fuel_proxy", "fastsim_fuel_energy"}:
            raise ValueError(f"unsupported energy metric: {self.metric!r}")
        if not math.isfinite(self.distance_m) or self.distance_m < 0.0:
            raise ValueError("energy metric distance must be finite and non-negative")
        if self.energy_j is None and self.fuel_ml is None:
            raise ValueError("energy metrics require energy_j or fuel_ml")
        for name, value in (("energy_j", self.energy_j), ("fuel_ml", self.fuel_ml)):
            if value is not None and (not math.isfinite(value) or value < 0.0):
                raise ValueError(f"energy metric {name} must be finite and non-negative")

    @property
    def energy_wh(self) -> float | None:
        return None if self.energy_j is None else self.energy_j / 3_600.0

    @property
    def energy_j_per_km(self) -> float | None:
        return self._per_km(self.energy_j)

    @property
    def energy_wh_per_km(self) -> float | None:
        return self._per_km(self.energy_wh)

    @property
    def fuel_ml_per_km(self) -> float | None:
        return self._per_km(self.fuel_ml)

    def _per_km(self, value: float | None) -> float | None:
        if value is None or self.distance_m == 0.0:
            return None
        return value * 1_000.0 / self.distance_m


class EnergyMetricProvider(Protocol):
    """Measure energy over an executed time/speed trace."""

    def measure(self, trace: EnergyTrace) -> EnergyMetrics: ...


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


def _readonly_vector(value: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 1 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite one-dimensional array")
    result = result.copy()
    result.setflags(write=False)
    return result
