"""Minimal auditable longitudinal-dynamics power proxy (Issue #113 Task 2).

A study-scoped second physics candidate for the E-062 fixed-trace review. It
is a point-mass longitudinal power balance over a window-mean speed series:

    P_tractive = m_eff * a * v_mid
               + Crr * m * g * v_mid
               + 0.5 * rho * Cd * A * v_mid^3

with a constant effective path efficiency ``eta = driveline_efficiency *
engine_peak_efficiency``. Positive tractive work burns fuel at that
efficiency; negative tractive work is braking and is fully dissipated (no
regeneration, DFCO: no fuel while braking); the auxiliary and idle loads are
charged at the same efficiency for every window. The model has no gear
shifts, no efficiency map, no thermal states and no grade; every parameter
is frozen in the study configuration and sourced from the pinned FASTSim
vehicle resource. It is an auditable contrast candidate, not a calibrated
energy model, and never enters the online reward or the domain contracts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt


class PhysicsProxyConfig(BaseModel):
    """Frozen point-mass longitudinal-dynamics parameters."""

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid", allow_inf_nan=False)

    vehicle_mass_kg: StrictFloat = Field(gt=0.0)
    wheel_inertia_kg_m2: StrictFloat = Field(ge=0.0)
    wheel_count: StrictInt = Field(gt=0)
    wheel_radius_m: StrictFloat = Field(gt=0.0)
    drag_coefficient: StrictFloat = Field(ge=0.0)
    frontal_area_m2: StrictFloat = Field(gt=0.0)
    rolling_resistance_coefficient: StrictFloat = Field(ge=0.0)
    driveline_efficiency: StrictFloat = Field(gt=0.0, le=1.0)
    engine_peak_efficiency: StrictFloat = Field(gt=0.0, le=1.0)
    air_density_kg_m3: StrictFloat = Field(gt=0.0)
    auxiliary_power_w: StrictFloat = Field(ge=0.0)
    idle_fuel_power_w: StrictFloat = Field(ge=0.0)
    gravity_mps2: StrictFloat = Field(gt=0.0)

    @property
    def effective_mass_kg(self) -> float:
        """Vehicle mass plus the translational equivalent of wheel inertia."""

        return float(
            self.vehicle_mass_kg
            + self.wheel_count * self.wheel_inertia_kg_m2 / self.wheel_radius_m**2
        )

    @property
    def path_efficiency(self) -> float:
        """Constant fuel-conversion efficiency of the tractive request."""

        return self.driveline_efficiency * self.engine_peak_efficiency


@dataclass(frozen=True, slots=True)
class PhysicsProxyResult:
    """Fuel energy and loss decomposition over one speed/duration series."""

    fuel_energy_j: float
    tractive_energy_j: float
    aero_energy_j: float
    rolling_energy_j: float
    inertial_energy_j: float
    braking_dissipated_j: float
    auxiliary_fuel_energy_j: float
    idle_fuel_energy_j: float
    distance_m: float
    window_count: int

    @property
    def fuel_energy_j_per_km(self) -> float | None:
        if self.distance_m == 0.0:
            return None
        return self.fuel_energy_j * 1_000.0 / self.distance_m

    @property
    def fuel_energy_wh_per_km(self) -> float | None:
        per_km = self.fuel_energy_j_per_km
        return None if per_km is None else per_km / 3_600.0


def measure_physics_proxy(
    speeds_mps: np.ndarray,
    durations_s: np.ndarray,
    *,
    initial_speed_mps: float,
    config: PhysicsProxyConfig,
) -> PhysicsProxyResult:
    """Integrate the frozen power balance over window-mean speeds.

    ``speeds_mps[j]`` is the mean speed over window ``j`` lasting
    ``durations_s[j]`` seconds. The speed before the first window is
    ``initial_speed_mps``; for a mid-trip series it equals the first window
    speed (steady entry, zero initial acceleration), for a from-rest
    synthetic profile it is zero.
    """

    speeds = np.asarray(speeds_mps, dtype=np.float64)
    durations = np.asarray(durations_s, dtype=np.float64)
    if speeds.ndim != 1 or durations.ndim != 1 or speeds.size != durations.size:
        raise ValueError("physics proxy requires aligned one-dimensional speed/duration series")
    if speeds.size == 0:
        raise ValueError("physics proxy requires at least one window")
    if not np.isfinite(speeds).all() or not np.isfinite(durations).all():
        raise ValueError("physics proxy requires finite speeds and durations")
    if np.any(speeds < 0.0) or np.any(durations <= 0.0):
        raise ValueError("physics proxy requires non-negative speeds and positive durations")
    if not math.isfinite(initial_speed_mps) or initial_speed_mps < 0.0:
        raise ValueError("physics proxy requires a finite non-negative initial speed")

    previous = np.concatenate([[initial_speed_mps], speeds[:-1]])
    mid_speed = 0.5 * (speeds + previous)
    accel = (speeds - previous) / durations
    effective_mass = config.effective_mass_kg

    aero = (
        0.5
        * config.air_density_kg_m3
        * config.drag_coefficient
        * config.frontal_area_m2
        * mid_speed**3
        * durations
    )
    rolling = (
        config.rolling_resistance_coefficient
        * config.vehicle_mass_kg
        * config.gravity_mps2
        * mid_speed
        * durations
    )
    inertial = effective_mass * accel * mid_speed * durations
    tractive = aero + rolling + inertial
    efficiency = config.path_efficiency
    if efficiency <= 0.0:
        raise ValueError("physics proxy path efficiency must be positive")

    positive = np.maximum(tractive, 0.0)
    braking = np.maximum(-tractive, 0.0)
    auxiliary_fuel = config.auxiliary_power_w * durations / efficiency
    idle_fuel = config.idle_fuel_power_w * durations / efficiency
    fuel = (
        float(positive.sum(dtype=np.float64) / efficiency)
        + float(auxiliary_fuel.sum(dtype=np.float64))
        + float(idle_fuel.sum(dtype=np.float64))
    )
    distance = float((speeds * durations).sum(dtype=np.float64))
    return PhysicsProxyResult(
        fuel_energy_j=fuel,
        tractive_energy_j=float(tractive.sum(dtype=np.float64)),
        aero_energy_j=float(aero.sum(dtype=np.float64)),
        rolling_energy_j=float(rolling.sum(dtype=np.float64)),
        inertial_energy_j=float(inertial.sum(dtype=np.float64)),
        braking_dissipated_j=float(braking.sum(dtype=np.float64)),
        auxiliary_fuel_energy_j=float(auxiliary_fuel.sum(dtype=np.float64)),
        idle_fuel_energy_j=float(idle_fuel.sum(dtype=np.float64)),
        distance_m=distance,
        window_count=int(speeds.size),
    )


__all__ = ["PhysicsProxyConfig", "PhysicsProxyResult", "measure_physics_proxy"]
