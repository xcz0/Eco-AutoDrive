"""Objective-neutral transition facts and derived motion/energy metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Protocol, TypeAlias

import numpy as np

from .traffic import TrafficFrame

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


@dataclass(frozen=True, slots=True)
class TransitionMetricInput:
    """Simulator facts for one actual 0.1 s execution transition."""

    previous_position_xy_m: tuple[float, float]
    position_xy_m: tuple[float, float]
    previous_velocity_xy_mps: tuple[float, float]
    velocity_xy_mps: tuple[float, float]
    previous_acceleration_xy_mps2: tuple[float, float]
    heading_rad: float
    yaw_rate_radps: float
    route_progress_delta_m: float
    route_heading_rad: float
    speed_limit_mps: float
    ego_width_m: float
    ego_length_m: float
    traffic_frame: TrafficFrame
    target_position_xy_m: tuple[float, float]
    target_heading_rad: float
    crash_vehicle: bool
    crash_object: bool
    crash_building: bool
    crash_human: bool
    crash_sidewalk: bool
    out_of_road: bool
    native_step_energy_ml: float
    native_episode_energy_ml: float
    timestep_s: float


@dataclass(frozen=True, slots=True)
class TransitionMetrics:
    """Execution facts plus objective-neutral motion and energy quantities."""

    input: TransitionMetricInput
    speed_mps: float
    longitudinal_acceleration_mps2: float
    lateral_acceleration_mps2: float
    jerk_mps3: float
    step_distance_m: float
    position_error_m: float
    heading_error_rad: float
    energy: EnergyMetrics


def derive_transition_metrics(
    step: TransitionMetricInput,
    energy_provider: EnergyMetricProvider,
) -> TransitionMetrics:
    """Derive motion and energy metrics from simulator facts once at the boundary."""

    previous_position = _vector(step.previous_position_xy_m, "previous position")
    position = _vector(step.position_xy_m, "position")
    previous_velocity = _vector(step.previous_velocity_xy_mps, "previous velocity")
    velocity = _vector(step.velocity_xy_mps, "velocity")
    previous_acceleration = _vector(step.previous_acceleration_xy_mps2, "previous acceleration")
    target_position = _vector(step.target_position_xy_m, "target position")
    scalars = {
        "heading": step.heading_rad,
        "yaw rate": step.yaw_rate_radps,
        "route progress": step.route_progress_delta_m,
        "route heading": step.route_heading_rad,
        "speed limit": step.speed_limit_mps,
        "ego width": step.ego_width_m,
        "ego length": step.ego_length_m,
        "target heading": step.target_heading_rad,
        "native step energy": step.native_step_energy_ml,
        "native episode energy": step.native_episode_energy_ml,
        "timestep": step.timestep_s,
    }
    if not all(math.isfinite(value) for value in scalars.values()):
        raise ValueError("transition metric scalars must be finite")
    if step.speed_limit_mps <= 0.0 or step.ego_width_m <= 0.0 or step.ego_length_m <= 0.0:
        raise ValueError("speed limit and ego dimensions must be positive")
    if step.native_step_energy_ml < 0.0 or step.native_episode_energy_ml < 0.0:
        raise ValueError("native MetaDrive energy must be non-negative")
    if step.timestep_s <= 0.0:
        raise ValueError("transition metric timestep must be positive")

    speed_mps = float(np.linalg.norm(velocity))
    step_distance_m = float(np.linalg.norm(position - previous_position))
    acceleration = (velocity - previous_velocity) / step.timestep_s
    forward = np.asarray([math.cos(step.heading_rad), math.sin(step.heading_rad)])
    left = np.asarray([-forward[1], forward[0]])
    return TransitionMetrics(
        input=step,
        speed_mps=speed_mps,
        longitudinal_acceleration_mps2=float(np.dot(acceleration, forward)),
        lateral_acceleration_mps2=float(np.dot(acceleration, left)),
        jerk_mps3=float(np.linalg.norm(acceleration - previous_acceleration) / step.timestep_s),
        step_distance_m=step_distance_m,
        position_error_m=float(np.linalg.norm(position - target_position)),
        heading_error_rad=abs(_shortest_angle_delta(step.heading_rad - step.target_heading_rad)),
        energy=energy_provider.measure(
            EnergyTrace(
                time_s=np.asarray([0.0, step.timestep_s], dtype=np.float64),
                speed_mps=np.asarray(
                    [float(np.linalg.norm(previous_velocity)), speed_mps], dtype=np.float64
                ),
                step_distance_m=np.asarray([step_distance_m], dtype=np.float64),
            )
        ),
    )


def _vector(value: tuple[float, float], name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (2,) or not np.isfinite(result).all():
        raise ValueError(f"transition metric {name} must be a finite 2D vector")
    return result


def _readonly_vector(value: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 1 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite one-dimensional array")
    result = result.copy()
    result.setflags(write=False)
    return result


def _shortest_angle_delta(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


__all__ = [
    "EnergyMetricName",
    "EnergyMetricProvider",
    "EnergyMetrics",
    "EnergyTrace",
    "TransitionMetricInput",
    "TransitionMetrics",
    "derive_transition_metrics",
]
