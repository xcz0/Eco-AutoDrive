"""Objective-neutral transition facts and derived motion/energy metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np

from .energy import EnergyMetricProvider, EnergyMetrics, EnergyTrace
from .traffic import TrafficFrame

WRONG_DIRECTION_MAX_HEADING_ERROR_RAD: Final = math.pi / 2.0
STOPPED_SPEED_THRESHOLD_MPS: Final = 0.1


def any_collision(
    crash_vehicle: bool,
    crash_object: bool,
    crash_building: bool,
    crash_human: bool,
    crash_sidewalk: bool,
) -> bool:
    """Canonical ``collision = any(crash_*)`` derivation over raw MetaDrive crash facts."""

    return crash_vehicle or crash_object or crash_building or crash_human or crash_sidewalk


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
    """Execution facts plus objective-neutral motion, behavior, and energy quantities.

    ``heading_error_rad`` is the absolute execution error against the target trajectory
    heading; ``route_heading_error_rad`` is the absolute wrapped error against the route
    forward tangent and is the single source for the ``wrong_direction`` fact.
    """

    input: TransitionMetricInput
    speed_mps: float
    longitudinal_acceleration_mps2: float
    lateral_acceleration_mps2: float
    jerk_mps3: float
    step_distance_m: float
    position_error_m: float
    heading_error_rad: float
    route_heading_error_rad: float
    wrong_direction: bool
    stopped: bool
    collision: bool
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
    route_heading_error_rad = abs(_shortest_angle_delta(step.heading_rad - step.route_heading_rad))
    collision = any_collision(
        step.crash_vehicle,
        step.crash_object,
        step.crash_building,
        step.crash_human,
        step.crash_sidewalk,
    )
    return TransitionMetrics(
        input=step,
        speed_mps=speed_mps,
        longitudinal_acceleration_mps2=float(np.dot(acceleration, forward)),
        lateral_acceleration_mps2=float(np.dot(acceleration, left)),
        jerk_mps3=float(np.linalg.norm(acceleration - previous_acceleration) / step.timestep_s),
        step_distance_m=step_distance_m,
        position_error_m=float(np.linalg.norm(position - target_position)),
        heading_error_rad=abs(_shortest_angle_delta(step.heading_rad - step.target_heading_rad)),
        route_heading_error_rad=route_heading_error_rad,
        wrong_direction=route_heading_error_rad > WRONG_DIRECTION_MAX_HEADING_ERROR_RAD,
        stopped=speed_mps < STOPPED_SPEED_THRESHOLD_MPS,
        collision=collision,
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


def _shortest_angle_delta(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))
