"""Simulator-independent Pure Pursuit and longitudinal PID feedback control."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, model_validator

from eco_planner.contracts import PLANNER_HORIZON, SIMULATOR_STEP_S

from .arrays import WorldVectorArray
from .geometry import rear_axle_position, shortest_angle_delta, world_points_to_local
from .trajectory import WorldTrajectory


class _StrictControllerModel(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid", allow_inf_nan=False)


class LookaheadConfig(_StrictControllerModel):
    base_m: float = Field(ge=0.0)
    speed_gain_s: float = Field(ge=0.0)
    min_m: float = Field(gt=0.0)
    max_m: float = Field(gt=0.0)

    @model_validator(mode="after")
    def validate_bounds(self) -> LookaheadConfig:
        if self.min_m > self.max_m:
            raise ValueError("lookahead min_m must not exceed max_m")
        return self


class LongitudinalConfig(_StrictControllerModel):
    kp: float = Field(ge=0.0)
    ki: float = Field(ge=0.0)
    kd: float = Field(ge=0.0)
    integral_limit: float = Field(ge=0.0)
    output_limit: float = Field(gt=0.0, le=1.0)


class TrajectoryTrackingConfig(_StrictControllerModel):
    type: Literal["pure_pursuit_pid"]
    version: StrictInt = Field(ge=1, le=1)
    lookahead: LookaheadConfig
    longitudinal: LongitudinalConfig


class TrackingVehicleParameters(_StrictControllerModel):
    """Geometry in metres and steering limit in degrees; forward-only actuation."""

    front_wheelbase_m: float = Field(gt=0.0)
    rear_wheelbase_m: float = Field(gt=0.0)
    max_steering_deg: float = Field(gt=0.0)
    enable_reverse: StrictBool

    @model_validator(mode="after")
    def validate_forward_only(self) -> TrackingVehicleParameters:
        if self.enable_reverse:
            raise ValueError("trajectory tracking requires enable_reverse=False")
        return self


@dataclass(frozen=True, slots=True)
class TrackingVehicleState:
    """Actual world-frame vehicle centre, heading (rad), and velocity (m/s)."""

    center_position: WorldVectorArray
    heading_rad: float
    velocity: WorldVectorArray

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "center_position", _reference_array(self.center_position, (2,), "center_position")
        )
        object.__setattr__(self, "velocity", _reference_array(self.velocity, (2,), "velocity"))
        _finite_scalar(self.heading_rad, "heading_rad")


@dataclass(frozen=True, slots=True)
class TrackingControl:
    """Normalized steering and signed throttle/brake for the vehicle adapter."""

    steering: float
    throttle_brake: float


class TrajectoryTrackingController:
    """Track a cached reference using fresh actual state on every compute call.

    The caller owns reference time and episode reset. No simulator is advanced or
    mutated here; a future backend adapter must call compute at each physics tick.
    """

    def __init__(
        self, config: TrajectoryTrackingConfig, vehicle_parameters: TrackingVehicleParameters
    ) -> None:
        self._config = config
        self._vehicle = vehicle_parameters
        self.reset()

    def reset(self) -> None:
        """Clear reference and PID memory at an episode boundary."""

        self._reference: WorldTrajectory | None = None
        self._reference_speeds: np.ndarray | None = None
        self._rear_axles: np.ndarray | None = None
        self._integral = 0.0
        self._previous_speed: float | None = None

    def set_reference(self, trajectory: WorldTrajectory) -> None:
        """Validate and copy a reference without resetting feedback memory."""

        reference = WorldTrajectory(
            centers=_reference_array(trajectory.centers, (PLANNER_HORIZON + 1, 2), "centers"),
            headings=_reference_array(trajectory.headings, (PLANNER_HORIZON + 1,), "headings"),
            velocities=_reference_array(trajectory.velocities, (PLANNER_HORIZON, 2), "velocities"),
            angular_velocities=_reference_array(
                trajectory.angular_velocities, (PLANNER_HORIZON,), "angular_velocities"
            ),
        )
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            speeds = np.hypot(reference.velocities[:, 0], reference.velocities[:, 1])
            directions = np.column_stack((np.cos(reference.headings), np.sin(reference.headings)))
            rear_axles = reference.centers - self._vehicle.rear_wheelbase_m * directions
        self._reference = reference
        self._reference_speeds = speeds
        self._rear_axles = rear_axles

    def compute(
        self, actual_state: TrackingVehicleState, reference_time_s: float, dt_s: float
    ) -> TrackingControl:
        """Return one feedback action; reference time is seconds since the latest replan."""

        _finite_scalar(reference_time_s, "reference_time_s")
        _finite_scalar(dt_s, "dt_s")
        if reference_time_s < 0.0 or dt_s <= 0.0:
            raise ValueError("reference_time_s must be non-negative and dt_s must be positive")
        if self._reference is None:
            raise RuntimeError("trajectory tracking requires a reference")
        # Raise on non-finite intermediates before actuator/integral bounds could hide them.
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            target_rear, target_speed, next_index = self._sample_reference(reference_time_s)
            heading = actual_state.heading_rad
            direction = np.array([np.cos(heading), np.sin(heading)], dtype=np.float64)
            speed = float(actual_state.velocity @ direction)
            _finite_scalar(speed, "longitudinal speed")
            rear = rear_axle_position(
                actual_state.center_position, heading, self._vehicle.rear_wheelbase_m
            )
            lookahead = self._config.lookahead
            distance = lookahead.base_m + lookahead.speed_gain_s * max(speed, 0.0)
            _finite_scalar(distance, "lookahead distance")
            distance = float(np.clip(distance, lookahead.min_m, lookahead.max_m))
            target = self._lookahead_target(rear, target_rear, next_index, distance)
            local = world_points_to_local(target[None, :], rear, heading)[0]
            curvature = 2.0 * local[1] / np.square(distance)
            wheelbase = self._vehicle.front_wheelbase_m + self._vehicle.rear_wheelbase_m
            _finite_scalar(wheelbase, "wheelbase")
            steering = np.arctan(wheelbase * curvature) / np.deg2rad(self._vehicle.max_steering_deg)
            _finite_scalar(float(steering), "steering")
            throttle_brake, integral = self._longitudinal(target_speed, speed, dt_s)
        self._integral = integral
        self._previous_speed = speed
        return TrackingControl(float(np.clip(steering, -1.0, 1.0)), throttle_brake)

    def _sample_reference(self, time_s: float) -> tuple[np.ndarray, float, int]:
        assert self._reference is not None and self._reference_speeds is not None
        coordinate = min(time_s, PLANNER_HORIZON * SIMULATOR_STEP_S) / SIMULATOR_STEP_S
        left = min(int(coordinate), PLANNER_HORIZON - 1)
        fraction = coordinate - left
        reference = self._reference
        center = (1.0 - fraction) * reference.centers[left] + fraction * reference.centers[left + 1]
        heading_delta = shortest_angle_delta(np.diff(reference.headings[left : left + 2]))[0]
        heading = float(reference.headings[left] + fraction * heading_delta)
        rear = rear_axle_position(center, heading, self._vehicle.rear_wheelbase_m)
        # Velocity samples belong to interval endpoints, not to t=0.
        speed_times = np.arange(1, PLANNER_HORIZON + 1) * SIMULATOR_STEP_S
        speed = float(np.interp(time_s, speed_times, self._reference_speeds))
        _finite_scalar(speed, "reference speed")
        return rear, speed, left + 1

    def _lookahead_target(
        self, rear: np.ndarray, start: np.ndarray, next_index: int, distance: float
    ) -> np.ndarray:
        assert self._rear_axles is not None
        if np.hypot(*(start - rear)) >= distance:
            return start
        for end in self._rear_axles[next_index:]:
            if np.hypot(*(end - rear)) >= distance:
                # First exit from the lookahead circle, in forward path order.
                segment = end - start
                offset = start - rear
                a = float(segment @ segment)
                b = float(offset @ segment)
                c = float(offset @ offset - np.square(distance))
                discriminant = np.square(b) - a * c
                fraction = (-b + np.sqrt(discriminant)) / a
                return start + fraction * segment
            start = end
        return self._rear_axles[-1]

    def _longitudinal(self, target_speed: float, speed: float, dt_s: float) -> tuple[float, float]:
        config = self._config.longitudinal
        error = target_speed - speed
        derivative = 0.0 if self._previous_speed is None else -(speed - self._previous_speed) / dt_s
        candidate = self._integral + error * dt_s
        _finite_scalar(candidate, "integral candidate")
        integral = float(np.clip(candidate, -config.integral_limit, config.integral_limit))
        lower = -config.output_limit
        upper = 0.0 if target_speed == 0.0 else config.output_limit
        proportional_derivative = config.kp * error + config.kd * derivative
        output = proportional_derivative + config.ki * integral
        _finite_scalar(output, "longitudinal output")
        if (output > upper and error > 0.0) or (output < lower and error < 0.0):
            integral = self._integral
            output = proportional_derivative + config.ki * integral
            _finite_scalar(output, "longitudinal output")
        return float(np.clip(output, lower, upper)), integral


def _finite_scalar(value: float, name: str) -> None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be a finite real scalar")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite real scalar")


def _reference_array(value: np.ndarray, shape: tuple[int, ...], name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.dtype != np.float64 or value.shape != shape:
        raise ValueError(f"{name} must be a float64 array with shape {shape}")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")
    result = value.copy()
    result.setflags(write=False)
    return result
