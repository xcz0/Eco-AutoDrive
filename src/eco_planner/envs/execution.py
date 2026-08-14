"""Typed trajectory execution result at the MetaDrive boundary."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from eco_planner.envs.traffic_state import TrafficFrame


@dataclass(frozen=True)
class TrajectoryExecutionRecord:
    world_centers: np.ndarray
    world_headings: np.ndarray
    substep_states: np.ndarray
    target_centers: np.ndarray
    target_headings: np.ndarray
    position_errors_m: np.ndarray
    heading_errors_rad: np.ndarray
    substep_rewards: np.ndarray
    substep_terminated: np.ndarray
    substep_truncated: np.ndarray
    traffic_frames: tuple[TrafficFrame, ...]
    route_completion: float
    arrive_dest: bool
    out_of_road: bool
    crash_vehicle: bool
    crash_object: bool
    crash_building: bool
    crash_human: bool
    max_step: bool

    @classmethod
    def from_info(cls, info: Mapping[str, Any]) -> TrajectoryExecutionRecord:
        """Parse the trajectory fields returned by one environment action."""

        states = _array(info, "trajectory_substep_states", np.float64, columns=7)
        steps = states.shape[0]
        if not 1 <= steps <= 5:
            raise ValueError("trajectory execution must contain between one and five substeps")
        return cls(
            world_centers=_array(info, "trajectory_world_centers", np.float64, shape=(80, 2)),
            world_headings=_array(info, "trajectory_world_headings", np.float64, shape=(80,)),
            substep_states=states,
            target_centers=_array(info, "trajectory_target_centers", np.float64, shape=(steps, 2)),
            target_headings=_array(info, "trajectory_target_headings", np.float64, shape=(steps,)),
            position_errors_m=_array(
                info, "trajectory_position_errors_m", np.float64, shape=(steps,)
            ),
            heading_errors_rad=_array(
                info, "trajectory_heading_errors_rad", np.float64, shape=(steps,)
            ),
            substep_rewards=_array(info, "trajectory_substep_rewards", np.float64, shape=(steps,)),
            substep_terminated=_array(
                info, "trajectory_substep_terminated", np.bool_, shape=(steps,)
            ),
            substep_truncated=_array(
                info, "trajectory_substep_truncated", np.bool_, shape=(steps,)
            ),
            traffic_frames=_traffic_frames(info),
            route_completion=_finite_scalar(info, "route_completion"),
            arrive_dest=_boolean(info, "arrive_dest"),
            out_of_road=_boolean(info, "out_of_road"),
            crash_vehicle=_boolean(info, "crash_vehicle"),
            crash_object=_boolean(info, "crash_object"),
            crash_building=_boolean(info, "crash_building"),
            crash_human=_boolean(info, "crash_human"),
            max_step=_boolean(info, "max_step"),
        )


def _array(
    info: Mapping[str, Any],
    name: str,
    dtype: type[np.generic],
    *,
    shape: tuple[int, ...] | None = None,
    columns: int | None = None,
) -> np.ndarray:
    value = np.asarray(info[name], dtype=dtype)
    if shape is not None and value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if columns is not None and (value.ndim != 2 or value.shape[1] != columns):
        raise ValueError(f"{name} must have shape [N, {columns}]")
    if value.dtype.kind in "fc" and not np.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")
    return value.copy()


def _traffic_frames(info: Mapping[str, Any]) -> tuple[TrafficFrame, ...]:
    frames = info.get("traffic_substep_frames")
    if not isinstance(frames, tuple) or not frames:
        raise ValueError("traffic_substep_frames must be a non-empty tuple")
    if not all(isinstance(frame, TrafficFrame) for frame in frames):
        raise TypeError("traffic_substep_frames must contain TrafficFrame values")
    return frames


def _finite_scalar(info: Mapping[str, Any], name: str) -> float:
    value = info.get(name)
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be finite and numeric")
    if not np.isfinite(value):
        raise TypeError(f"{name} must be finite and numeric")
    return float(value)


def _boolean(info: Mapping[str, Any], name: str) -> bool:
    value = info.get(name)
    if type(value) is not bool:
        raise TypeError(f"{name} must be boolean")
    return value
