"""Typed trajectory execution result at the MetaDrive boundary."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from eco_planner.envs.traffic_state import TrafficFrame


@dataclass(frozen=True)
class TrajectoryExecutionRecord:
    start_center: np.ndarray
    start_heading: float
    world_centers: np.ndarray
    world_headings: np.ndarray
    substep_states: np.ndarray
    target_centers: np.ndarray
    target_headings: np.ndarray
    position_errors_m: np.ndarray
    heading_errors_rad: np.ndarray
    substep_rewards: np.ndarray
    substep_dense_rewards: np.ndarray
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
