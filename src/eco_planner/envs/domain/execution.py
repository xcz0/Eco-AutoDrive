"""Immutable facts emitted by one planner-trajectory execution."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from eco_planner.envs.array_types import (
    ExecutionBooleanArray,
    ExecutionPointArray,
    ExecutionScalarArray,
    ExecutionStateArray,
    WorldHeadingArray,
    WorldPointArray,
    WorldVectorArray,
)
from eco_planner.envs.domain.metrics import TransitionMetrics
from eco_planner.envs.domain.traffic import TrafficFrame


@dataclass(frozen=True, slots=True)
class TrajectoryExecutionRecord:
    """Target, actual, metric, reward, and termination facts for one executed prefix."""

    start_center: WorldVectorArray
    start_heading: float
    world_centers: WorldPointArray
    world_headings: WorldHeadingArray
    substep_states: ExecutionStateArray
    target_centers: ExecutionPointArray
    target_headings: ExecutionScalarArray
    position_errors_m: ExecutionScalarArray
    heading_errors_rad: ExecutionScalarArray
    substep_rewards: ExecutionScalarArray
    substep_dense_rewards: ExecutionScalarArray
    substep_native_energy_ml: ExecutionScalarArray
    substep_native_episode_energy_ml: ExecutionScalarArray
    substep_terminated: ExecutionBooleanArray
    substep_truncated: ExecutionBooleanArray
    traffic_frames: tuple[TrafficFrame, ...]
    route_completion: float
    arrive_dest: bool
    out_of_road: bool
    crash_vehicle: bool
    crash_object: bool
    crash_building: bool
    crash_human: bool
    max_step: bool
    crash_sidewalk: bool = False
    substep_executed_fuel_proxy_energy_ml: ExecutionScalarArray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64)
    )
    substep_distance_m: ExecutionScalarArray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64)
    )
    substep_metrics: tuple[TransitionMetrics, ...] = ()
