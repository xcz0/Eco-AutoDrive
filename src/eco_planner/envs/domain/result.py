"""Immutable results emitted by a simulator execution boundary."""

from __future__ import annotations

from dataclasses import dataclass

from eco_planner.envs.array_types import (
    ExecutionBooleanArray,
    ExecutionPointArray,
    ExecutionScalarArray,
    ExecutionStateArray,
    WorldHeadingArray,
    WorldPointArray,
    WorldVectorArray,
)
from eco_planner.envs.domain.traffic import TrafficFrame


@dataclass(frozen=True, slots=True)
class EpisodeStatus:
    arrive_dest: bool
    out_of_road: bool
    crash_vehicle: bool
    crash_object: bool
    crash_building: bool
    crash_human: bool
    max_step: bool


@dataclass(frozen=True, slots=True)
class ExecutionTrace:
    start_center: WorldVectorArray
    start_heading: float
    world_centers: WorldPointArray
    world_headings: WorldHeadingArray
    substep_states: ExecutionStateArray
    target_centers: ExecutionPointArray
    target_headings: ExecutionScalarArray
    substep_rewards: ExecutionScalarArray
    substep_dense_rewards: ExecutionScalarArray
    substep_energy_ml: ExecutionScalarArray
    substep_episode_energy_ml: ExecutionScalarArray
    substep_terminated: ExecutionBooleanArray
    substep_truncated: ExecutionBooleanArray
    traffic_frames: tuple[TrafficFrame, ...]


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    trace: ExecutionTrace
    route_completion: float
    status: EpisodeStatus
