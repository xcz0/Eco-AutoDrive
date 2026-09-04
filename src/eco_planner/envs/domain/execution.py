"""Immutable facts emitted by one planner-trajectory execution."""

from __future__ import annotations

from dataclasses import dataclass

from .arrays import (
    ExecutionBooleanArray,
    ExecutionPointArray,
    ExecutionScalarArray,
    ExecutionStateArray,
    WorldHeadingArray,
    WorldPointArray,
    WorldVectorArray,
)
from .traffic import TrafficFrame
from .transition import TransitionMetrics


@dataclass(frozen=True, slots=True)
class TrajectoryExecutionRecord:
    """Target, actual, traffic, and termination facts for one executed prefix."""

    start_center: WorldVectorArray
    start_heading: float
    world_centers: WorldPointArray
    world_headings: WorldHeadingArray
    substep_states: ExecutionStateArray
    target_centers: ExecutionPointArray
    target_headings: ExecutionScalarArray
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


@dataclass(frozen=True, slots=True)
class TrajectoryExecutionResult:
    """Objective-neutral result of one executed planner trajectory prefix."""

    execution: TrajectoryExecutionRecord
    metrics: tuple[TransitionMetrics, ...]
    terminated: bool
    truncated: bool
