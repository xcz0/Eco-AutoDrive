"""Lightweight evaluation trace schema and structural validation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from eco_planner.contracts import (
    AGENT_COUNT,
    AGENT_HISTORY_DIM,
    EVALUATION_EXECUTION_STEPS,
    LANE_COUNT,
    LANE_FEATURE_DIM,
    LANE_POINTS,
    PLANNER_ACTOR_COUNT,
    PLANNER_HORIZON,
    PLANNER_STATE_DIM,
    ROUTE_LANE_COUNT,
    STATIC_OBJECT_COUNT,
    STATIC_OBJECT_DIM,
    TRAFFIC_HISTORY_FRAMES,
)

EXECUTION_PREFIX_STEPS = EVALUATION_EXECUTION_STEPS

_PLAN = "plan"
_SIMULATOR = "simulator"
_WARMUP = "warmup"


@dataclass(frozen=True)
class TraceFieldSpec:
    """Shape and dtype for one persisted trace array."""

    axes: tuple[str | int, ...]
    dtype: np.dtype | None
    guided_only: bool = False
    finite: bool = True


OBSERVATION_FIELDS: dict[str, tuple[tuple[int, ...], np.dtype]] = {
    "ego_current_state": ((10,), np.dtype(np.float32)),
    "neighbor_agents_past": (
        (AGENT_COUNT, TRAFFIC_HISTORY_FRAMES, AGENT_HISTORY_DIM),
        np.dtype(np.float32),
    ),
    "static_objects": ((STATIC_OBJECT_COUNT, STATIC_OBJECT_DIM), np.dtype(np.float32)),
    "lanes": ((LANE_COUNT, LANE_POINTS, LANE_FEATURE_DIM), np.dtype(np.float32)),
    "lanes_speed_limit": ((LANE_COUNT, 1), np.dtype(np.float32)),
    "lanes_has_speed_limit": ((LANE_COUNT, 1), np.dtype(np.bool_)),
    "route_lanes": ((ROUTE_LANE_COUNT, LANE_POINTS, LANE_FEATURE_DIM), np.dtype(np.float32)),
    "route_lanes_speed_limit": ((ROUTE_LANE_COUNT, 1), np.dtype(np.float32)),
    "route_lanes_has_speed_limit": ((ROUTE_LANE_COUNT, 1), np.dtype(np.bool_)),
}

_BASE_TRACE_FIELDS: dict[str, TraceFieldSpec] = {
    "trace_status": TraceFieldSpec((), None, finite=False),
    "warmup_initial_state": TraceFieldSpec((7,), np.dtype(np.float64)),
    "warmup_initial_state_valid": TraceFieldSpec((), np.dtype(np.bool_), finite=False),
    "initial_state": TraceFieldSpec((7,), np.dtype(np.float64)),
    "initial_state_valid": TraceFieldSpec((), np.dtype(np.bool_), finite=False),
    "warmup_states": TraceFieldSpec((_WARMUP, 7), np.dtype(np.float64)),
    "warmup_native_step_energy_ml": TraceFieldSpec((_WARMUP,), np.dtype(np.float64)),
    "warmup_native_episode_energy_ml": TraceFieldSpec((_WARMUP,), np.dtype(np.float64)),
    "warmup_fuel_proxy_step_energy_ml": TraceFieldSpec((_WARMUP,), np.dtype(np.float64)),
    "warmup_step_distance_m": TraceFieldSpec((_WARMUP,), np.dtype(np.float64)),
    "warmup_terminated": TraceFieldSpec((_WARMUP,), np.dtype(np.bool_), finite=False),
    "warmup_truncated": TraceFieldSpec((_WARMUP,), np.dtype(np.bool_), finite=False),
    "warmup_participant_counts": TraceFieldSpec((_WARMUP,), np.dtype(np.int64), finite=False),
    "warmup_static_object_counts": TraceFieldSpec((_WARMUP,), np.dtype(np.int64), finite=False),
    "planning_anchors": TraceFieldSpec((_PLAN, 7), np.dtype(np.float64)),
    "initial_noise": TraceFieldSpec(
        (_PLAN, PLANNER_ACTOR_COUNT, PLANNER_HORIZON, PLANNER_STATE_DIM),
        np.dtype(np.float32),
    ),
    "predictions_local": TraceFieldSpec(
        (_PLAN, PLANNER_ACTOR_COUNT, PLANNER_HORIZON, PLANNER_STATE_DIM),
        np.dtype(np.float32),
    ),
    "ego_predictions_world": TraceFieldSpec(
        (_PLAN, PLANNER_HORIZON, PLANNER_STATE_DIM), np.dtype(np.float64)
    ),
    "executed_states": TraceFieldSpec((_SIMULATOR, 7), np.dtype(np.float64)),
    "executed_native_step_energy_ml": TraceFieldSpec((_SIMULATOR,), np.dtype(np.float64)),
    "executed_native_episode_energy_ml": TraceFieldSpec((_SIMULATOR,), np.dtype(np.float64)),
    "executed_fuel_proxy_step_energy_ml": TraceFieldSpec((_SIMULATOR,), np.dtype(np.float64)),
    "executed_step_distance_m": TraceFieldSpec((_SIMULATOR,), np.dtype(np.float64)),
    "executed_terminated": TraceFieldSpec((_SIMULATOR,), np.dtype(np.bool_), finite=False),
    "executed_truncated": TraceFieldSpec((_SIMULATOR,), np.dtype(np.bool_), finite=False),
    "executed_plan_indices": TraceFieldSpec((_SIMULATOR,), np.dtype(np.int64), finite=False),
    "trajectory_target_centers": TraceFieldSpec((_SIMULATOR, 2), np.dtype(np.float64)),
    "trajectory_target_headings": TraceFieldSpec((_SIMULATOR,), np.dtype(np.float64)),
    "trajectory_position_errors_m": TraceFieldSpec((_SIMULATOR,), np.dtype(np.float64)),
    "trajectory_heading_errors_rad": TraceFieldSpec((_SIMULATOR,), np.dtype(np.float64)),
    "traffic_selected_ids": TraceFieldSpec((_PLAN, AGENT_COUNT), np.dtype("<U64"), finite=False),
    "traffic_participant_counts": TraceFieldSpec((_PLAN,), np.dtype(np.int64), finite=False),
    "traffic_static_object_counts": TraceFieldSpec((_PLAN,), np.dtype(np.int64), finite=False),
    "traffic_nearest_distance_m": TraceFieldSpec((_PLAN,), np.dtype(np.float64)),
    "traffic_has_nearest": TraceFieldSpec((_PLAN,), np.dtype(np.bool_), finite=False),
}

_GUIDANCE_TRACE_FIELDS: dict[str, TraceFieldSpec] = {
    "reference_predictions_local": TraceFieldSpec(
        (_PLAN, PLANNER_ACTOR_COUNT, PLANNER_HORIZON, PLANNER_STATE_DIM),
        np.dtype(np.float32),
        guided_only=True,
    ),
    "guidance_actions": TraceFieldSpec((_PLAN, 2), np.dtype(np.float32), guided_only=True),
    "guidance_lateral_target_offset_m": TraceFieldSpec(
        (_PLAN,), np.dtype(np.float32), guided_only=True
    ),
    "guidance_longitudinal_target_speed_fraction": TraceFieldSpec(
        (_PLAN,), np.dtype(np.float32), guided_only=True
    ),
    "guidance_longitudinal_target_speed_delta_mps": TraceFieldSpec(
        (_PLAN, PLANNER_HORIZON), np.dtype(np.float32), guided_only=True
    ),
    "guidance_lateral_objective_delta": TraceFieldSpec(
        (_PLAN, 5), np.dtype(np.float32), guided_only=True
    ),
    "guidance_longitudinal_objective_delta": TraceFieldSpec(
        (_PLAN, 5), np.dtype(np.float32), guided_only=True
    ),
    "guidance_applied_gradient_l2": TraceFieldSpec(
        (_PLAN, 5), np.dtype(np.float32), guided_only=True
    ),
    "guidance_applied_gradient_max_abs": TraceFieldSpec(
        (_PLAN, 5), np.dtype(np.float32), guided_only=True
    ),
    "guidance_raw_neighbor_gradient_l2": TraceFieldSpec(
        (_PLAN, 5), np.dtype(np.float32), guided_only=True
    ),
    "guidance_zero_speed_count": TraceFieldSpec(
        (_PLAN, 5), np.dtype(np.int64), guided_only=True, finite=False
    ),
}

TRACE_FIELDS = {
    **_BASE_TRACE_FIELDS,
    **{
        f"observation_{name}": TraceFieldSpec((_PLAN, *shape), dtype, finite=dtype.kind == "f")
        for name, (shape, dtype) in OBSERVATION_FIELDS.items()
    },
    **_GUIDANCE_TRACE_FIELDS,
}
GUIDED_TRACE_FIELDS = frozenset(name for name, spec in TRACE_FIELDS.items() if spec.guided_only)
STATIC_TRACE_FIELDS = frozenset(
    {
        "trace_status",
        "warmup_initial_state",
        "warmup_initial_state_valid",
        "initial_state",
        "initial_state_valid",
    }
)


@dataclass(frozen=True)
class LoadedTraceArtifact:
    """Structurally validated trace arrays."""

    trace_status: str
    arrays: dict[str, np.ndarray]


def trace_shape(spec: TraceFieldSpec, *, plan: int, simulator: int, warmup: int) -> tuple[int, ...]:
    """Resolve declarative axes to a concrete persisted array shape."""

    capacities = {_PLAN: plan, _SIMULATOR: simulator, _WARMUP: warmup}
    return tuple(capacities[axis] if isinstance(axis, str) else axis for axis in spec.axes)


def allocate_trace_arrays(
    max_plan_cycles: int, max_warmup_steps: int, guided: bool
) -> dict[str, np.ndarray]:
    """Allocate recorder-owned arrays from the trace field declaration."""

    capacities = {
        "plan": max_plan_cycles,
        "simulator": max_plan_cycles * EXECUTION_PREFIX_STEPS,
        "warmup": max_warmup_steps,
    }
    return {
        name: np.empty(trace_shape(spec, **capacities), dtype=spec.dtype)
        for name, spec in TRACE_FIELDS.items()
        if name not in STATIC_TRACE_FIELDS and (guided or not spec.guided_only)
    }


def validate_trace_arrays(arrays: Mapping[str, np.ndarray]) -> None:
    """Validate persisted fields and cross-array trace invariants once at the I/O boundary."""

    mapping = arrays
    present_guidance = GUIDED_TRACE_FIELDS & set(mapping)
    if present_guidance and present_guidance != GUIDED_TRACE_FIELDS:
        missing = sorted(GUIDED_TRACE_FIELDS - present_guidance)
        raise ValueError(f"guided trace is missing arrays: {missing}")
    expected_fields = {
        name: spec
        for name, spec in TRACE_FIELDS.items()
        if present_guidance or not spec.guided_only
    }
    missing = sorted(set(expected_fields) - set(mapping))
    if missing:
        raise ValueError(f"trace is missing arrays: {missing}")
    unexpected = sorted(set(mapping) - set(expected_fields))
    if unexpected:
        raise ValueError(f"trace contains unexpected arrays: {unexpected}")

    dynamic_shape: dict[str, int | None] = {"plan": None, "simulator": None, "warmup": None}
    for name, spec in expected_fields.items():
        value = mapping[name]
        if not isinstance(value, np.ndarray):
            raise TypeError(f"trace array {name!r} must be a numpy.ndarray")
        expected_shape = tuple(
            dynamic_shape[axis] if isinstance(axis, str) else axis for axis in spec.axes
        )
        if len(value.shape) != len(expected_shape) or any(
            expected is not None and actual != expected
            for actual, expected in zip(value.shape, expected_shape, strict=True)
        ):
            raise ValueError(
                f"trace array {name!r} has shape {value.shape}, expected {expected_shape}"
            )
        if spec.dtype is not None and value.dtype != spec.dtype:
            raise TypeError(f"trace array {name!r} has dtype {value.dtype}, expected {spec.dtype}")
        if spec.finite and value.dtype.kind in "fc" and not np.isfinite(value).all():
            raise ValueError(f"trace array {name!r} contains non-finite values")

    plan_cycles = mapping["initial_noise"].shape[0]
    simulator_steps = mapping["executed_states"].shape[0]
    warmup_steps = mapping["warmup_states"].shape[0]
    trace_status = str(mapping["trace_status"].item())
    if trace_status not in {"complete", "partial", "empty"}:
        raise ValueError("trace status is invalid")
    if trace_status == "complete" and (plan_cycles <= 0 or simulator_steps <= 0):
        raise ValueError("complete trace must contain planning and simulator steps")
    if trace_status == "empty" and (plan_cycles or simulator_steps or warmup_steps):
        raise ValueError("empty trace contains recorded steps")
    if trace_status == "complete" and not bool(mapping["initial_state_valid"].item()):
        raise ValueError("complete trace requires a valid initial state")

    axis_sizes = {_PLAN: plan_cycles, _SIMULATOR: simulator_steps, _WARMUP: warmup_steps}
    for name, spec in expected_fields.items():
        if spec.axes and isinstance(spec.axes[0], str):
            if mapping[name].shape[0] != axis_sizes[spec.axes[0]]:
                raise ValueError(f"trace array {name!r} is not {spec.axes[0]}-aligned")

    for name in (
        "warmup_participant_counts",
        "warmup_static_object_counts",
        "warmup_native_step_energy_ml",
        "warmup_native_episode_energy_ml",
        "warmup_fuel_proxy_step_energy_ml",
        "warmup_step_distance_m",
        "executed_plan_indices",
        "traffic_participant_counts",
        "traffic_static_object_counts",
        "guidance_zero_speed_count",
        "executed_native_step_energy_ml",
        "executed_native_episode_energy_ml",
        "executed_fuel_proxy_step_energy_ml",
        "executed_step_distance_m",
    ):
        if name in mapping and np.any(mapping[name] < 0):
            raise ValueError(f"trace array {name!r} must be non-negative")

    plan_indices = mapping["executed_plan_indices"]
    if plan_cycles:
        if not np.array_equal(np.unique(plan_indices), np.arange(plan_cycles)):
            raise ValueError("trace plan indices are not contiguous")
        counts = np.bincount(plan_indices, minlength=plan_cycles)
        if np.any(counts[:-1] != EXECUTION_PREFIX_STEPS) or not (
            1 <= counts[-1] <= EXECUTION_PREFIX_STEPS
        ):
            raise ValueError(
                f"trace plan indices do not encode {EXECUTION_PREFIX_STEPS}-step prefixes"
            )
        if not np.array_equal(plan_indices, np.repeat(np.arange(plan_cycles), counts)):
            raise ValueError("trace plan indices are not ordered by planning cycle")
    elif simulator_steps:
        raise ValueError("trace has simulator steps without planning cycles")

    terminal = mapping["executed_terminated"] | mapping["executed_truncated"]
    if terminal[:-1].any():
        raise ValueError("trace contains a terminal flag before its final simulator step")
    episode_energy = np.concatenate(
        (
            mapping["warmup_native_episode_energy_ml"],
            mapping["executed_native_episode_energy_ml"],
        )
    )
    if np.any(np.diff(episode_energy) < 0.0):
        raise ValueError("trace episode energy must be cumulative")
    nearest = mapping["traffic_nearest_distance_m"][mapping["traffic_has_nearest"]]
    if np.any(nearest < 0.0):
        raise ValueError("trace nearest traffic distances must be non-negative")
