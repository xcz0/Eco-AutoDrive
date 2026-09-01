"""Evaluation trace schema, validation, and online recording."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
from tensordict import TensorDictBase

from eco_planner.envs.array_types import SingleObservation
from eco_planner.execution_contracts import EVALUATION_EXECUTION_STEPS, PLANNER_FUTURE_STEPS

if TYPE_CHECKING:
    from eco_planner.envs import TrafficObservationAudit, TrajectoryExecutionRecord


PLANNER_ACTOR_COUNT = 11
PLANNER_STATE_DIM = 4
EXECUTION_PREFIX_STEPS = EVALUATION_EXECUTION_STEPS
TRACE_ARTIFACT_SCHEMA_VERSION = 2

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
    "neighbor_agents_past": ((32, 21, 11), np.dtype(np.float32)),
    "static_objects": ((5, 10), np.dtype(np.float32)),
    "lanes": ((70, 20, 12), np.dtype(np.float32)),
    "lanes_speed_limit": ((70, 1), np.dtype(np.float32)),
    "lanes_has_speed_limit": ((70, 1), np.dtype(np.bool_)),
    "route_lanes": ((25, 20, 12), np.dtype(np.float32)),
    "route_lanes_speed_limit": ((25, 1), np.dtype(np.float32)),
    "route_lanes_has_speed_limit": ((25, 1), np.dtype(np.bool_)),
}

_BASE_TRACE_FIELDS: dict[str, TraceFieldSpec] = {
    "artifact_schema_version": TraceFieldSpec((), np.dtype(np.int64), finite=False),
    "trace_status": TraceFieldSpec((), None, finite=False),
    "warmup_initial_state": TraceFieldSpec((7,), np.dtype(np.float64)),
    "warmup_initial_state_valid": TraceFieldSpec((), np.dtype(np.bool_), finite=False),
    "initial_state": TraceFieldSpec((7,), np.dtype(np.float64)),
    "initial_state_valid": TraceFieldSpec((), np.dtype(np.bool_), finite=False),
    "warmup_states": TraceFieldSpec((_WARMUP, 7), np.dtype(np.float64)),
    "warmup_rewards": TraceFieldSpec((_WARMUP,), np.dtype(np.float64)),
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
        (_PLAN, PLANNER_ACTOR_COUNT, PLANNER_FUTURE_STEPS, PLANNER_STATE_DIM), np.dtype(np.float32)
    ),
    "predictions_local": TraceFieldSpec(
        (_PLAN, PLANNER_ACTOR_COUNT, PLANNER_FUTURE_STEPS, PLANNER_STATE_DIM), np.dtype(np.float32)
    ),
    "ego_predictions_world": TraceFieldSpec(
        (_PLAN, PLANNER_FUTURE_STEPS, PLANNER_STATE_DIM), np.dtype(np.float64)
    ),
    "executed_states": TraceFieldSpec((_SIMULATOR, 7), np.dtype(np.float64)),
    "executed_rewards": TraceFieldSpec((_SIMULATOR,), np.dtype(np.float64)),
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
    "traffic_selected_ids": TraceFieldSpec((_PLAN, 32), np.dtype("<U64"), finite=False),
    "traffic_participant_counts": TraceFieldSpec((_PLAN,), np.dtype(np.int64), finite=False),
    "traffic_static_object_counts": TraceFieldSpec((_PLAN,), np.dtype(np.int64), finite=False),
    "traffic_nearest_distance_m": TraceFieldSpec((_PLAN,), np.dtype(np.float64)),
    "traffic_has_nearest": TraceFieldSpec((_PLAN,), np.dtype(np.bool_), finite=False),
}

_GUIDANCE_TRACE_FIELDS: dict[str, TraceFieldSpec] = {
    "reference_predictions_local": TraceFieldSpec(
        (_PLAN, PLANNER_ACTOR_COUNT, PLANNER_FUTURE_STEPS, PLANNER_STATE_DIM),
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
        (_PLAN, PLANNER_FUTURE_STEPS), np.dtype(np.float32), guided_only=True
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
        "artifact_schema_version",
        "trace_status",
        "warmup_initial_state",
        "warmup_initial_state_valid",
        "initial_state",
        "initial_state_valid",
    }
)


@dataclass(frozen=True)
class LoadedTraceArtifact:
    """Validated current-schema trace arrays."""

    trace_status: str
    arrays: dict[str, np.ndarray]


def trace_shape(spec: TraceFieldSpec, *, plan: int, simulator: int, warmup: int) -> tuple[int, ...]:
    """Resolve declarative axes to a concrete persisted array shape."""

    capacities = {_PLAN: plan, _SIMULATOR: simulator, _WARMUP: warmup}
    return tuple(capacities[axis] if isinstance(axis, str) else axis for axis in spec.axes)


def allocate_trace_arrays(
    max_plan_cycles: int, max_warmup_steps: int, guided: bool
) -> dict[str, np.ndarray]:
    """Allocate all recorder-owned arrays directly from ``TRACE_FIELDS``."""

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


def validate_trace_arrays(
    arrays: Mapping[str, np.ndarray],
    *,
    expected_plan_cycles: int | None = None,
    expected_simulator_steps: int | None = None,
    expected_warmup_steps: int | None = None,
    require_traffic: bool = False,
    expected_trace_status: str | None = None,
    require_finite: bool = True,
) -> None:
    """Validate field declarations and cross-array trace invariants."""

    mapping = arrays
    schema_version = mapping.get("artifact_schema_version")
    if (
        not isinstance(schema_version, np.ndarray)
        or schema_version.shape != ()
        or schema_version.dtype != np.dtype(np.int64)
        or int(schema_version.item()) != TRACE_ARTIFACT_SCHEMA_VERSION
    ):
        raise ValueError(f"trace artifact schema version must be {TRACE_ARTIFACT_SCHEMA_VERSION}")
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
    dynamic_shape = {"plan": None, "simulator": None, "warmup": None}
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
        if (
            require_finite
            and spec.finite
            and value.dtype.kind in "fc"
            and not np.isfinite(value).all()
        ):
            raise ValueError(f"trace array {name!r} contains non-finite values")

    plan_cycles = mapping["initial_noise"].shape[0]
    simulator_steps = mapping["executed_states"].shape[0]
    warmup_steps = mapping["warmup_states"].shape[0]
    trace_status = str(mapping["trace_status"].item())
    if trace_status not in {"complete", "partial", "empty"}:
        raise ValueError("trace status is invalid")
    if expected_trace_status is not None and trace_status != expected_trace_status:
        raise ValueError("trace status disagrees with summary")
    if trace_status == "complete" and (plan_cycles <= 0 or simulator_steps <= 0):
        raise ValueError("complete trace must contain planning and simulator steps")
    if trace_status == "empty" and (plan_cycles or simulator_steps or warmup_steps):
        raise ValueError("empty trace contains recorded steps")
    if trace_status == "complete" and not bool(mapping["initial_state_valid"].item()):
        raise ValueError("complete trace requires a valid initial state")
    if expected_plan_cycles is not None and plan_cycles != expected_plan_cycles:
        raise ValueError("trace planning cycle count disagrees with summary")
    if expected_simulator_steps is not None and simulator_steps != expected_simulator_steps:
        raise ValueError("trace simulator step count disagrees with summary")
    if expected_warmup_steps is not None and warmup_steps != expected_warmup_steps:
        raise ValueError(f"trace must contain exactly {expected_warmup_steps} warmup states")
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
    if require_traffic and not np.any(mapping["traffic_participant_counts"] > 0):
        raise ValueError("trace never observed traffic within the query radius")
    nearest = mapping["traffic_nearest_distance_m"][mapping["traffic_has_nearest"]]
    if np.any(nearest < 0.0):
        raise ValueError("trace nearest traffic distances must be non-negative")


class EpisodeTraceRecorder:
    """Record one evaluation trace"""

    def __init__(
        self,
        initial_state: np.ndarray,
        *,
        max_plan_cycles: int,
        max_warmup_steps: int,
        guided: bool,
        initial_state_valid: bool = True,
    ) -> None:
        if type(max_plan_cycles) is not int or max_plan_cycles < 0:
            raise ValueError("max_plan_cycles must be a non-negative integer")
        if type(max_warmup_steps) is not int or max_warmup_steps < 0:
            raise ValueError("max_warmup_steps must be a non-negative integer")
        if type(guided) is not bool:
            raise TypeError("guided must be a bool")
        initial = np.asarray(initial_state, dtype=np.float64)
        if initial.shape != (7,) or not np.isfinite(initial).all():
            raise ValueError("initial episode state must be a finite [7] array")
        self.warmup_initial_state = initial.copy()
        self.initial_state = initial.copy()
        self.warmup_initial_state_valid = initial_state_valid
        self.initial_state_valid = initial_state_valid
        self._max_plan_cycles = max_plan_cycles
        self._max_warmup_steps = max_warmup_steps
        self._guided = guided
        self._warmup_steps = 0
        self._plan_cycles = 0
        self._simulator_steps = 0
        self._finalized = False
        self._arrays = allocate_trace_arrays(max_plan_cycles, max_warmup_steps, guided)

    @property
    def has_recorded_steps(self) -> bool:
        return bool(self._warmup_steps or self._plan_cycles)

    @property
    def warmup_state_arrays(self) -> tuple[np.ndarray, ...]:
        if not self._warmup_steps:
            return ()
        return (self._arrays["warmup_states"][: self._warmup_steps],)

    def replace_initial_state(self, initial_state: np.ndarray) -> None:
        self._require_open()
        value = np.asarray(initial_state, dtype=np.float64)
        if value.shape != (7,) or not np.isfinite(value).all():
            raise ValueError("initial episode state must be a finite [7] array")
        self.initial_state = value.copy()

    @classmethod
    def from_initial_state(
        cls,
        initial_state: np.ndarray,
        *,
        max_plan_cycles: int,
        max_warmup_steps: int,
        guided: bool,
    ) -> EpisodeTraceRecorder:
        return cls(
            initial_state,
            max_plan_cycles=max_plan_cycles,
            max_warmup_steps=max_warmup_steps,
            guided=guided,
        )

    @classmethod
    def empty(cls) -> EpisodeTraceRecorder:
        """Create a recorder for a failure before a valid simulator state exists."""

        state = np.zeros(7, dtype=np.float64)
        return cls(
            state,
            max_plan_cycles=0,
            max_warmup_steps=0,
            guided=False,
            initial_state_valid=False,
        )

    def append_warmup(
        self,
        execution: TrajectoryExecutionRecord,
        participant_counts: np.ndarray,
        static_object_counts: np.ndarray,
    ) -> None:
        """Append one stationary warmup trajectory action."""

        self._require_open()
        participants = np.asarray(participant_counts, dtype=np.int64)
        static_objects = np.asarray(static_object_counts, dtype=np.int64)
        steps = execution.substep_states.shape[0]
        if participants.shape != (steps,) or static_objects.shape != (steps,):
            raise ValueError("warmup traffic counts must align with execution substeps")
        end = self._warmup_steps + steps
        if end > self._max_warmup_steps:
            raise RuntimeError("warmup trace capacity exceeded")
        target = slice(self._warmup_steps, end)
        for name, value in _execution_arrays(execution).items():
            self._arrays[f"warmup_{name}"][target] = value
        self._arrays["warmup_participant_counts"][target] = participants
        self._arrays["warmup_static_object_counts"][target] = static_objects
        self._warmup_steps = end

    def append_cycle(
        self,
        anchor: np.ndarray,
        observation: SingleObservation,
        inference: TensorDictBase,
        execution: TrajectoryExecutionRecord,
        plan_index: int,
        traffic_audit: TrafficObservationAudit | None,
    ) -> None:
        """Append one planning cycle and its executed simulator prefix."""

        self._require_open()
        substep_states = execution.substep_states
        substep_count = substep_states.shape[0]
        target_centers = execution.target_centers
        target_headings = execution.target_headings
        position_errors_m = execution.position_errors_m
        heading_errors_rad = execution.heading_errors_rad

        if plan_index != self._plan_cycles:
            raise ValueError("planning indices must be contiguous")
        if self._plan_cycles >= self._max_plan_cycles:
            raise RuntimeError("planning trace capacity exceeded")
        end = self._simulator_steps + substep_count
        if end > self._max_plan_cycles * EXECUTION_PREFIX_STEPS:
            raise RuntimeError("simulator-step trace capacity exceeded")
        raw_observation = _raw_observation_for_trace(observation)
        anchor_array = np.asarray(anchor, dtype=np.float64)
        if anchor_array.shape != (7,) or not np.isfinite(anchor_array).all():
            raise ValueError("planning anchor must be a finite [7] array")
        if inference.batch_size != torch.Size([1]):
            raise TypeError("inference must be a batch-one TensorDict")
        cycle = self._plan_cycles
        self._arrays["planning_anchors"][cycle] = anchor_array
        self._arrays["initial_noise"][cycle] = _batch_one(inference["initial_noise"], "noise")
        self._arrays["predictions_local"][cycle] = _batch_one(inference["prediction"], "prediction")
        for name, value in raw_observation.items():
            self._arrays[f"observation_{name}"][cycle] = value
        self._arrays["ego_predictions_world"][cycle] = _world_prediction(execution)
        target = slice(self._simulator_steps, end)
        for name, value in _execution_arrays(execution).items():
            self._arrays[f"executed_{name}"][target] = value
        self._arrays["executed_plan_indices"][target] = plan_index
        self._arrays["trajectory_target_centers"][target] = target_centers
        self._arrays["trajectory_target_headings"][target] = target_headings
        self._arrays["trajectory_position_errors_m"][target] = position_errors_m
        self._arrays["trajectory_heading_errors_rad"][target] = heading_errors_rad
        _write_traffic_audit(self._arrays, cycle, traffic_audit)
        self._write_guidance(cycle, inference)
        self._plan_cycles += 1
        self._simulator_steps = end

    def finalize(self, trace_status: str = "complete") -> dict[str, np.ndarray]:
        """Return validated arrays and reject repeated finalization."""

        self._require_open()
        if trace_status not in {"complete", "partial", "empty"}:
            raise ValueError("trace_status must be complete, partial, or empty")
        if trace_status == "complete" and not self._plan_cycles:
            raise RuntimeError("complete trace must contain planning and simulator steps")
        if trace_status == "empty" and self.has_recorded_steps:
            raise RuntimeError("empty trace cannot contain recorded steps")
        self._finalized = True
        arrays = self._final_arrays(trace_status)
        validate_trace_arrays(
            arrays,
            expected_trace_status=trace_status,
            require_finite=False,
        )
        return arrays

    def _final_arrays(self, trace_status: str) -> dict[str, np.ndarray]:
        arrays: dict[str, np.ndarray] = {
            "artifact_schema_version": np.asarray(
                TRACE_ARTIFACT_SCHEMA_VERSION, dtype=np.int64
            ),
            "trace_status": np.asarray(trace_status),
            "warmup_initial_state": self.warmup_initial_state,
            "warmup_initial_state_valid": np.asarray(
                self.warmup_initial_state_valid, dtype=np.bool_
            ),
            "initial_state": self.initial_state,
            "initial_state_valid": np.asarray(self.initial_state_valid, dtype=np.bool_),
        }
        for name, value in self._arrays.items():
            if name.startswith("warmup_"):
                arrays[name] = value[: self._warmup_steps]
            elif name.startswith(("executed_", "trajectory_")):
                arrays[name] = value[: self._simulator_steps]
            else:
                arrays[name] = value[: self._plan_cycles]
        return arrays

    def _write_guidance(self, cycle: int, result: TensorDictBase) -> None:
        guidance_keys = frozenset(_GUIDANCE_DIAGNOSTIC_NAMES) | {
            "reference_prediction",
            "guidance_action",
        }
        present_keys = frozenset(result.keys())
        guided_result = guidance_keys <= present_keys
        if guidance_keys & present_keys and not guided_result:
            raise ValueError("inference guidance data is incomplete")
        if guided_result != self._guided:
            raise ValueError("inference guidance data disagrees with recorder configuration")
        if not guided_result:
            return
        self._arrays["reference_predictions_local"][cycle] = _batch_one(
            result["reference_prediction"], "reference prediction"
        )
        self._arrays["guidance_actions"][cycle] = _batch_one(
            result["guidance_action"], "guidance action"
        )
        for source, target in _GUIDANCE_DIAGNOSTIC_NAMES.items():
            self._arrays[target][cycle] = _batch_one(result[source], source)

    def _require_open(self) -> None:
        if self._finalized:
            raise RuntimeError("episode trace was already finalized")


_OBSERVATION_ARRAYS = OBSERVATION_FIELDS

_GUIDANCE_DIAGNOSTIC_NAMES = {
    "lateral_target_offset_m": "guidance_lateral_target_offset_m",
    "longitudinal_target_speed_fraction": "guidance_longitudinal_target_speed_fraction",
    "longitudinal_target_speed_delta_mps": "guidance_longitudinal_target_speed_delta_mps",
    "lateral_objective_delta": "guidance_lateral_objective_delta",
    "longitudinal_objective_delta": "guidance_longitudinal_objective_delta",
    "applied_gradient_l2": "guidance_applied_gradient_l2",
    "applied_gradient_max_abs": "guidance_applied_gradient_max_abs",
    "raw_neighbor_gradient_l2": "guidance_raw_neighbor_gradient_l2",
    "zero_speed_count": "guidance_zero_speed_count",
}


def _execution_arrays(execution: TrajectoryExecutionRecord) -> dict[str, np.ndarray]:
    return {
        "states": execution.substep_states,
        "rewards": execution.substep_rewards,
        "native_step_energy_ml": execution.substep_native_energy_ml,
        "native_episode_energy_ml": execution.substep_native_episode_energy_ml,
        "fuel_proxy_step_energy_ml": execution.substep_executed_fuel_proxy_energy_ml,
        "step_distance_m": execution.substep_distance_m,
        "terminated": execution.substep_terminated,
        "truncated": execution.substep_truncated,
    }


def _batch_one(value: torch.Tensor, _name: str) -> np.ndarray:
    return value.numpy()[0]


def _write_traffic_audit(
    arrays: dict[str, np.ndarray],
    cycle: int,
    audit: TrafficObservationAudit | None,
) -> None:
    arrays["traffic_selected_ids"][cycle].fill("")
    if audit is None:
        arrays["traffic_participant_counts"][cycle] = 0
        arrays["traffic_static_object_counts"][cycle] = 0
        arrays["traffic_nearest_distance_m"][cycle] = 0.0
        arrays["traffic_has_nearest"][cycle] = False
        return
    ids = audit.selected_participant_ids
    if len(ids) > 32:
        raise RuntimeError("traffic observation selected more than 32 participants")
    arrays["traffic_selected_ids"][cycle, : len(ids)] = ids
    arrays["traffic_participant_counts"][cycle] = audit.participant_count_in_radius
    arrays["traffic_static_object_counts"][cycle] = audit.static_object_count_in_radius
    nearest = audit.nearest_participant_distance_m
    arrays["traffic_nearest_distance_m"][cycle] = 0.0 if nearest is None else nearest
    arrays["traffic_has_nearest"][cycle] = nearest is not None


def _raw_observation_for_trace(
    observation: SingleObservation,
) -> dict[str, np.ndarray]:
    raw: dict[str, np.ndarray] = {}
    for name in _OBSERVATION_ARRAYS:
        value = observation[name]
        array = value.detach().numpy()
        raw[name] = array
    return raw


def _world_prediction(execution: TrajectoryExecutionRecord) -> np.ndarray:
    centers = execution.world_centers
    headings = execution.world_headings
    return np.column_stack((centers, np.cos(headings), np.sin(headings)))
