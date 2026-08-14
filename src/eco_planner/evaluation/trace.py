"""Closed-loop trace recording and validation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np
import torch

from eco_planner.evaluation.inference import HostInferenceResult
from eco_planner.evaluation.schema import ARTIFACT_SCHEMA_VERSION

if TYPE_CHECKING:
    from eco_planner.envs import TrafficObservationAudit, TrajectoryExecutionRecord

PLANNER_ACTOR_COUNT = 11
PLANNER_FUTURE_STEPS = 80
PLANNER_STATE_DIM = 4
EXECUTION_PREFIX_STEPS = 5


class EpisodeTraceRecorder:
    """Record one Artifact v3 trace directly into fixed-capacity arrays."""

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
        self._arrays = _allocate_trace_arrays(max_plan_cycles, max_warmup_steps, guided)

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
        observation: dict[str, torch.Tensor],
        inference: HostInferenceResult,
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
        if target_centers.shape != (substep_count, 2):
            raise RuntimeError("environment returned invalid trajectory target centers")
        expected_shape = (substep_count,)
        if target_headings.shape != expected_shape:
            raise RuntimeError("environment returned invalid trajectory target headings")
        if position_errors_m.shape != expected_shape or heading_errors_rad.shape != expected_shape:
            raise RuntimeError("environment returned invalid trajectory execution errors")

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
        if not isinstance(inference, HostInferenceResult):
            raise TypeError("inference must be HostInferenceResult")
        cycle = self._plan_cycles
        self._arrays["planning_anchors"][cycle] = anchor_array
        self._arrays["initial_noise"][cycle] = _batch_one(inference.initial_noise, "noise")
        self._arrays["predictions_local"][cycle] = _batch_one(inference.prediction, "prediction")
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
            "schema_version": np.asarray(ARTIFACT_SCHEMA_VERSION, dtype=np.int64),
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

    def _write_guidance(self, cycle: int, result: HostInferenceResult) -> None:
        values = (result.reference_prediction, result.guidance_action, result.guidance_diagnostics)
        guided_result = all(value is not None for value in values)
        if guided_result != self._guided:
            raise ValueError("inference guidance data disagrees with recorder configuration")
        if not guided_result:
            return
        reference = result.reference_prediction
        action = result.guidance_action
        diagnostics = result.guidance_diagnostics
        assert reference is not None and action is not None and diagnostics is not None
        self._arrays["reference_predictions_local"][cycle] = _batch_one(
            reference, "reference prediction"
        )
        self._arrays["guidance_actions"][cycle] = _batch_one(action, "guidance action")
        for source, target in _GUIDANCE_DIAGNOSTIC_NAMES.items():
            self._arrays[target][cycle] = _batch_one(getattr(diagnostics, source), source)

    def _require_open(self) -> None:
        if self._finalized:
            raise RuntimeError("episode trace was already finalized")


_OBSERVATION_ARRAYS: dict[str, tuple[tuple[int, ...], np.dtype]] = {
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


def _allocate_trace_arrays(
    max_plan_cycles: int,
    max_warmup_steps: int,
    guided: bool,
) -> dict[str, np.ndarray]:
    max_simulator_steps = max_plan_cycles * EXECUTION_PREFIX_STEPS
    arrays = {
        "warmup_states": np.empty((max_warmup_steps, 7), dtype=np.float64),
        "warmup_rewards": np.empty(max_warmup_steps, dtype=np.float64),
        "warmup_terminated": np.empty(max_warmup_steps, dtype=np.bool_),
        "warmup_truncated": np.empty(max_warmup_steps, dtype=np.bool_),
        "warmup_participant_counts": np.empty(max_warmup_steps, dtype=np.int64),
        "warmup_static_object_counts": np.empty(max_warmup_steps, dtype=np.int64),
        "planning_anchors": np.empty((max_plan_cycles, 7), dtype=np.float64),
        "initial_noise": np.empty(
            (max_plan_cycles, PLANNER_ACTOR_COUNT, PLANNER_FUTURE_STEPS, PLANNER_STATE_DIM),
            dtype=np.float32,
        ),
        "predictions_local": np.empty(
            (max_plan_cycles, PLANNER_ACTOR_COUNT, PLANNER_FUTURE_STEPS, PLANNER_STATE_DIM),
            dtype=np.float32,
        ),
        "ego_predictions_world": np.empty(
            (max_plan_cycles, PLANNER_FUTURE_STEPS, PLANNER_STATE_DIM), dtype=np.float64
        ),
        "executed_states": np.empty((max_simulator_steps, 7), dtype=np.float64),
        "executed_rewards": np.empty(max_simulator_steps, dtype=np.float64),
        "executed_terminated": np.empty(max_simulator_steps, dtype=np.bool_),
        "executed_truncated": np.empty(max_simulator_steps, dtype=np.bool_),
        "executed_plan_indices": np.empty(max_simulator_steps, dtype=np.int64),
        "trajectory_target_centers": np.empty((max_simulator_steps, 2), dtype=np.float64),
        "trajectory_target_headings": np.empty(max_simulator_steps, dtype=np.float64),
        "trajectory_position_errors_m": np.empty(max_simulator_steps, dtype=np.float64),
        "trajectory_heading_errors_rad": np.empty(max_simulator_steps, dtype=np.float64),
        "traffic_selected_ids": np.empty((max_plan_cycles, 32), dtype="<U64"),
        "traffic_participant_counts": np.empty(max_plan_cycles, dtype=np.int64),
        "traffic_static_object_counts": np.empty(max_plan_cycles, dtype=np.int64),
        "traffic_nearest_distance_m": np.empty(max_plan_cycles, dtype=np.float64),
        "traffic_has_nearest": np.empty(max_plan_cycles, dtype=np.bool_),
    }
    for name, (shape, dtype) in _OBSERVATION_ARRAYS.items():
        arrays[f"observation_{name}"] = np.empty((max_plan_cycles, *shape), dtype=dtype)
    if guided:
        arrays.update(
            {
                "reference_predictions_local": np.empty(
                    (
                        max_plan_cycles,
                        PLANNER_ACTOR_COUNT,
                        PLANNER_FUTURE_STEPS,
                        PLANNER_STATE_DIM,
                    ),
                    dtype=np.float32,
                ),
                "guidance_actions": np.empty((max_plan_cycles, 2), dtype=np.float32),
                "guidance_lateral_target_offset_m": np.empty(max_plan_cycles, dtype=np.float32),
                "guidance_longitudinal_target_speed_fraction": np.empty(
                    max_plan_cycles, dtype=np.float32
                ),
                "guidance_longitudinal_target_speed_delta_mps": np.empty(
                    (max_plan_cycles, PLANNER_FUTURE_STEPS), dtype=np.float32
                ),
                "guidance_lateral_objective_delta": np.empty(
                    (max_plan_cycles, 5), dtype=np.float32
                ),
                "guidance_longitudinal_objective_delta": np.empty(
                    (max_plan_cycles, 5), dtype=np.float32
                ),
                "guidance_applied_gradient_l2": np.empty((max_plan_cycles, 5), dtype=np.float32),
                "guidance_applied_gradient_max_abs": np.empty(
                    (max_plan_cycles, 5), dtype=np.float32
                ),
                "guidance_raw_neighbor_gradient_l2": np.empty(
                    (max_plan_cycles, 5), dtype=np.float32
                ),
                "guidance_zero_speed_count": np.empty((max_plan_cycles, 5), dtype=np.int64),
            }
        )
    return arrays


def _execution_arrays(execution: TrajectoryExecutionRecord) -> dict[str, np.ndarray]:
    return {
        "states": execution.substep_states,
        "rewards": execution.substep_rewards,
        "terminated": execution.substep_terminated,
        "truncated": execution.substep_truncated,
    }


def _batch_one(value: np.ndarray, name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.ndim < 1 or value.shape[0] != 1:
        raise ValueError(f"host inference {name} must be a batch-one numpy array")
    return value[0]


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
    """Validate the producer/consumer trace contract."""

    if "schema_version" not in arrays:
        raise ValueError("trace is missing schema version")
    required_shapes: dict[str, tuple[int | None, ...]] = {
        "warmup_initial_state": (7,),
        "warmup_states": (None, 7),
        "warmup_rewards": (None,),
        "warmup_terminated": (None,),
        "warmup_truncated": (None,),
        "warmup_participant_counts": (None,),
        "warmup_static_object_counts": (None,),
        "initial_state": (7,),
        "planning_anchors": (None, 7),
        "initial_noise": (None, PLANNER_ACTOR_COUNT, PLANNER_FUTURE_STEPS, PLANNER_STATE_DIM),
        "predictions_local": (
            None,
            PLANNER_ACTOR_COUNT,
            PLANNER_FUTURE_STEPS,
            PLANNER_STATE_DIM,
        ),
        "observation_ego_current_state": (None, 10),
        "observation_neighbor_agents_past": (None, 32, 21, 11),
        "observation_static_objects": (None, 5, 10),
        "observation_lanes": (None, 70, 20, 12),
        "observation_lanes_speed_limit": (None, 70, 1),
        "observation_lanes_has_speed_limit": (None, 70, 1),
        "observation_route_lanes": (None, 25, 20, 12),
        "ego_predictions_world": (None, PLANNER_FUTURE_STEPS, PLANNER_STATE_DIM),
        "executed_states": (None, 7),
        "executed_rewards": (None,),
        "executed_terminated": (None,),
        "executed_truncated": (None,),
        "executed_plan_indices": (None,),
        "trajectory_target_centers": (None, 2),
        "trajectory_target_headings": (None,),
        "trajectory_position_errors_m": (None,),
        "trajectory_heading_errors_rad": (None,),
        "traffic_selected_ids": (None, 32),
        "traffic_participant_counts": (None,),
        "traffic_static_object_counts": (None,),
        "traffic_nearest_distance_m": (None,),
        "traffic_has_nearest": (None,),
    }
    required_shapes.update(
        {
            "schema_version": (),
            "trace_status": (),
            "warmup_initial_state_valid": (),
            "initial_state_valid": (),
            "observation_route_lanes_speed_limit": (None, 25, 1),
            "observation_route_lanes_has_speed_limit": (None, 25, 1),
        }
    )
    guidance_shapes: dict[str, tuple[int | None, ...]] = {
        "reference_predictions_local": (
            None,
            PLANNER_ACTOR_COUNT,
            PLANNER_FUTURE_STEPS,
            PLANNER_STATE_DIM,
        ),
        "guidance_actions": (None, 2),
        "guidance_lateral_target_offset_m": (None,),
        "guidance_longitudinal_target_speed_fraction": (None,),
        "guidance_longitudinal_target_speed_delta_mps": (None, 80),
        "guidance_lateral_objective_delta": (None, 5),
        "guidance_longitudinal_objective_delta": (None, 5),
        "guidance_applied_gradient_l2": (None, 5),
        "guidance_applied_gradient_max_abs": (None, 5),
        "guidance_raw_neighbor_gradient_l2": (None, 5),
        "guidance_zero_speed_count": (None, 5),
    }
    present_guidance = set(guidance_shapes) & set(arrays)
    if present_guidance and present_guidance != set(guidance_shapes):
        missing_guidance = sorted(set(guidance_shapes) - present_guidance)
        raise ValueError(f"guided trace is missing arrays: {missing_guidance}")
    if present_guidance:
        required_shapes.update(guidance_shapes)
    required_dtypes = {
        "schema_version": np.dtype(np.int64),
        "warmup_initial_state": np.dtype(np.float64),
        "warmup_initial_state_valid": np.dtype(np.bool_),
        "warmup_states": np.dtype(np.float64),
        "warmup_rewards": np.dtype(np.float64),
        "warmup_terminated": np.dtype(np.bool_),
        "warmup_truncated": np.dtype(np.bool_),
        "warmup_participant_counts": np.dtype(np.int64),
        "warmup_static_object_counts": np.dtype(np.int64),
        "initial_state": np.dtype(np.float64),
        "initial_state_valid": np.dtype(np.bool_),
        "planning_anchors": np.dtype(np.float64),
        "initial_noise": np.dtype(np.float32),
        "predictions_local": np.dtype(np.float32),
        "observation_ego_current_state": np.dtype(np.float32),
        "observation_neighbor_agents_past": np.dtype(np.float32),
        "observation_static_objects": np.dtype(np.float32),
        "observation_lanes": np.dtype(np.float32),
        "observation_lanes_speed_limit": np.dtype(np.float32),
        "observation_lanes_has_speed_limit": np.dtype(np.bool_),
        "observation_route_lanes": np.dtype(np.float32),
        "observation_route_lanes_speed_limit": np.dtype(np.float32),
        "observation_route_lanes_has_speed_limit": np.dtype(np.bool_),
        "ego_predictions_world": np.dtype(np.float64),
        "executed_states": np.dtype(np.float64),
        "executed_rewards": np.dtype(np.float64),
        "executed_terminated": np.dtype(np.bool_),
        "executed_truncated": np.dtype(np.bool_),
        "executed_plan_indices": np.dtype(np.int64),
        "trajectory_target_centers": np.dtype(np.float64),
        "trajectory_target_headings": np.dtype(np.float64),
        "trajectory_position_errors_m": np.dtype(np.float64),
        "trajectory_heading_errors_rad": np.dtype(np.float64),
        "traffic_selected_ids": np.dtype("<U64"),
        "traffic_participant_counts": np.dtype(np.int64),
        "traffic_static_object_counts": np.dtype(np.int64),
        "traffic_nearest_distance_m": np.dtype(np.float64),
        "traffic_has_nearest": np.dtype(np.bool_),
        "reference_predictions_local": np.dtype(np.float32),
        "guidance_actions": np.dtype(np.float32),
        "guidance_lateral_target_offset_m": np.dtype(np.float32),
        "guidance_longitudinal_target_speed_fraction": np.dtype(np.float32),
        "guidance_longitudinal_target_speed_delta_mps": np.dtype(np.float32),
        "guidance_lateral_objective_delta": np.dtype(np.float32),
        "guidance_longitudinal_objective_delta": np.dtype(np.float32),
        "guidance_applied_gradient_l2": np.dtype(np.float32),
        "guidance_applied_gradient_max_abs": np.dtype(np.float32),
        "guidance_raw_neighbor_gradient_l2": np.dtype(np.float32),
        "guidance_zero_speed_count": np.dtype(np.int64),
    }
    missing = sorted(set(required_shapes) - set(arrays))
    if missing:
        raise ValueError(f"trace is missing arrays: {missing}")
    unexpected = sorted(set(arrays) - set(required_shapes))
    if unexpected:
        raise ValueError(f"trace contains unexpected arrays: {unexpected}")
    for name, expected_shape in required_shapes.items():
        value = arrays[name]
        if not isinstance(value, np.ndarray):
            raise TypeError(f"trace array {name!r} must be a numpy.ndarray")
        if len(value.shape) != len(expected_shape) or any(
            expected is not None and actual != expected
            for actual, expected in zip(value.shape, expected_shape, strict=True)
        ):
            raise ValueError(
                f"trace array {name!r} has shape {value.shape}, expected {expected_shape}"
            )
        expected_dtype = required_dtypes.get(name)
        if expected_dtype is not None and value.dtype != expected_dtype:
            raise TypeError(
                f"trace array {name!r} has dtype {value.dtype}, expected {expected_dtype}"
            )
        if require_finite and value.dtype.kind in "fc" and not np.isfinite(value).all():
            raise ValueError(f"trace array {name!r} contains non-finite values")

    for name in (
        "warmup_initial_state_valid",
        "initial_state_valid",
        "warmup_terminated",
        "warmup_truncated",
        "observation_lanes_has_speed_limit",
        "observation_route_lanes_has_speed_limit",
        "executed_terminated",
        "executed_truncated",
        "traffic_has_nearest",
    ):
        if name not in arrays:
            continue
        if arrays[name].dtype != np.bool_:
            raise TypeError(f"trace array {name!r} must use bool dtype")
    for name in (
        "warmup_participant_counts",
        "warmup_static_object_counts",
        "executed_plan_indices",
        "traffic_participant_counts",
        "traffic_static_object_counts",
        "guidance_zero_speed_count",
    ):
        if name not in arrays:
            continue
        if arrays[name].dtype.kind not in "iu":
            raise TypeError(f"trace array {name!r} must use an integer dtype")
        if np.any(arrays[name] < 0):
            raise ValueError(f"trace array {name!r} must be non-negative")
    if arrays["traffic_selected_ids"].dtype.kind not in "US":
        raise TypeError("trace array 'traffic_selected_ids' must use a string dtype")

    plan_cycles = arrays["initial_noise"].shape[0]
    simulator_steps = arrays["executed_states"].shape[0]
    warmup_steps = arrays["warmup_states"].shape[0]
    if arrays["schema_version"].item() != ARTIFACT_SCHEMA_VERSION:
        raise ValueError("trace schema version is unsupported")
    trace_status = str(arrays["trace_status"].item())
    if trace_status not in {"complete", "partial", "empty"}:
        raise ValueError("trace status is invalid")
    if expected_trace_status is not None and trace_status != expected_trace_status:
        raise ValueError("trace status disagrees with summary")
    if trace_status == "complete" and (plan_cycles <= 0 or simulator_steps <= 0):
        raise ValueError("complete trace must contain planning and simulator steps")
    if trace_status == "empty" and (plan_cycles or simulator_steps or warmup_steps):
        raise ValueError("empty trace contains recorded steps")
    if trace_status == "complete" and not bool(arrays["initial_state_valid"].item()):
        raise ValueError("complete trace requires a valid initial state")
    if expected_plan_cycles is not None and plan_cycles != expected_plan_cycles:
        raise ValueError("trace planning cycle count disagrees with summary")
    if expected_simulator_steps is not None and simulator_steps != expected_simulator_steps:
        raise ValueError("trace simulator step count disagrees with summary")
    if expected_warmup_steps is not None and warmup_steps != expected_warmup_steps:
        raise ValueError(f"trace must contain exactly {expected_warmup_steps} warmup states")

    for name in (
        "planning_anchors",
        "predictions_local",
        "observation_ego_current_state",
        "observation_neighbor_agents_past",
        "observation_static_objects",
        "observation_lanes",
        "observation_lanes_speed_limit",
        "observation_lanes_has_speed_limit",
        "observation_route_lanes",
        "observation_route_lanes_speed_limit",
        "observation_route_lanes_has_speed_limit",
        "ego_predictions_world",
        "traffic_selected_ids",
        "traffic_participant_counts",
        "traffic_static_object_counts",
        "traffic_nearest_distance_m",
        "traffic_has_nearest",
    ):
        if name not in arrays:
            continue
        if arrays[name].shape[0] != plan_cycles:
            raise ValueError(f"trace array {name!r} is not planning-cycle aligned")
    for name in guidance_shapes:
        if name in arrays and arrays[name].shape[0] != plan_cycles:
            raise ValueError(f"trace array {name!r} is not planning-cycle aligned")
    for name in (
        "executed_rewards",
        "executed_terminated",
        "executed_truncated",
        "executed_plan_indices",
        "trajectory_target_centers",
        "trajectory_target_headings",
        "trajectory_position_errors_m",
        "trajectory_heading_errors_rad",
    ):
        if arrays[name].shape[0] != simulator_steps:
            raise ValueError(f"trace array {name!r} is not simulator-step aligned")
    for name in (
        "warmup_rewards",
        "warmup_terminated",
        "warmup_truncated",
        "warmup_participant_counts",
        "warmup_static_object_counts",
    ):
        if arrays[name].shape[0] != warmup_steps:
            raise ValueError(f"trace array {name!r} is not warmup-step aligned")

    plan_indices = arrays["executed_plan_indices"]
    if plan_cycles:
        if not np.array_equal(np.unique(plan_indices), np.arange(plan_cycles)):
            raise ValueError("trace plan indices are not contiguous")
        counts = np.bincount(plan_indices, minlength=plan_cycles)
        if np.any(counts[:-1] != EXECUTION_PREFIX_STEPS) or not 1 <= counts[-1] <= 5:
            raise ValueError("trace plan indices do not encode five-step prefixes")
        expected_indices = np.repeat(np.arange(plan_cycles), counts)
        if not np.array_equal(plan_indices, expected_indices):
            raise ValueError("trace plan indices are not ordered by planning cycle")
    elif simulator_steps:
        raise ValueError("trace has simulator steps without planning cycles")
    terminal = arrays["executed_terminated"] | arrays["executed_truncated"]
    if terminal[:-1].any():
        raise ValueError("trace contains a terminal flag before its final simulator step")
    if require_traffic and not np.any(arrays["traffic_participant_counts"] > 0):
        raise ValueError("trace never observed traffic within the query radius")
    nearest = arrays["traffic_nearest_distance_m"][arrays["traffic_has_nearest"]]
    if np.any(nearest < 0.0):
        raise ValueError("trace nearest traffic distances must be non-negative")


def _raw_observation_for_trace(
    observation: dict[str, torch.Tensor],
) -> dict[str, np.ndarray]:
    raw: dict[str, np.ndarray] = {}
    for name, (shape, dtype) in _OBSERVATION_ARRAYS.items():
        value = observation.get(name)
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != (1, *shape):
            raise ValueError(f"raw observation {name} has an invalid batch-one shape")
        if value.device.type != "cpu":
            raise ValueError(f"raw observation {name} must remain on CPU")
        array = value.detach().numpy()
        if array.dtype != dtype:
            raise TypeError(f"raw observation {name} has an invalid dtype")
        raw[name] = array[0]
    return raw


def _world_prediction(execution: TrajectoryExecutionRecord) -> np.ndarray:
    centers = execution.world_centers
    headings = execution.world_headings
    return np.column_stack((centers, np.cos(headings), np.sin(headings)))
