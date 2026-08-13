"""Closed-loop trace recording and validation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from eco_planner.envs import TrafficObservationAudit, TrajectoryExecutionRecord
from eco_planner.evaluation.schema import ARTIFACT_SCHEMA_VERSION
from eco_planner.models.pretrained import PlannerInferenceResult

PLANNER_ACTOR_COUNT = 11
PLANNER_FUTURE_STEPS = 80
PLANNER_STATE_DIM = 4
EXECUTION_PREFIX_STEPS = 5


@dataclass(frozen=True)
class _WarmupRecord:
    execution: TrajectoryExecutionRecord
    participant_counts: np.ndarray
    static_object_counts: np.ndarray


@dataclass(frozen=True)
class _TrafficAuditRecord:
    selected_ids: np.ndarray
    participant_count: int
    static_object_count: int
    nearest_distance_m: float
    has_nearest: bool


@dataclass(frozen=True)
class _GuidanceRecord:
    arrays: dict[str, np.ndarray]


@dataclass(frozen=True)
class _PlanningCycleRecord:
    anchor: np.ndarray
    observation: dict[str, np.ndarray]
    noise: np.ndarray
    prediction: np.ndarray
    guidance: _GuidanceRecord | None
    ego_prediction_world: np.ndarray
    execution: TrajectoryExecutionRecord
    plan_index: int
    traffic: _TrafficAuditRecord


@dataclass
class EpisodeTraceRecorder:
    """Accumulate one episode and finalize its stable NPZ schema exactly once."""

    warmup_initial_state: np.ndarray
    initial_state: np.ndarray
    warmup_initial_state_valid: bool = True
    initial_state_valid: bool = True
    warmups: list[_WarmupRecord] = field(default_factory=list)
    cycles: list[_PlanningCycleRecord] = field(default_factory=list)
    _finalized: bool = field(default=False, init=False, repr=False)
    _guided: bool | None = field(default=None, init=False, repr=False)

    @property
    def has_recorded_steps(self) -> bool:
        return bool(self.warmups or self.cycles)

    @property
    def warmup_state_arrays(self) -> tuple[np.ndarray, ...]:
        return tuple(record.execution.substep_states for record in self.warmups)

    def replace_initial_state(self, initial_state: np.ndarray) -> None:
        self._require_open()
        value = np.asarray(initial_state, dtype=np.float64)
        if value.shape != (7,) or not np.isfinite(value).all():
            raise ValueError("initial episode state must be a finite [7] array")
        self.initial_state = value.copy()

    @classmethod
    def from_initial_state(cls, initial_state: np.ndarray) -> EpisodeTraceRecorder:
        initial = np.asarray(initial_state, dtype=np.float64)
        if initial.shape != (7,) or not np.isfinite(initial).all():
            raise ValueError("initial episode state must be a finite [7] array")
        return cls(warmup_initial_state=initial.copy(), initial_state=initial.copy())

    @classmethod
    def empty(cls) -> EpisodeTraceRecorder:
        """Create a recorder for a failure before a valid simulator state exists."""

        state = np.zeros(7, dtype=np.float64)
        return cls(
            warmup_initial_state=state.copy(),
            initial_state=state.copy(),
            warmup_initial_state_valid=False,
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
        self.warmups.append(_WarmupRecord(execution, participants, static_objects))

    def append_cycle(
        self,
        anchor: np.ndarray,
        observation: dict[str, torch.Tensor],
        noise: torch.Tensor,
        planner_result: PlannerInferenceResult,
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

        raw_observation = _raw_observation_for_trace(observation)
        anchor_array = np.asarray(anchor, dtype=np.float64)
        if anchor_array.shape != (7,) or not np.isfinite(anchor_array).all():
            raise ValueError("planning anchor must be a finite [7] array")
        if not isinstance(planner_result, PlannerInferenceResult):
            raise TypeError("planner_result must be PlannerInferenceResult")
        guidance = self._guidance_record(planner_result)
        traffic = _traffic_audit_record(traffic_audit)
        self.cycles.append(
            _PlanningCycleRecord(
                anchor=anchor_array.copy(),
                observation=raw_observation,
                noise=noise.detach().cpu().numpy(),
                prediction=planner_result.prediction.detach().cpu().numpy(),
                guidance=guidance,
                ego_prediction_world=_world_prediction(execution),
                execution=execution,
                plan_index=plan_index,
                traffic=traffic,
            )
        )

    def finalize(self, trace_status: str = "complete") -> dict[str, np.ndarray]:
        """Return validated arrays and reject repeated finalization."""

        self._require_open()
        if trace_status not in {"complete", "partial", "empty"}:
            raise ValueError("trace_status must be complete, partial, or empty")
        if trace_status == "complete" and not self.cycles:
            raise RuntimeError("complete trace must contain planning and simulator steps")
        if trace_status == "empty" and self.has_recorded_steps:
            raise RuntimeError("empty trace cannot contain recorded steps")
        self._finalized = True
        arrays = self._base_arrays(trace_status)
        guidance_arrays = [record.guidance for record in self.cycles if record.guidance is not None]
        if guidance_arrays:
            for name in guidance_arrays[0].arrays:
                arrays[name] = np.concatenate(
                    [record.arrays[name] for record in guidance_arrays], axis=0
                )
        validate_trace_arrays(arrays, expected_trace_status=trace_status)
        return arrays

    def _base_arrays(self, trace_status: str) -> dict[str, np.ndarray]:
        executions = [record.execution for record in self.cycles]
        warmups = [record.execution for record in self.warmups]
        arrays = {
            "schema_version": np.asarray(ARTIFACT_SCHEMA_VERSION, dtype=np.int64),
            "trace_status": np.asarray(trace_status),
            "warmup_initial_state": self.warmup_initial_state,
            "warmup_initial_state_valid": np.asarray(
                self.warmup_initial_state_valid, dtype=np.bool_
            ),
            "warmup_states": _concatenate_or_empty(
                [record.substep_states for record in warmups], (0, 7), np.float64
            ),
            "warmup_rewards": _concatenate_or_empty(
                [record.substep_rewards for record in warmups], (0,), np.float64
            ),
            "warmup_terminated": _concatenate_or_empty(
                [record.substep_terminated for record in warmups], (0,), np.bool_
            ),
            "warmup_truncated": _concatenate_or_empty(
                [record.substep_truncated for record in warmups], (0,), np.bool_
            ),
            "warmup_participant_counts": _concatenate_or_empty(
                [record.participant_counts for record in self.warmups], (0,), np.int64
            ),
            "warmup_static_object_counts": _concatenate_or_empty(
                [record.static_object_counts for record in self.warmups], (0,), np.int64
            ),
            "initial_state": self.initial_state,
            "initial_state_valid": np.asarray(self.initial_state_valid, dtype=np.bool_),
            "planning_anchors": _stack_or_empty(
                [record.anchor for record in self.cycles], (0, 7), np.float64
            ),
            "initial_noise": _concatenate_or_empty(
                [record.noise for record in self.cycles],
                (0, PLANNER_ACTOR_COUNT, PLANNER_FUTURE_STEPS, PLANNER_STATE_DIM),
                np.float32,
            ),
            "predictions_local": _concatenate_or_empty(
                [record.prediction for record in self.cycles],
                (0, PLANNER_ACTOR_COUNT, PLANNER_FUTURE_STEPS, PLANNER_STATE_DIM),
                np.float32,
            ),
            "observation_ego_current_state": _stack_or_empty(
                [record.observation["ego_current_state"] for record in self.cycles],
                (0, 10),
                np.float32,
            ),
            "observation_neighbor_agents_past": _stack_or_empty(
                [record.observation["neighbor_agents_past"] for record in self.cycles],
                (0, 32, 21, 11),
                np.float32,
            ),
            "observation_static_objects": _stack_or_empty(
                [record.observation["static_objects"] for record in self.cycles],
                (0, 5, 10),
                np.float32,
            ),
            "observation_lanes": _stack_or_empty(
                [record.observation["lanes"] for record in self.cycles],
                (0, 70, 20, 12),
                np.float32,
            ),
            "observation_lanes_speed_limit": _stack_or_empty(
                [record.observation["lanes_speed_limit"] for record in self.cycles],
                (0, 70, 1),
                np.float32,
            ),
            "observation_lanes_has_speed_limit": _stack_or_empty(
                [record.observation["lanes_has_speed_limit"] for record in self.cycles],
                (0, 70, 1),
                np.bool_,
            ),
            "observation_route_lanes": _stack_or_empty(
                [record.observation["route_lanes"] for record in self.cycles],
                (0, 25, 20, 12),
                np.float32,
            ),
            "observation_route_lanes_speed_limit": _stack_or_empty(
                [record.observation["route_lanes_speed_limit"] for record in self.cycles],
                (0, 25, 1),
                np.float32,
            ),
            "observation_route_lanes_has_speed_limit": _stack_or_empty(
                [record.observation["route_lanes_has_speed_limit"] for record in self.cycles],
                (0, 25, 1),
                np.bool_,
            ),
            "ego_predictions_world": _stack_or_empty(
                [record.ego_prediction_world for record in self.cycles],
                (0, PLANNER_FUTURE_STEPS, PLANNER_STATE_DIM),
                np.float64,
            ),
            "executed_states": _concatenate_or_empty(
                [record.substep_states for record in executions], (0, 7), np.float64
            ),
            "executed_rewards": _concatenate_or_empty(
                [record.substep_rewards for record in executions], (0,), np.float64
            ),
            "executed_terminated": _concatenate_or_empty(
                [record.substep_terminated for record in executions], (0,), np.bool_
            ),
            "executed_truncated": _concatenate_or_empty(
                [record.substep_truncated for record in executions], (0,), np.bool_
            ),
            "executed_plan_indices": _concatenate_or_empty(
                [
                    np.full(record.execution.substep_states.shape[0], record.plan_index, np.int64)
                    for record in self.cycles
                ],
                (0,),
                np.int64,
            ),
            "trajectory_target_centers": _concatenate_or_empty(
                [record.target_centers for record in executions], (0, 2), np.float64
            ),
            "trajectory_target_headings": _concatenate_or_empty(
                [record.target_headings for record in executions], (0,), np.float64
            ),
            "trajectory_position_errors_m": _concatenate_or_empty(
                [record.position_errors_m for record in executions], (0,), np.float64
            ),
            "trajectory_heading_errors_rad": _concatenate_or_empty(
                [record.heading_errors_rad for record in executions], (0,), np.float64
            ),
            "traffic_selected_ids": _stack_or_empty(
                [record.traffic.selected_ids for record in self.cycles],
                (0, 32),
                np.dtype("<U64"),
            ),
            "traffic_participant_counts": np.asarray(
                [record.traffic.participant_count for record in self.cycles], dtype=np.int64
            ),
            "traffic_static_object_counts": np.asarray(
                [record.traffic.static_object_count for record in self.cycles], dtype=np.int64
            ),
            "traffic_nearest_distance_m": np.asarray(
                [record.traffic.nearest_distance_m for record in self.cycles], dtype=np.float64
            ),
            "traffic_has_nearest": np.asarray(
                [record.traffic.has_nearest for record in self.cycles], dtype=np.bool_
            ),
        }
        return arrays

    def _guidance_record(self, result: PlannerInferenceResult) -> _GuidanceRecord | None:
        values = (
            result.reference_prediction,
            result.guidance_action,
            result.guidance_diagnostics,
        )
        guided = all(value is not None for value in values)
        if not guided and any(value is not None for value in values):
            raise ValueError("planner guidance result is incomplete")
        if self._guided is None:
            self._guided = guided
        elif self._guided != guided:
            raise ValueError("episode cannot mix guided and unguided planning cycles")
        if not guided:
            return None
        reference = result.reference_prediction
        action = result.guidance_action
        diagnostics = result.guidance_diagnostics
        if reference is None or action is None or diagnostics is None:
            raise RuntimeError("validated guided result unexpectedly lost audit values")
        return _GuidanceRecord(
            arrays={
                "reference_predictions_local": reference.detach().cpu().numpy(),
                "guidance_actions": action.detach().cpu().numpy(),
                "guidance_lateral_target_offset_m": (
                    diagnostics.lateral_target_offset_m.detach().cpu().numpy()
                ),
                "guidance_longitudinal_target_speed_fraction": (
                    diagnostics.longitudinal_target_speed_fraction.detach().cpu().numpy()
                ),
                "guidance_longitudinal_target_speed_delta_mps": (
                    diagnostics.longitudinal_target_speed_delta_mps.detach().cpu().numpy()
                ),
                "guidance_lateral_objective_delta": (
                    diagnostics.lateral_objective_delta.detach().cpu().numpy()
                ),
                "guidance_longitudinal_objective_delta": (
                    diagnostics.longitudinal_objective_delta.detach().cpu().numpy()
                ),
                "guidance_applied_gradient_l2": (
                    diagnostics.applied_gradient_l2.detach().cpu().numpy()
                ),
                "guidance_applied_gradient_max_abs": (
                    diagnostics.applied_gradient_max_abs.detach().cpu().numpy()
                ),
                "guidance_raw_neighbor_gradient_l2": (
                    diagnostics.raw_neighbor_gradient_l2.detach().cpu().numpy()
                ),
                "guidance_zero_speed_count": (diagnostics.zero_speed_count.detach().cpu().numpy()),
            }
        )

    def _require_open(self) -> None:
        if self._finalized:
            raise RuntimeError("episode trace was already finalized")


def validate_trace_arrays(
    arrays: Mapping[str, np.ndarray],
    *,
    expected_plan_cycles: int | None = None,
    expected_simulator_steps: int | None = None,
    expected_warmup_steps: int | None = None,
    require_traffic: bool = False,
    expected_trace_status: str | None = None,
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
        if value.dtype.kind in "fc" and not np.isfinite(value).all():
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
    names = (
        "ego_current_state",
        "neighbor_agents_past",
        "static_objects",
        "lanes",
        "lanes_speed_limit",
        "lanes_has_speed_limit",
        "route_lanes",
        "route_lanes_speed_limit",
        "route_lanes_has_speed_limit",
    )
    raw: dict[str, np.ndarray] = {}
    for name in names:
        value = observation.get(name)
        if not isinstance(value, torch.Tensor) or value.ndim < 1 or value.shape[0] != 1:
            raise ValueError(f"raw observation {name} must be a batch-one torch tensor")
        raw[name] = value.detach().cpu().numpy()[0].copy()
    return raw


def _world_prediction(execution: TrajectoryExecutionRecord) -> np.ndarray:
    centers = execution.world_centers
    headings = execution.world_headings
    return np.column_stack((centers, np.cos(headings), np.sin(headings)))


def _traffic_audit_record(audit: TrafficObservationAudit | None) -> _TrafficAuditRecord:
    selected_ids = np.full(32, "", dtype="<U64")
    if audit is None:
        return _TrafficAuditRecord(selected_ids, 0, 0, 0.0, False)
    ids = audit.selected_participant_ids
    if len(ids) > selected_ids.size:
        raise RuntimeError("traffic observation selected more than 32 participants")
    selected_ids[: len(ids)] = ids
    nearest = audit.nearest_participant_distance_m
    return _TrafficAuditRecord(
        selected_ids=selected_ids,
        participant_count=audit.participant_count_in_radius,
        static_object_count=audit.static_object_count_in_radius,
        nearest_distance_m=0.0 if nearest is None else nearest,
        has_nearest=nearest is not None,
    )


def _concatenate_or_empty(
    values: list[np.ndarray], empty_shape: tuple[int, ...], dtype: np.dtype[Any]
) -> np.ndarray:
    return np.concatenate(values, axis=0) if values else np.empty(empty_shape, dtype=dtype)


def _stack_or_empty(
    values: list[np.ndarray], empty_shape: tuple[int, ...], dtype: np.dtype[Any]
) -> np.ndarray:
    return np.stack(values) if values else np.empty(empty_shape, dtype=dtype)
