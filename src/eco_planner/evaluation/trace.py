"""Closed-loop trace recording and validation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from eco_planner.envs import TrafficObservationAudit
from eco_planner.evaluation.artifacts import ARTIFACT_SCHEMA_VERSION
from eco_planner.models.pretrained import PlannerInferenceResult

PLANNER_ACTOR_COUNT = 11
PLANNER_FUTURE_STEPS = 80
PLANNER_STATE_DIM = 4
EXECUTION_PREFIX_STEPS = 5


@dataclass
class EpisodeTraceRecorder:
    """Accumulate one episode and finalize its stable NPZ schema exactly once."""

    warmup_initial_state: np.ndarray
    initial_state: np.ndarray
    warmup_initial_state_valid: bool = True
    initial_state_valid: bool = True
    warmup_states: list[np.ndarray] = field(default_factory=list)
    warmup_rewards: list[np.ndarray] = field(default_factory=list)
    warmup_terminated: list[np.ndarray] = field(default_factory=list)
    warmup_truncated: list[np.ndarray] = field(default_factory=list)
    warmup_participant_counts: list[np.ndarray] = field(default_factory=list)
    warmup_static_object_counts: list[np.ndarray] = field(default_factory=list)
    planning_anchors: list[np.ndarray] = field(default_factory=list)
    noises: list[np.ndarray] = field(default_factory=list)
    predictions_local: list[np.ndarray] = field(default_factory=list)
    reference_predictions_local: list[np.ndarray] = field(default_factory=list)
    guidance_actions: list[np.ndarray] = field(default_factory=list)
    guidance_lateral_target_offset_m: list[np.ndarray] = field(default_factory=list)
    guidance_longitudinal_target_speed_fraction: list[np.ndarray] = field(default_factory=list)
    guidance_longitudinal_target_speed_delta_mps: list[np.ndarray] = field(default_factory=list)
    guidance_lateral_objective_delta: list[np.ndarray] = field(default_factory=list)
    guidance_longitudinal_objective_delta: list[np.ndarray] = field(default_factory=list)
    guidance_applied_gradient_l2: list[np.ndarray] = field(default_factory=list)
    guidance_applied_gradient_max_abs: list[np.ndarray] = field(default_factory=list)
    guidance_raw_neighbor_gradient_l2: list[np.ndarray] = field(default_factory=list)
    guidance_zero_speed_count: list[np.ndarray] = field(default_factory=list)
    observation_ego_current_state: list[np.ndarray] = field(default_factory=list)
    observation_neighbor_agents_past: list[np.ndarray] = field(default_factory=list)
    observation_static_objects: list[np.ndarray] = field(default_factory=list)
    observation_lanes: list[np.ndarray] = field(default_factory=list)
    observation_lanes_speed_limit: list[np.ndarray] = field(default_factory=list)
    observation_lanes_has_speed_limit: list[np.ndarray] = field(default_factory=list)
    observation_route_lanes: list[np.ndarray] = field(default_factory=list)
    observation_route_lanes_speed_limit: list[np.ndarray] = field(default_factory=list)
    observation_route_lanes_has_speed_limit: list[np.ndarray] = field(default_factory=list)
    ego_world: list[np.ndarray] = field(default_factory=list)
    substep_states: list[np.ndarray] = field(default_factory=list)
    substep_rewards: list[np.ndarray] = field(default_factory=list)
    substep_terminated: list[np.ndarray] = field(default_factory=list)
    substep_truncated: list[np.ndarray] = field(default_factory=list)
    substep_plan_indices: list[np.ndarray] = field(default_factory=list)
    target_centers: list[np.ndarray] = field(default_factory=list)
    target_headings: list[np.ndarray] = field(default_factory=list)
    position_errors_m: list[np.ndarray] = field(default_factory=list)
    heading_errors_rad: list[np.ndarray] = field(default_factory=list)
    traffic_selected_ids: list[np.ndarray] = field(default_factory=list)
    traffic_participant_counts: list[int] = field(default_factory=list)
    traffic_static_object_counts: list[int] = field(default_factory=list)
    traffic_nearest_distance_m: list[float] = field(default_factory=list)
    traffic_has_nearest: list[bool] = field(default_factory=list)
    _finalized: bool = field(default=False, init=False, repr=False)
    _guided: bool | None = field(default=None, init=False, repr=False)

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
        info: Mapping[str, Any],
        participant_counts: np.ndarray,
        static_object_counts: np.ndarray,
    ) -> None:
        """Append one stationary warmup trajectory action."""

        self._require_open()
        self.warmup_states.append(np.asarray(info["trajectory_substep_states"], dtype=np.float64))
        self.warmup_rewards.append(np.asarray(info["trajectory_substep_rewards"], dtype=np.float64))
        self.warmup_terminated.append(
            np.asarray(info["trajectory_substep_terminated"], dtype=np.bool_)
        )
        self.warmup_truncated.append(
            np.asarray(info["trajectory_substep_truncated"], dtype=np.bool_)
        )
        self.warmup_participant_counts.append(np.asarray(participant_counts, dtype=np.int64))
        self.warmup_static_object_counts.append(np.asarray(static_object_counts, dtype=np.int64))

    def append_cycle(
        self,
        anchor: np.ndarray,
        observation: dict[str, torch.Tensor],
        noise: torch.Tensor,
        planner_result: PlannerInferenceResult,
        info: Mapping[str, Any],
        plan_index: int,
        traffic_audit: TrafficObservationAudit | None,
    ) -> None:
        """Append one planning cycle and its executed simulator prefix."""

        self._require_open()
        substep_states = np.asarray(info["trajectory_substep_states"], dtype=np.float64)
        if substep_states.ndim != 2 or substep_states.shape[1] != 7:
            raise RuntimeError("environment returned invalid trajectory substep states")
        substep_count = substep_states.shape[0]
        target_centers = np.asarray(info["trajectory_target_centers"], dtype=np.float64)
        target_headings = np.asarray(info["trajectory_target_headings"], dtype=np.float64)
        position_errors_m = np.asarray(info["trajectory_position_errors_m"], dtype=np.float64)
        heading_errors_rad = np.asarray(info["trajectory_heading_errors_rad"], dtype=np.float64)
        if target_centers.shape != (substep_count, 2):
            raise RuntimeError("environment returned invalid trajectory target centers")
        expected_shape = (substep_count,)
        if target_headings.shape != expected_shape:
            raise RuntimeError("environment returned invalid trajectory target headings")
        if position_errors_m.shape != expected_shape or heading_errors_rad.shape != expected_shape:
            raise RuntimeError("environment returned invalid trajectory execution errors")

        raw_observation = _raw_observation_for_trace(observation)
        self.planning_anchors.append(np.asarray(anchor, dtype=np.float64).copy())
        self.noises.append(noise.detach().cpu().numpy())
        if not isinstance(planner_result, PlannerInferenceResult):
            raise TypeError("planner_result must be PlannerInferenceResult")
        self.predictions_local.append(planner_result.prediction.detach().cpu().numpy())
        self._append_guidance(planner_result)
        self.observation_ego_current_state.append(raw_observation["ego_current_state"])
        self.observation_neighbor_agents_past.append(raw_observation["neighbor_agents_past"])
        self.observation_static_objects.append(raw_observation["static_objects"])
        self.observation_lanes.append(raw_observation["lanes"])
        self.observation_lanes_speed_limit.append(raw_observation["lanes_speed_limit"])
        self.observation_lanes_has_speed_limit.append(raw_observation["lanes_has_speed_limit"])
        self.observation_route_lanes.append(raw_observation["route_lanes"])
        self.observation_route_lanes_speed_limit.append(raw_observation["route_lanes_speed_limit"])
        self.observation_route_lanes_has_speed_limit.append(
            raw_observation["route_lanes_has_speed_limit"]
        )
        self.ego_world.append(_world_prediction(info))
        self.substep_states.append(substep_states)
        self.substep_rewards.append(
            np.asarray(info["trajectory_substep_rewards"], dtype=np.float64)
        )
        self.substep_terminated.append(
            np.asarray(info["trajectory_substep_terminated"], dtype=np.bool_)
        )
        self.substep_truncated.append(
            np.asarray(info["trajectory_substep_truncated"], dtype=np.bool_)
        )
        self.substep_plan_indices.append(np.full(substep_count, plan_index, dtype=np.int64))
        self.target_centers.append(target_centers)
        self.target_headings.append(target_headings)
        self.position_errors_m.append(position_errors_m)
        self.heading_errors_rad.append(heading_errors_rad)
        self._append_traffic_audit(traffic_audit)

    def finalize(self, trace_status: str = "complete") -> dict[str, np.ndarray]:
        """Return validated arrays and reject repeated finalization."""

        self._require_open()
        if trace_status not in {"complete", "partial", "empty"}:
            raise ValueError("trace_status must be complete, partial, or empty")
        if trace_status == "complete" and (not self.noises or not self.substep_states):
            raise RuntimeError("complete trace must contain planning and simulator steps")
        if trace_status == "empty" and (self.noises or self.substep_states or self.warmup_states):
            raise RuntimeError("empty trace cannot contain recorded steps")
        self._finalized = True
        arrays = {
            "schema_version": np.asarray(ARTIFACT_SCHEMA_VERSION, dtype=np.int64),
            "trace_status": np.asarray(trace_status),
            "warmup_initial_state": self.warmup_initial_state,
            "warmup_initial_state_valid": np.asarray(
                self.warmup_initial_state_valid, dtype=np.bool_
            ),
            "warmup_states": _concatenate_or_empty(self.warmup_states, (0, 7), np.float64),
            "warmup_rewards": _concatenate_or_empty(self.warmup_rewards, (0,), np.float64),
            "warmup_terminated": _concatenate_or_empty(self.warmup_terminated, (0,), np.bool_),
            "warmup_truncated": _concatenate_or_empty(self.warmup_truncated, (0,), np.bool_),
            "warmup_participant_counts": _concatenate_or_empty(
                self.warmup_participant_counts, (0,), np.int64
            ),
            "warmup_static_object_counts": _concatenate_or_empty(
                self.warmup_static_object_counts, (0,), np.int64
            ),
            "initial_state": self.initial_state,
            "initial_state_valid": np.asarray(self.initial_state_valid, dtype=np.bool_),
            "planning_anchors": _stack_or_empty(self.planning_anchors, (0, 7), np.float64),
            "initial_noise": _concatenate_or_empty(
                self.noises,
                (0, PLANNER_ACTOR_COUNT, PLANNER_FUTURE_STEPS, PLANNER_STATE_DIM),
                np.float32,
            ),
            "predictions_local": _concatenate_or_empty(
                self.predictions_local,
                (0, PLANNER_ACTOR_COUNT, PLANNER_FUTURE_STEPS, PLANNER_STATE_DIM),
                np.float32,
            ),
            "observation_ego_current_state": _stack_or_empty(
                self.observation_ego_current_state, (0, 10), np.float32
            ),
            "observation_neighbor_agents_past": _stack_or_empty(
                self.observation_neighbor_agents_past, (0, 32, 21, 11), np.float32
            ),
            "observation_static_objects": _stack_or_empty(
                self.observation_static_objects, (0, 5, 10), np.float32
            ),
            "observation_lanes": _stack_or_empty(
                self.observation_lanes, (0, 70, 20, 12), np.float32
            ),
            "observation_lanes_speed_limit": _stack_or_empty(
                self.observation_lanes_speed_limit, (0, 70, 1), np.float32
            ),
            "observation_lanes_has_speed_limit": _stack_or_empty(
                self.observation_lanes_has_speed_limit, (0, 70, 1), np.bool_
            ),
            "observation_route_lanes": _stack_or_empty(
                self.observation_route_lanes, (0, 25, 20, 12), np.float32
            ),
            "observation_route_lanes_speed_limit": _stack_or_empty(
                self.observation_route_lanes_speed_limit, (0, 25, 1), np.float32
            ),
            "observation_route_lanes_has_speed_limit": _stack_or_empty(
                self.observation_route_lanes_has_speed_limit, (0, 25, 1), np.bool_
            ),
            "ego_predictions_world": _stack_or_empty(
                self.ego_world, (0, PLANNER_FUTURE_STEPS, PLANNER_STATE_DIM), np.float64
            ),
            "executed_states": _concatenate_or_empty(self.substep_states, (0, 7), np.float64),
            "executed_rewards": _concatenate_or_empty(self.substep_rewards, (0,), np.float64),
            "executed_terminated": _concatenate_or_empty(self.substep_terminated, (0,), np.bool_),
            "executed_truncated": _concatenate_or_empty(self.substep_truncated, (0,), np.bool_),
            "executed_plan_indices": _concatenate_or_empty(
                self.substep_plan_indices, (0,), np.int64
            ),
            "trajectory_target_centers": _concatenate_or_empty(
                self.target_centers, (0, 2), np.float64
            ),
            "trajectory_target_headings": _concatenate_or_empty(
                self.target_headings, (0,), np.float64
            ),
            "trajectory_position_errors_m": _concatenate_or_empty(
                self.position_errors_m, (0,), np.float64
            ),
            "trajectory_heading_errors_rad": _concatenate_or_empty(
                self.heading_errors_rad, (0,), np.float64
            ),
            "traffic_selected_ids": _stack_or_empty(
                self.traffic_selected_ids, (0, 32), np.dtype("<U64")
            ),
            "traffic_participant_counts": np.asarray(
                self.traffic_participant_counts, dtype=np.int64
            ),
            "traffic_static_object_counts": np.asarray(
                self.traffic_static_object_counts, dtype=np.int64
            ),
            "traffic_nearest_distance_m": np.asarray(
                self.traffic_nearest_distance_m, dtype=np.float64
            ),
            "traffic_has_nearest": np.asarray(self.traffic_has_nearest, dtype=np.bool_),
        }
        if self._guided:
            arrays.update(
                {
                    "reference_predictions_local": np.concatenate(
                        self.reference_predictions_local, axis=0
                    ),
                    "guidance_actions": np.concatenate(self.guidance_actions, axis=0),
                    "guidance_lateral_target_offset_m": np.concatenate(
                        self.guidance_lateral_target_offset_m, axis=0
                    ),
                    "guidance_longitudinal_target_speed_fraction": np.concatenate(
                        self.guidance_longitudinal_target_speed_fraction, axis=0
                    ),
                    "guidance_longitudinal_target_speed_delta_mps": np.concatenate(
                        self.guidance_longitudinal_target_speed_delta_mps, axis=0
                    ),
                    "guidance_lateral_objective_delta": np.concatenate(
                        self.guidance_lateral_objective_delta, axis=0
                    ),
                    "guidance_longitudinal_objective_delta": np.concatenate(
                        self.guidance_longitudinal_objective_delta, axis=0
                    ),
                    "guidance_applied_gradient_l2": np.concatenate(
                        self.guidance_applied_gradient_l2, axis=0
                    ),
                    "guidance_applied_gradient_max_abs": np.concatenate(
                        self.guidance_applied_gradient_max_abs, axis=0
                    ),
                    "guidance_raw_neighbor_gradient_l2": np.concatenate(
                        self.guidance_raw_neighbor_gradient_l2, axis=0
                    ),
                    "guidance_zero_speed_count": np.concatenate(
                        self.guidance_zero_speed_count, axis=0
                    ),
                }
            )
        validate_trace_arrays(arrays, expected_trace_status=trace_status)
        return arrays

    def _append_guidance(self, result: PlannerInferenceResult) -> None:
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
            return
        reference = result.reference_prediction
        action = result.guidance_action
        diagnostics = result.guidance_diagnostics
        if reference is None or action is None or diagnostics is None:
            raise RuntimeError("validated guided result unexpectedly lost audit values")
        self.reference_predictions_local.append(reference.detach().cpu().numpy())
        self.guidance_actions.append(action.detach().cpu().numpy())
        self.guidance_lateral_target_offset_m.append(
            diagnostics.lateral_target_offset_m.detach().cpu().numpy()
        )
        self.guidance_longitudinal_target_speed_fraction.append(
            diagnostics.longitudinal_target_speed_fraction.detach().cpu().numpy()
        )
        self.guidance_longitudinal_target_speed_delta_mps.append(
            diagnostics.longitudinal_target_speed_delta_mps.detach().cpu().numpy()
        )
        for target, value in (
            (self.guidance_lateral_objective_delta, diagnostics.lateral_objective_delta),
            (
                self.guidance_longitudinal_objective_delta,
                diagnostics.longitudinal_objective_delta,
            ),
            (self.guidance_applied_gradient_l2, diagnostics.applied_gradient_l2),
            (
                self.guidance_applied_gradient_max_abs,
                diagnostics.applied_gradient_max_abs,
            ),
            (
                self.guidance_raw_neighbor_gradient_l2,
                diagnostics.raw_neighbor_gradient_l2,
            ),
            (self.guidance_zero_speed_count, diagnostics.zero_speed_count),
        ):
            target.append(value.detach().cpu().numpy())

    def _append_traffic_audit(self, audit: TrafficObservationAudit | None) -> None:
        selected_ids = np.full(32, "", dtype="<U64")
        if audit is None:
            participant_count = 0
            static_count = 0
            nearest_distance = 0.0
            has_nearest = False
        else:
            ids = audit.selected_participant_ids
            if len(ids) > selected_ids.size:
                raise RuntimeError("traffic observation selected more than 32 participants")
            selected_ids[: len(ids)] = ids
            participant_count = audit.participant_count_in_radius
            static_count = audit.static_object_count_in_radius
            nearest = audit.nearest_participant_distance_m
            nearest_distance = 0.0 if nearest is None else nearest
            has_nearest = nearest is not None
        self.traffic_selected_ids.append(selected_ids)
        self.traffic_participant_counts.append(participant_count)
        self.traffic_static_object_counts.append(static_count)
        self.traffic_nearest_distance_m.append(nearest_distance)
        self.traffic_has_nearest.append(has_nearest)

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

    is_v1 = "schema_version" not in arrays
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
    if not is_v1:
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
    for name, expected_shape in required_shapes.items():
        value = arrays[name]
        if not isinstance(value, np.ndarray):
            raise TypeError(f"trace array {name!r} must be a numpy.ndarray")
        if len(value.shape) != len(expected_shape) or any(
            expected is not None and actual != expected
            for actual, expected in zip(value.shape, expected_shape)
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
    if not is_v1 and arrays["schema_version"].item() != ARTIFACT_SCHEMA_VERSION:
        raise ValueError("trace schema version is unsupported")
    trace_status = "complete" if is_v1 else str(arrays["trace_status"].item())
    if trace_status not in {"complete", "partial", "empty"}:
        raise ValueError("trace status is invalid")
    if expected_trace_status is not None and trace_status != expected_trace_status:
        raise ValueError("trace status disagrees with summary")
    if trace_status == "complete" and (plan_cycles <= 0 or simulator_steps <= 0):
        raise ValueError("complete trace must contain planning and simulator steps")
    if trace_status == "empty" and (plan_cycles or simulator_steps or warmup_steps):
        raise ValueError("empty trace contains recorded steps")
    if not is_v1 and trace_status == "complete" and not bool(arrays["initial_state_valid"].item()):
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


def _world_prediction(info: Mapping[str, Any]) -> np.ndarray:
    centers = np.asarray(info["trajectory_world_centers"], dtype=np.float64)
    headings = np.asarray(info["trajectory_world_headings"], dtype=np.float64)
    if centers.shape != (80, 2) or headings.shape != (80,):
        raise RuntimeError("environment returned an invalid world trajectory")
    return np.column_stack((centers, np.cos(headings), np.sin(headings)))


def _concatenate_or_empty(
    values: list[np.ndarray], empty_shape: tuple[int, ...], dtype: np.dtype[Any]
) -> np.ndarray:
    return np.concatenate(values, axis=0) if values else np.empty(empty_shape, dtype=dtype)


def _stack_or_empty(
    values: list[np.ndarray], empty_shape: tuple[int, ...], dtype: np.dtype[Any]
) -> np.ndarray:
    return np.stack(values) if values else np.empty(empty_shape, dtype=dtype)
