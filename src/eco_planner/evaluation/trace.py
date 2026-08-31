"""Online preallocated evaluation trace recording."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from tensordict import TensorDictBase

from eco_planner.envs.array_types import SingleObservation
from eco_planner.evaluation.artifacts import (
    EXECUTION_PREFIX_STEPS,
    OBSERVATION_FIELDS,
    allocate_trace_arrays,
    validate_trace_arrays,
)

if TYPE_CHECKING:
    from eco_planner.envs import TrafficObservationAudit, TrajectoryExecutionRecord


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
        if not isinstance(inference, TensorDictBase) or inference.batch_size != torch.Size([1]):
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


def _batch_one(value: torch.Tensor, name: str) -> np.ndarray:
    if not isinstance(value, torch.Tensor) or value.ndim < 1 or value.shape[0] != 1:
        raise ValueError(f"host inference {name} must be a batch-one tensor")
    if value.device.type != "cpu":
        raise ValueError(f"host inference {name} must remain on CPU")
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
    for name, (shape, dtype) in _OBSERVATION_ARRAYS.items():
        value = observation.get(name)
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
            raise ValueError(f"raw observation {name} has an invalid single-observation shape")
        if value.device.type != "cpu":
            raise ValueError(f"raw observation {name} must remain on CPU")
        array = value.detach().numpy()
        if array.dtype != dtype:
            raise TypeError(f"raw observation {name} has an invalid dtype")
        raw[name] = array
    return raw


def _world_prediction(execution: TrajectoryExecutionRecord) -> np.ndarray:
    centers = execution.world_centers
    headings = execution.world_headings
    return np.column_stack((centers, np.cos(headings), np.sin(headings)))
