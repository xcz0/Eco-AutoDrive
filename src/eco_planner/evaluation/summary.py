"""Evaluation episode summary construction."""

from __future__ import annotations

from typing import Any

import numpy as np

from eco_planner.envs import TrajectoryExecutionRecord
from eco_planner.execution_contracts import SIMULATOR_STEP_S

from .metrics import compute_episode_metrics, compute_trace_energy
from .models import (
    CompletedEpisodeSummary,
    FailedEpisodeSummary,
    FailurePhase,
)


def build_failed_episode_summary(
    scenario: dict[str, Any],
    *,
    noise_seed: int,
    evaluation_mode: str,
    traffic_density: float,
    sampler: dict[str, Any],
    guidance: dict[str, Any],
    trace_status: str,
    phase: FailurePhase,
    cause: Exception,
    traceback_text: str,
    trace_arrays: dict[str, np.ndarray],
) -> FailedEpisodeSummary:
    """Build a failure summary without inventing unavailable episode metrics."""

    return FailedEpisodeSummary.model_validate(
        {
            "status": "failed",
            "scenario": scenario,
            "evaluation_mode": evaluation_mode,
            "traffic_density": traffic_density,
            "noise_seed": noise_seed,
            "sampler": sampler,
            "guidance": guidance,
            "trace_status": trace_status,
            "energy": compute_trace_energy(trace_arrays),
            "termination": {"type": "runtime_error", "detail": phase.value},
            "failure": {
                "phase": phase,
                "exception_type": type(cause).__name__,
                "message": str(cause),
                "traceback": traceback_text,
            },
        }
    )


def build_episode_summary(
    scenario: dict[str, Any],
    trace_arrays: dict[str, np.ndarray],
    final_execution: TrajectoryExecutionRecord,
    terminated: bool,
    truncated: bool,
    total_reward: float,
    noise_seed: int,
    environment_map_audit: dict[str, object],
    evaluation_mode: str,
    traffic_density: float,
    route_length_m: float,
    sampler: dict[str, Any],
    guidance: dict[str, Any],
) -> CompletedEpisodeSummary:
    """Build the stable per-episode summary JSON payload."""

    metrics = compute_episode_metrics(trace_arrays, final_execution, total_reward=total_reward)
    warmup_states = trace_arrays["warmup_states"]
    warmup_displacement = (
        np.linalg.norm(
            warmup_states[:, :2] - trace_arrays["warmup_initial_state"][None, :2], axis=1
        )
        if warmup_states.size
        else np.empty(0, dtype=np.float64)
    )
    traffic_counts = trace_arrays["traffic_participant_counts"]
    nearest_distances = trace_arrays["traffic_nearest_distance_m"][
        trace_arrays["traffic_has_nearest"]
    ]
    return CompletedEpisodeSummary.model_validate(
        {
            "status": "completed",
            "trace_status": "complete",
            "scenario": scenario,
            "evaluation_mode": evaluation_mode,
            "traffic_density": traffic_density,
            "route_length_m": route_length_m,
            "noise_seed": noise_seed,
            "sampler": sampler,
            "guidance": guidance,
            "plan_cycles": int(trace_arrays["initial_noise"].shape[0]),
            "simulator_steps": int(trace_arrays["executed_states"].shape[0]),
            "environment_steps_including_warmup": int(
                warmup_states.shape[0] + trace_arrays["executed_states"].shape[0]
            ),
            "metrics": metrics,
            "crash_vehicle": final_execution.crash_vehicle,
            "crash_object": final_execution.crash_object,
            "crash_building": final_execution.crash_building,
            "crash_human": final_execution.crash_human,
            "crash_sidewalk": final_execution.crash_sidewalk,
            "terminated": terminated,
            "truncated": truncated,
            "terminal_reason": _terminal_reason(final_execution, terminated, truncated),
            "termination": _termination(final_execution, terminated, truncated),
            "map_input_audit": _map_input_audit(trace_arrays, environment_map_audit),
            "history_warmup": {
                "simulator_steps": int(warmup_states.shape[0]),
                "simulated_seconds": float(warmup_states.shape[0] * SIMULATOR_STEP_S),
                "ego_displacement_m_maximum": float(warmup_displacement.max())
                if warmup_displacement.size
                else 0.0,
                "participant_count_minimum": int(trace_arrays["warmup_participant_counts"].min())
                if trace_arrays["warmup_participant_counts"].size
                else 0,
                "participant_count_maximum": int(trace_arrays["warmup_participant_counts"].max())
                if trace_arrays["warmup_participant_counts"].size
                else 0,
            },
            "traffic_observation": {
                "planning_frames": int(traffic_counts.size),
                "frames_with_participants": int(np.count_nonzero(traffic_counts)),
                "frames_with_participants_fraction": float(np.mean(traffic_counts > 0)),
                "participant_count_minimum": int(traffic_counts.min()),
                "participant_count_maximum": int(traffic_counts.max()),
                "nearest_participant_distance_m_minimum": float(nearest_distances.min())
                if nearest_distances.size
                else None,
            },
            "trajectory_execution_error": {
                "position_m": _error_summary(trace_arrays["trajectory_position_errors_m"]),
                "heading_rad": _error_summary(trace_arrays["trajectory_heading_errors_rad"]),
            },
        }
    )


def _terminal_reason(
    execution: TrajectoryExecutionRecord, terminated: bool, truncated: bool
) -> str:
    ordered_flags = (
        (execution.arrive_dest, "arrive_dest"),
        (execution.out_of_road, "out_of_road"),
        (execution.crash_vehicle, "crash_vehicle"),
        (execution.crash_object, "crash_object"),
        (execution.crash_building, "crash_building"),
        (execution.crash_human, "crash_human"),
        (execution.max_step, "max_step"),
    )
    for active, reason in ordered_flags:
        if active:
            return reason
    if truncated:
        return "truncated"
    if terminated:
        return "terminated"
    raise RuntimeError("episode ended without a terminal reason")


def _termination(
    execution: TrajectoryExecutionRecord, terminated: bool, truncated: bool
) -> dict[str, str]:
    detail = _terminal_reason(execution, terminated, truncated)
    if detail == "arrive_dest":
        kind = "arrive_dest"
    elif detail == "out_of_road":
        kind = "out_of_road"
    elif detail.startswith("crash_"):
        kind = "collision"
    elif detail in {"max_step", "truncated"}:
        kind = "time_truncation"
    else:
        kind = "runtime_error"
    return {"type": kind, "detail": detail}


def _map_input_audit(
    trace_arrays: dict[str, np.ndarray], environment_map_audit: dict[str, object]
) -> dict[str, object]:
    speed_limits = trace_arrays["observation_lanes_speed_limit"]
    has_speed_limit = trace_arrays["observation_lanes_has_speed_limit"]
    valid_counts = has_speed_limit.sum(axis=(1, 2), dtype=np.int64)
    valid_speed_limits = speed_limits[has_speed_limit]
    result = dict(environment_map_audit)
    result.update(
        {
            "valid_lane_count_min": int(valid_counts.min()),
            "valid_lane_count_max": int(valid_counts.max()),
            "speed_limit_valid_count_min": int(valid_counts.min()),
            "speed_limit_valid_count_max": int(valid_counts.max()),
            "speed_limit_mps_min": None,
            "speed_limit_mps_max": None,
            "speed_limit_mps_unique_values": (),
        }
    )
    if valid_speed_limits.size:
        result["speed_limit_mps_min"] = float(valid_speed_limits.min())
        result["speed_limit_mps_max"] = float(valid_speed_limits.max())
        result["speed_limit_mps_unique_values"] = tuple(
            float(value) for value in np.unique(valid_speed_limits)
        )
    return result


def _error_summary(errors: np.ndarray) -> dict[str, float]:
    return {
        "maximum": float(errors.max()),
        "mean": float(errors.mean()),
        "final": float(errors[-1]),
    }
