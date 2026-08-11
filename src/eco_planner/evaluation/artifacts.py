"""Evaluation summary, metadata, and artifact persistence."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra.utils import to_absolute_path
from metadrive.utils.doc_utils import generate_gif
from omegaconf import DictConfig

from eco_planner.models.pretrained import CheckpointLoadReport


def build_episode_summary(
    scenario: dict[str, Any],
    report: CheckpointLoadReport,
    trace_arrays: dict[str, np.ndarray],
    final_info: Mapping[str, Any],
    terminated: bool,
    truncated: bool,
    total_reward: float,
    noise_seed: int,
    environment_map_audit: dict[str, object],
    evaluation_mode: str,
    traffic_density: float,
    route_length_m: float,
) -> dict[str, Any]:
    """Build the stable per-episode summary JSON payload."""

    positions = np.vstack(
        (trace_arrays["initial_state"][None, :2], trace_arrays["executed_states"][:, :2])
    )
    distance_m = float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())
    speeds = trace_arrays["executed_states"][:, 5]
    position_errors = trace_arrays["trajectory_position_errors_m"]
    heading_errors = trace_arrays["trajectory_heading_errors_rad"]
    warmup_states = trace_arrays["warmup_states"]
    warmup_displacement = (
        np.linalg.norm(
            warmup_states[:, :2] - trace_arrays["warmup_initial_state"][None, :2], axis=1
        )
        if warmup_states.size
        else np.empty(0, dtype=np.float64)
    )
    traffic_counts = trace_arrays["traffic_participant_counts"]
    traffic_has_nearest = trace_arrays["traffic_has_nearest"]
    nearest_distances = trace_arrays["traffic_nearest_distance_m"][traffic_has_nearest]
    return {
        "scenario": scenario,
        "evaluation_mode": evaluation_mode,
        "traffic_density": traffic_density,
        "route_length_m": route_length_m,
        "noise_seed": noise_seed,
        "checkpoint": asdict(report),
        "plan_cycles": int(trace_arrays["initial_noise"].shape[0]),
        "simulator_steps": int(trace_arrays["executed_states"].shape[0]),
        "simulated_seconds": float(trace_arrays["executed_states"].shape[0] * 0.1),
        "environment_steps_including_warmup": int(
            warmup_states.shape[0] + trace_arrays["executed_states"].shape[0]
        ),
        "total_reward": total_reward,
        "distance_m": distance_m,
        "speed_mps": {
            "minimum": float(speeds.min()),
            "mean": float(speeds.mean()),
            "maximum": float(speeds.max()),
        },
        "route_completion": float(final_info["route_completion"]),
        "arrive_dest": bool(final_info["arrive_dest"]),
        "out_of_road": bool(final_info["out_of_road"]),
        "crash_vehicle": bool(final_info["crash_vehicle"]),
        "crash_object": bool(final_info["crash_object"]),
        "crash_building": bool(final_info["crash_building"]),
        "crash_human": bool(final_info["crash_human"]),
        "terminated": terminated,
        "truncated": truncated,
        "terminal_reason": _terminal_reason(final_info, terminated, truncated),
        "map_input_audit": _map_input_audit(trace_arrays, environment_map_audit),
        "history_warmup": {
            "simulator_steps": int(warmup_states.shape[0]),
            "simulated_seconds": float(warmup_states.shape[0] * 0.1),
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
            "position_m": _error_summary(position_errors),
            "heading_rad": _error_summary(heading_errors),
        },
    }


def write_episode_artifacts(
    output_dir: Path,
    trace_arrays: dict[str, np.ndarray],
    frames: list[np.ndarray],
    summary: dict[str, Any],
    video_config: DictConfig,
) -> None:
    """Persist one finalized episode without recomputing trace arrays."""

    output_dir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output_dir / "trace.npz", **trace_arrays)
    write_json(output_dir / "summary.json", summary)
    if video_config.enabled:
        if not frames:
            raise RuntimeError("video output was enabled but no frames were rendered")
        duration_ms = round(1000 / video_config.fps)
        generate_gif(frames, str(output_dir / "closed_loop.gif"), duration=duration_ms)


def write_runtime_metadata(output_dir: Path) -> None:
    """Write reproducibility metadata and the possibly empty tracked diff."""

    repository_root = Path(to_absolute_path("."))
    metadata = {
        "git_head": _git_output(repository_root, "rev-parse", "HEAD").strip(),
        "git_status_short": _git_output(repository_root, "status", "--short").splitlines(),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "metadrive": version("metadrive-simulator"),
    }
    write_json(output_dir / "runtime_metadata.json", metadata)
    (output_dir / "tracked_diff.patch").write_text(
        _git_output(repository_root, "diff", "--binary", "--no-ext-diff"), encoding="utf-8"
    )


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )


def _git_output(repository_root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout


def _terminal_reason(info: Mapping[str, Any], terminated: bool, truncated: bool) -> str:
    ordered_flags = (
        ("arrive_dest", "arrive_dest"),
        ("out_of_road", "out_of_road"),
        ("crash_vehicle", "crash_vehicle"),
        ("crash_object", "crash_object"),
        ("crash_building", "crash_building"),
        ("crash_human", "crash_human"),
        ("max_step", "max_step"),
    )
    for key, reason in ordered_flags:
        if bool(info[key]):
            return reason
    if truncated:
        return "truncated"
    if terminated:
        return "terminated"
    raise RuntimeError("episode ended without a terminal reason")


def _map_input_audit(
    trace_arrays: dict[str, np.ndarray], environment_map_audit: dict[str, object]
) -> dict[str, object]:
    speed_limits = trace_arrays["observation_lanes_speed_limit"]
    has_speed_limit = trace_arrays["observation_lanes_has_speed_limit"]
    if speed_limits.shape != has_speed_limit.shape:
        raise RuntimeError("trace lane speed limits and validity mask have incompatible shapes")
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
            "speed_limit_mps_unique_values": [],
        }
    )
    if valid_speed_limits.size:
        result["speed_limit_mps_min"] = float(valid_speed_limits.min())
        result["speed_limit_mps_max"] = float(valid_speed_limits.max())
        result["speed_limit_mps_unique_values"] = [
            float(value) for value in np.unique(valid_speed_limits)
        ]
    return result


def _error_summary(errors: np.ndarray) -> dict[str, float]:
    if errors.ndim != 1 or not errors.size or not np.isfinite(errors).all():
        raise RuntimeError(
            "trajectory execution errors must be a non-empty finite one-dimensional array"
        )
    return {
        "maximum": float(errors.max()),
        "mean": float(errors.mean()),
        "final": float(errors[-1]),
    }
