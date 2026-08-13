from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from eco_planner.evaluation.matrix import build_matrix_report, summarize_matrix
from eco_planner.evaluation.schema import ARTIFACT_SCHEMA_VERSION
from eco_planner.evaluation.trace import EpisodeTraceRecorder


def _sampler() -> dict[str, object]:
    return {
        "name": "dpm10",
        "implementation": "diffusers",
        "num_steps": 10,
        "timesteps": None,
        "initial_noise_scale": 0.5,
        "ddim_stochasticity": 0.0,
        "parity_label": "official_diffusion_planner_baseline",
    }


def _map_audit() -> dict[str, object]:
    return {
        "speed_limit_sentinel_replaced_count": 1,
        "speed_limit_existing_preserved_count": 0,
        "configured_programmatic_lane_speed_limit_kmh": 50.0,
        "lane_speed_limit_kmh_counts": {"50": 1},
        "valid_lane_count_min": 1,
        "valid_lane_count_max": 1,
        "speed_limit_valid_count_min": 1,
        "speed_limit_valid_count_max": 1,
        "speed_limit_mps_min": 50.0 / 3.6,
        "speed_limit_mps_max": 50.0 / 3.6,
        "speed_limit_mps_unique_values": [50.0 / 3.6],
    }


def _execution_metadata() -> dict[str, object]:
    return {
        "mode": "serial",
        "launcher": "basic",
        "worker_count": 1,
        "torch_threads_per_worker": None,
        "deterministic": False,
        "resolved_accelerator": "cpu",
        "process_id": 1,
        "logical_cpu_count": 1,
    }


def _trace_arrays(*, bad_plan_indices: bool = False) -> dict[str, np.ndarray]:
    plan_indices = np.array([0, 0, 0, 0, 1] if bad_plan_indices else [0, 0, 0, 0, 0])
    return {
        "schema_version": np.asarray(ARTIFACT_SCHEMA_VERSION, dtype=np.int64),
        "trace_status": np.asarray("complete"),
        "warmup_initial_state": np.zeros(7),
        "warmup_initial_state_valid": np.asarray(True, dtype=np.bool_),
        "warmup_states": np.zeros((20, 7)),
        "warmup_rewards": np.zeros(20),
        "warmup_terminated": np.zeros(20, dtype=np.bool_),
        "warmup_truncated": np.zeros(20, dtype=np.bool_),
        "warmup_participant_counts": np.ones(20, dtype=np.int64),
        "warmup_static_object_counts": np.zeros(20, dtype=np.int64),
        "initial_state": np.zeros(7),
        "initial_state_valid": np.asarray(True, dtype=np.bool_),
        "planning_anchors": np.zeros((1, 7)),
        "initial_noise": np.zeros((1, 11, 80, 4), dtype=np.float32),
        "predictions_local": np.zeros((1, 11, 80, 4), dtype=np.float32),
        "observation_ego_current_state": np.zeros((1, 10), dtype=np.float32),
        "observation_neighbor_agents_past": np.zeros((1, 32, 21, 11), dtype=np.float32),
        "observation_static_objects": np.zeros((1, 5, 10), dtype=np.float32),
        "observation_lanes": np.zeros((1, 70, 20, 12), dtype=np.float32),
        "observation_lanes_speed_limit": np.zeros((1, 70, 1), dtype=np.float32),
        "observation_lanes_has_speed_limit": np.zeros((1, 70, 1), dtype=np.bool_),
        "observation_route_lanes": np.zeros((1, 25, 20, 12), dtype=np.float32),
        "observation_route_lanes_speed_limit": np.zeros((1, 25, 1), dtype=np.float32),
        "observation_route_lanes_has_speed_limit": np.zeros((1, 25, 1), dtype=np.bool_),
        "ego_predictions_world": np.zeros((1, 80, 4)),
        "executed_states": np.zeros((5, 7)),
        "executed_rewards": np.zeros(5),
        "executed_terminated": np.array([False, False, False, False, True]),
        "executed_truncated": np.zeros(5, dtype=np.bool_),
        "executed_plan_indices": plan_indices,
        "trajectory_target_centers": np.zeros((5, 2)),
        "trajectory_target_headings": np.zeros(5),
        "trajectory_position_errors_m": np.zeros(5),
        "trajectory_heading_errors_rad": np.zeros(5),
        "traffic_selected_ids": np.full((1, 32), "", dtype="<U64"),
        "traffic_participant_counts": np.ones(1, dtype=np.int64),
        "traffic_static_object_counts": np.zeros(1, dtype=np.int64),
        "traffic_nearest_distance_m": np.ones(1),
        "traffic_has_nearest": np.ones(1, dtype=np.bool_),
    }


def _episode(name: str, seed: int, density: float) -> dict[str, object]:
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "status": "completed",
        "trace_status": "complete",
        "scenario": {"name": name, "map_sequence": "S", "seed": seed},
        "evaluation_mode": "traffic",
        "noise_seed": seed,
        "traffic_density": density,
        "sampler": _sampler(),
        "guidance": {"name": "none"},
        "route_length_m": 2500.0,
        "plan_cycles": 1,
        "simulator_steps": 5,
        "simulated_seconds": 0.5,
        "environment_steps_including_warmup": 25,
        "distance_m": 2.0,
        "route_completion": 0.1,
        "total_reward": 1.0,
        "speed_mps": {"minimum": 4.0, "mean": 4.0, "maximum": 4.0},
        "arrive_dest": False,
        "out_of_road": False,
        "crash_vehicle": True,
        "crash_object": False,
        "crash_building": False,
        "crash_human": False,
        "terminal_reason": "crash_vehicle",
        "terminated": True,
        "truncated": False,
        "termination": {"type": "collision", "detail": "crash_vehicle"},
        "map_input_audit": _map_audit(),
        "history_warmup": {
            "simulator_steps": 20,
            "simulated_seconds": 2.0,
            "ego_displacement_m_maximum": 0.0,
            "participant_count_minimum": 1,
            "participant_count_maximum": 1,
        },
        "traffic_observation": {
            "planning_frames": 1,
            "frames_with_participants": 1,
            "frames_with_participants_fraction": 1.0,
            "participant_count_minimum": 1,
            "participant_count_maximum": 1,
            "nearest_participant_distance_m_minimum": 1.0,
        },
        "trajectory_execution_error": {
            "position_m": {"maximum": 0.0, "mean": 0.0, "final": 0.0},
            "heading_rad": {"maximum": 0.0, "mean": 0.0, "final": 0.0},
        },
    }


def _runtime(seed: int) -> dict[str, object]:
    return {
        "requested_accelerator": "cpu",
        "resolved_accelerator": "cpu",
        "requested_precision": "32-true",
        "resolved_precision": "32-true",
        "device": "cpu",
        "seed": seed,
        "world_size": 1,
    }


def _runtime_metadata(seed: int) -> dict[str, object]:
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "git_head": "fixture",
        "git_status_short": [],
        "platform": "fixture",
        "python": "fixture",
        "torch": "fixture",
        "lightning": "fixture",
        "metadrive": "fixture",
        "pydantic": "fixture",
        "inference_runtime": _runtime(seed),
        "sampler": _sampler(),
        "guidance": {"name": "none"},
        "execution": _execution_metadata(),
        "elapsed_seconds": 0.0,
        "cuda_memory": None,
    }


def _write_job(
    root: Path,
    job_id: int,
    seed: int,
    density: float,
    *,
    mismatch_episode_copy: bool = False,
    bad_plan_indices: bool = False,
    expected_seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
    expected_densities: tuple[float, ...] = (0.05, 0.10),
    video_enabled: bool = True,
) -> None:
    job = root / str(job_id)
    (job / ".hydra").mkdir(parents=True)
    resolved = f"""evaluation:
  history_warmup_steps: 20
  matrix:
    seeds: {list(expected_seeds)}
    traffic_densities: {list(expected_densities)}
scenarios:
  - name: long_straight
  - name: long_mixed
video:
  enabled: {str(video_enabled).lower()}
runtime:
  seed: 0
"""
    (job / "resolved_config.yaml").write_text(resolved, encoding="utf-8")
    (job / ".hydra" / "overrides.yaml").write_text("[]\n", encoding="utf-8")
    (job / "runtime_metadata.json").write_text(
        json.dumps(_runtime_metadata(seed)), encoding="utf-8"
    )
    (job / "tracked_diff.patch").write_bytes(b"")
    episodes = [
        _episode("long_straight", seed, density),
        _episode("long_mixed", seed, density),
    ]
    for index, episode in enumerate(episodes):
        episode_dir = job / str(episode["scenario"]["name"])
        episode_dir.mkdir()
        persisted = dict(episode)
        if mismatch_episode_copy and index == 0:
            persisted["distance_m"] = 3.0
        (episode_dir / "summary.json").write_text(json.dumps(persisted), encoding="utf-8")
        if video_enabled:
            (episode_dir / "closed_loop.gif").write_bytes(b"GIF89a")
        np.savez_compressed(
            episode_dir / "trace.npz", **_trace_arrays(bad_plan_indices=bad_plan_indices)
        )
    (job / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "status": "completed",
                "runtime": _runtime(seed),
                "checkpoint": {"ema_tensor_count": 276, "parameter_count": 6_042_628},
                "sampler": _sampler(),
                "guidance": {"name": "none"},
                "episodes": episodes,
            }
        ),
        encoding="utf-8",
    )


def _rewrite_trace(path: Path, case: str) -> None:
    with np.load(path, allow_pickle=False) as trace:
        arrays = {name: trace[name] for name in trace.files}
    if case == "missing":
        arrays.pop("predictions_local")
    elif case == "shape":
        arrays["executed_states"] = np.zeros((5, 6))
    elif case == "nonfinite":
        arrays["predictions_local"][0, 0, 0, 0] = np.nan
    elif case == "error_limit":
        arrays["trajectory_position_errors_m"][0] = 1e-3
    else:
        raise AssertionError(f"unknown trace rewrite case: {case}")
    np.savez_compressed(path, **arrays)


def test_partial_matrix_accepts_empty_tracked_diff_and_writes_once(tmp_path: Path) -> None:
    _write_job(tmp_path, 0, 0, 0.05)

    report = summarize_matrix(tmp_path, partial=True)

    assert report["matrix_complete"] is False
    assert report["validated_episode_count"] == 2
    assert (tmp_path / "partial_matrix_report.json").is_file()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        summarize_matrix(tmp_path, partial=True)


def test_complete_matrix_requires_and_reports_fixed_grid(tmp_path: Path) -> None:
    job_id = 0
    for seed in range(5):
        for density in (0.05, 0.10):
            _write_job(tmp_path, job_id, seed, density)
            job_id += 1

    report = build_matrix_report(tmp_path)

    assert report["matrix_complete"] is True
    assert report["validated_episode_count"] == 20
    assert len(report["statistics"]) == 4


def test_matrix_grid_and_video_rule_are_read_from_resolved_config(tmp_path: Path) -> None:
    for job_id, seed in enumerate((4, 7)):
        _write_job(
            tmp_path,
            job_id,
            seed,
            0.2,
            expected_seeds=(4, 7),
            expected_densities=(0.2,),
            video_enabled=False,
        )

    report = build_matrix_report(tmp_path)

    assert report["expected_job_grid"] == [
        {"seed": 4, "traffic_density": 0.2},
        {"seed": 7, "traffic_density": 0.2},
    ]
    assert report["validated_episode_count"] == 4


def test_matrix_includes_failed_episode_in_status_and_termination_counts(tmp_path: Path) -> None:
    job = tmp_path / "0"
    episode_dir = job / "long_straight"
    (job / ".hydra").mkdir(parents=True)
    episode_dir.mkdir()
    (job / "resolved_config.yaml").write_text(
        """evaluation:
  history_warmup_steps: 20
  matrix:
    seeds: [0]
    traffic_densities: [0.05]
scenarios:
  - name: long_straight
video:
  enabled: false
""",
        encoding="utf-8",
    )
    (job / ".hydra" / "overrides.yaml").write_text("[]\n", encoding="utf-8")
    runtime = _runtime(0)
    (job / "runtime_metadata.json").write_text(json.dumps(_runtime_metadata(0)), encoding="utf-8")
    (job / "tracked_diff.patch").write_bytes(b"")
    episode = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "status": "failed",
        "scenario": {"name": "long_straight", "map_sequence": "S", "seed": 0},
        "evaluation_mode": "traffic",
        "noise_seed": 0,
        "traffic_density": 0.05,
        "sampler": _sampler(),
        "guidance": {"name": "none"},
        "trace_status": "empty",
        "termination": {"type": "runtime_error", "detail": "reset"},
        "failure": {
            "stage": "reset",
            "exception_type": "RuntimeError",
            "message": "injected",
            "traceback": "RuntimeError: injected",
        },
    }
    (episode_dir / "summary.json").write_text(json.dumps(episode), encoding="utf-8")
    np.savez_compressed(episode_dir / "trace.npz", **EpisodeTraceRecorder.empty().finalize("empty"))
    (job / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "status": "failed",
                "runtime": runtime,
                "checkpoint": {"ema_tensor_count": 276, "parameter_count": 6_042_628},
                "sampler": _sampler(),
                "guidance": {"name": "none"},
                "episodes": [episode],
            }
        ),
        encoding="utf-8",
    )

    report = build_matrix_report(tmp_path)

    assert report["matrix_successful"] is False
    assert report["status_counts"] == {"completed": 0, "failed": 1}
    assert report["termination_type_counts"] == {"runtime_error": 1}
    assert report["statistics"] == {}


def test_partial_matrix_rejects_job_outside_expected_grid(tmp_path: Path) -> None:
    _write_job(tmp_path, 0, 7, 0.05)

    with pytest.raises(ValueError, match="unexpected matrix job"):
        build_matrix_report(tmp_path, partial=True)


def test_matrix_rejects_episode_copy_mismatch(tmp_path: Path) -> None:
    _write_job(tmp_path, 0, 0, 0.05, mismatch_episode_copy=True)

    with pytest.raises(ValueError, match="episode summary copy"):
        build_matrix_report(tmp_path, partial=True)


def test_matrix_rejects_invalid_trace_time_axis(tmp_path: Path) -> None:
    _write_job(tmp_path, 0, 0, 0.05, bad_plan_indices=True)

    with pytest.raises(ValueError, match="plan indices"):
        build_matrix_report(tmp_path, partial=True)


def test_matrix_rejects_duplicate_job_key(tmp_path: Path) -> None:
    _write_job(tmp_path, 0, 0, 0.05)
    _write_job(tmp_path, 1, 0, 0.05)

    with pytest.raises(ValueError, match="duplicate matrix job"):
        build_matrix_report(tmp_path, partial=True)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("missing", "missing arrays"),
        ("shape", "executed_states.*shape"),
        ("nonfinite", "non-finite"),
        ("error_limit", "position error limit"),
    ],
)
def test_matrix_rejects_malformed_trace(tmp_path: Path, case: str, message: str) -> None:
    _write_job(tmp_path, 0, 0, 0.05)
    _rewrite_trace(tmp_path / "0" / "long_straight" / "trace.npz", case)

    with pytest.raises(ValueError, match=message):
        build_matrix_report(tmp_path, partial=True)
