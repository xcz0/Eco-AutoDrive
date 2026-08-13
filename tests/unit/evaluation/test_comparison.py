from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from eco_planner.evaluation.comparison import compare_artifact_trees
from eco_planner.evaluation.schema import ARTIFACT_SCHEMA_VERSION
from eco_planner.evaluation.trace import EpisodeTraceRecorder


def _runtime() -> dict[str, object]:
    return {
        "requested_accelerator": "cpu",
        "resolved_accelerator": "cpu",
        "requested_precision": "32-true",
        "resolved_precision": "32-true",
        "device": "cpu",
        "seed": 0,
        "world_size": 1,
    }


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


def _write_tree(root: Path, *, changed: bool = False) -> None:
    job = root / "0"
    episode = job / "straight"
    episode.mkdir(parents=True)
    failed_episode = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "status": "failed",
        "scenario": {"name": "straight", "map_sequence": "S", "seed": 0},
        "evaluation_mode": "traffic",
        "traffic_density": 0.05,
        "noise_seed": 0,
        "sampler": _sampler(),
        "guidance": {"name": "none"},
        "trace_status": "empty",
        "termination": {"type": "runtime_error", "detail": "reset"},
        "failure": {
            "stage": "reset",
            "exception_type": "RuntimeError",
            "message": "fixture",
            "traceback": "RuntimeError: fixture",
        },
    }
    summary = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "status": "failed",
        "runtime": _runtime(),
        "checkpoint": {"ema_tensor_count": 276, "parameter_count": 6_042_628},
        "sampler": _sampler(),
        "guidance": {"name": "none"},
        "episodes": [failed_episode],
    }
    (job / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    arrays = EpisodeTraceRecorder.empty().finalize("empty")
    arrays["initial_state"] = np.full(7, 2 if changed else 1, dtype=np.float64)
    np.savez_compressed(episode / "trace.npz", **arrays)


def test_artifact_comparison_matches_jobs_by_grid_and_arrays_exactly(tmp_path: Path) -> None:
    serial = tmp_path / "serial"
    parallel = tmp_path / "parallel"
    _write_tree(serial)
    _write_tree(parallel)

    report = compare_artifact_trees(serial, parallel)

    assert report == {"job_count": 1, "episode_count": 1, "array_count": 39, "equal": True}


def test_artifact_comparison_reports_array_path(tmp_path: Path) -> None:
    serial = tmp_path / "serial"
    parallel = tmp_path / "parallel"
    _write_tree(serial)
    _write_tree(parallel, changed=True)

    with pytest.raises(ValueError, match="straight/initial_state"):
        compare_artifact_trees(serial, parallel)
