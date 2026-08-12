from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from eco_planner.evaluation.comparison import compare_artifact_trees


def _write_tree(root: Path, *, changed: bool = False) -> None:
    job = root / "0"
    episode = job / "straight"
    episode.mkdir(parents=True)
    summary = {
        "schema_version": 2,
        "status": "completed",
        "runtime": {"seed": 0},
        "episodes": [
            {
                "schema_version": 2,
                "status": "completed",
                "scenario": {"name": "straight"},
                "traffic_density": 0.05,
            }
        ],
    }
    (job / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    np.savez_compressed(
        episode / "trace.npz",
        schema_version=np.asarray(2),
        trace_status=np.asarray("complete"),
        values=np.asarray([2 if changed else 1]),
    )


def test_artifact_comparison_matches_jobs_by_grid_and_arrays_exactly(tmp_path: Path) -> None:
    serial = tmp_path / "serial"
    parallel = tmp_path / "parallel"
    _write_tree(serial)
    _write_tree(parallel)

    report = compare_artifact_trees(serial, parallel)

    assert report == {"job_count": 1, "episode_count": 1, "array_count": 3, "equal": True}


def test_artifact_comparison_reports_array_path(tmp_path: Path) -> None:
    serial = tmp_path / "serial"
    parallel = tmp_path / "parallel"
    _write_tree(serial)
    _write_tree(parallel, changed=True)

    with pytest.raises(ValueError, match="straight/values"):
        compare_artifact_trees(serial, parallel)
