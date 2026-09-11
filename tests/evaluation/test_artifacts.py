"""Episode metric, trace boundary and offline reader regressions."""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from eco_planner.envs import TrajectoryExecutionRecord
from eco_planner.evaluation import load_trace_artifact
from eco_planner.evaluation.artifacts import compute_episode_metrics
from eco_planner.evaluation.episodes import EpisodeTraceRecorder


@pytest.mark.parametrize(
    "initial_state,error,match",
    [
        pytest.param(np.zeros(8, dtype=np.float64), ValueError, "shape", id="shape"),
        pytest.param(np.zeros(7, dtype=np.float32), TypeError, "dtype", id="dtype"),
        pytest.param(
            np.array([np.nan, 0, 0, 0, 0, 0, 0]), ValueError, "non-finite", id="non-finite"
        ),
    ],
)
def test_trace_reader_rejects_invalid_initial_state(tmp_path: Path, initial_state, error, match):
    arrays = EpisodeTraceRecorder.empty().finalize("empty")
    arrays["initial_state"] = initial_state
    path = tmp_path / "trace.npz"
    np.savez(path, **arrays)  # pyright: ignore[reportArgumentType]
    with pytest.raises(error, match=f"initial_state.*{match}"):
        load_trace_artifact(path)


def _record() -> TrajectoryExecutionRecord:
    return TrajectoryExecutionRecord(
        start_center=np.zeros(2),
        start_heading=0.0,
        world_centers=np.zeros((80, 2)),
        world_headings=np.zeros(80),
        substep_states=np.zeros((0, 7)),
        target_centers=np.zeros((0, 2)),
        target_headings=np.zeros(0),
        substep_terminated=np.zeros(0, dtype=np.bool_),
        substep_truncated=np.zeros(0, dtype=np.bool_),
        traffic_frames=(),
        route_completion=0.5,
        arrive_dest=False,
        out_of_road=False,
        crash_vehicle=False,
        crash_object=False,
        crash_building=False,
        crash_human=False,
        max_step=False,
    )


def _trace(heading_errors: list[float]) -> dict[str, np.ndarray]:
    count = len(heading_errors)
    states = np.column_stack(
        (
            np.arange(1, count + 1, dtype=np.float64),
            np.zeros(count),
            np.zeros(count),
            np.full(count, 10.0),
            np.zeros(count),
            np.full(count, 10.0),
            np.zeros(count),
        )
    )
    return {
        "executed_states": states,
        "initial_state": np.zeros(7),
        "executed_native_step_energy_ml": np.full(count, 0.1),
        "executed_native_episode_energy_ml": np.arange(1, count + 1, dtype=np.float64) * 0.1,
        "executed_fuel_proxy_step_energy_ml": np.full(count, 0.1),
        "executed_step_distance_m": np.ones(count),
        "executed_route_heading_errors_rad": np.asarray(heading_errors, dtype=np.float64),
    }


def test_negative_route_heading_errors_are_rejected() -> None:
    arrays = _trace([0.1, 0.2])
    arrays["executed_route_heading_errors_rad"] = np.asarray([-0.1, 0.2])

    with pytest.raises(ValueError, match="route heading errors"):
        compute_episode_metrics(arrays, _record())


@pytest.mark.parametrize(
    "heading_error,wrong_direction,fraction",
    [(math.pi / 2.0 + 0.1, True, 1.0 / 3.0), (math.pi / 2.0, False, 0.0)],
    ids=["beyond-half-pi", "half-pi-boundary"],
)
def test_wrong_direction_threshold(heading_error, wrong_direction, fraction) -> None:
    metrics = compute_episode_metrics(_trace([0.1, heading_error, 0.2]), _record())
    assert metrics.wrong_direction is wrong_direction
    assert metrics.wrong_direction_fraction == pytest.approx(fraction)


@pytest.mark.parametrize("module", ["eco_planner.evaluation", "eco_planner.evaluation.artifacts"])
def test_offline_reader_import_does_not_load_online_dependencies(module: str) -> None:
    script = f"""
import sys
from {module} import build_matrix_report, load_job_summary, load_trace_artifact
assert callable(build_matrix_report)
assert callable(load_job_summary)
assert callable(load_trace_artifact)
assert "torch" not in sys.modules
assert "metadrive" not in sys.modules
assert "panda3d" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], check=True)
