from __future__ import annotations

import json
import math
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import eco_planner.evaluation.artifacts.report as evaluation_report
from eco_planner.envs import TrajectoryExecutionRecord
from eco_planner.envs.domain import WRONG_DIRECTION_MAX_HEADING_ERROR_RAD
from eco_planner.evaluation import (
    load_trace_artifact,
)
from eco_planner.evaluation.artifacts import compute_episode_metrics
from eco_planner.evaluation.episodes import EpisodeTraceRecorder
from eco_planner.experiments.guidance import sweep as energy_study
from tests.evaluation.helpers import _episode


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


def _trace(
    heading_errors: list[float],
    *,
    speeds: list[float] | None = None,
    stopped: list[bool] | None = None,
    wrong_direction: list[bool] | None = None,
) -> dict[str, np.ndarray]:
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
    speed_values = np.full(count, 10.0) if speeds is None else np.asarray(speeds, dtype=np.float64)
    stopped_values = (
        np.zeros(count, dtype=np.bool_) if stopped is None else np.asarray(stopped, dtype=np.bool_)
    )
    wrong_direction_values = (
        np.asarray(heading_errors, dtype=np.float64) > WRONG_DIRECTION_MAX_HEADING_ERROR_RAD
        if wrong_direction is None
        else np.asarray(wrong_direction, dtype=np.bool_)
    )
    return {
        "executed_states": states,
        "initial_state": np.zeros(7),
        "executed_native_step_energy_ml": np.full(count, 0.1),
        "executed_native_episode_energy_ml": np.arange(1, count + 1, dtype=np.float64) * 0.1,
        "executed_fuel_proxy_step_energy_ml": np.full(count, 0.1),
        "executed_step_distance_m": np.ones(count),
        "executed_route_heading_errors_rad": np.asarray(heading_errors, dtype=np.float64),
        "executed_speed_mps": speed_values,
        "executed_stopped": stopped_values,
        "executed_wrong_direction": wrong_direction_values,
    }


@pytest.mark.parametrize(
    "heading_error,wrong_direction,fraction",
    [(math.pi / 2.0 + 0.1, True, 1.0 / 3.0), (math.pi / 2.0, False, 0.0)],
    ids=["beyond-half-pi", "half-pi-boundary"],
)
def test_wrong_direction_threshold(heading_error, wrong_direction, fraction) -> None:
    metrics = compute_episode_metrics(_trace([0.1, heading_error, 0.2]), _record())
    assert metrics.wrong_direction is wrong_direction
    assert metrics.wrong_direction_fraction == pytest.approx(fraction)


def test_stopped_fraction_uses_domain_stopped_facts() -> None:
    metrics = compute_episode_metrics(
        _trace([0.1, 0.2], speeds=[10.0, 0.05], stopped=[False, True]), _record()
    )

    assert metrics.stopped_fraction == pytest.approx(0.5)
    assert metrics.speed_mps.minimum == pytest.approx(0.05)
    assert metrics.speed_mps.maximum == pytest.approx(10.0)


@pytest.mark.parametrize(
    "flag", ["crash_vehicle", "crash_object", "crash_building", "crash_human", "crash_sidewalk"]
)
def test_episode_collision_consumes_terminal_domain_fact(flag: str) -> None:
    record = replace(_record(), **{flag: True})
    assert record.collision is True
    assert compute_episode_metrics(_trace([0.0]), record).collision is True


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


def test_evaluation_matrix_summary_schema_and_statistics_are_stable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    validated = evaluation_report.ValidatedMatrix(
        matrix_root=tmp_path,
        partial=False,
        observed_jobs={(0, 0.2), (1, 0.2)},
        expected_jobs={(0, 0.2), (1, 0.2)},
        scenario_count=1,
        episodes=(
            _episode(seed=0, distance_m=100.0, energy_ml=10.0),
            _episode(seed=1, distance_m=200.0, energy_ml=30.0),
        ),
    )
    monkeypatch.setattr(
        evaluation_report,
        "validate_matrix_artifacts",
        lambda *_args, **_kw: validated,
    )

    report = evaluation_report.build_matrix_report(tmp_path)

    assert set(report) == {
        "matrix_root",
        "matrix_complete",
        "matrix_successful",
        "observed_job_grid",
        "expected_job_grid",
        "expected_episode_count",
        "validated_episode_count",
        "status_counts",
        "termination_type_counts",
        "bootstrap_seed",
        "bootstrap_samples",
        "interface_limits",
        "episodes",
        "statistics",
    }
    assert report["matrix_complete"] is True
    assert report["status_counts"] == {"completed": 2, "failed": 0}
    assert report["episodes"] == [
        {
            "scenario": "traffic",
            "seed": 0,
            "traffic_density": 0.2,
            "terminal_reason": "arrive_dest",
            "status": "completed",
            "termination": {"type": "arrive_dest", "detail": "fixture"},
            "simulated_seconds": 1.0,
            "distance_m": 100.0,
            "energy_total_ml": 10.0,
            "energy_ml_per_km": 100.0,
            "route_completion": 0.4,
            "mean_speed_mps": 6.0,
            "wrong_direction": False,
        },
        {
            "scenario": "traffic",
            "seed": 1,
            "traffic_density": 0.2,
            "terminal_reason": "arrive_dest",
            "status": "completed",
            "termination": {"type": "arrive_dest", "detail": "fixture"},
            "simulated_seconds": 1.0,
            "distance_m": 200.0,
            "energy_total_ml": 30.0,
            "energy_ml_per_km": 150.0,
            "route_completion": 0.5,
            "mean_speed_mps": 7.0,
            "wrong_direction": False,
        },
    ]
    statistics = report["statistics"]
    assert statistics["traffic/density_0.20"]["metrics"]["distance_m"]["mean"] == 150.0
    assert statistics["traffic/density_0.20"]["metrics"]["energy_total_ml"]["median"] == 20.0
    assert statistics["traffic/density_0.20"]["arrive_rate"] == 0.5
    assert statistics["traffic/density_0.20"]["wrong_direction_rate"] == 0.0


def test_energy_study_run_record_schema_preserves_episode_and_traffic_context(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "traffic" / "baseline"
    run_dir.mkdir(parents=True)
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "episodes": [
                    {"scenario": {"name": "s0", "seed": 0}, "distance_m": 123.0},
                    {"scenario": {"name": "s1", "seed": 0}, "distance_m": 456.0},
                ],
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "resolved_config.yaml").write_text(
        "evaluation:\n  mode: traffic\nenv:\n  traffic_density: 0.125\n", encoding="utf-8"
    )
    job = energy_study.EvaluationJobSpec(id="traffic", config_name="fixture")
    guidance = energy_study.GuidanceProfileSpec(
        id="baseline", config="none", longitudinal_scale=None
    )

    def episode_payload(name: str, distance: float) -> dict[str, object]:
        return {"scenario": {"name": name, "seed": 0}, "distance_m": distance}

    episodes = tuple(
        SimpleNamespace(
            evaluation_mode="traffic",
            traffic_density=0.125,
            scenario=SimpleNamespace(
                model_dump=lambda name=name, **_kwargs: {"name": name, "seed": 0}
            ),
            model_dump=lambda name=name, distance=distance, **_kwargs: episode_payload(
                name, distance
            ),
        )
        for name, distance in (("s0", 123.0), ("s1", 456.0))
    )
    monkeypatch.setattr(
        energy_study,
        "load_job_summary",
        lambda _path: SimpleNamespace(status="completed", episodes=episodes),
    )

    record = energy_study._collect_run(job, guidance, run_dir, returncode=0)

    assert record == {
        "job": "traffic",
        "guidance": "baseline",
        "returncode": 0,
        "status": "completed",
        "output_dir": str(run_dir),
        "episodes": [
            {
                "scenario_metadata": {
                    "name": "s0",
                    "seed": 0,
                    "traffic_condition": "low_density_trigger_0.125",
                },
                "evaluation": {"scenario": {"name": "s0", "seed": 0}, "distance_m": 123.0},
            },
            {
                "scenario_metadata": {
                    "name": "s1",
                    "seed": 0,
                    "traffic_condition": "low_density_trigger_0.125",
                },
                "evaluation": {"scenario": {"name": "s1", "seed": 0}, "distance_m": 456.0},
            },
        ],
    }
