"""Wrong-direction episode metrics computed from the shared trace fields."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from eco_planner.envs import TrajectoryExecutionRecord
from eco_planner.evaluation.artifacts import compute_episode_metrics


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


def test_wrong_direction_flags_steps_beyond_half_pi() -> None:
    metrics = compute_episode_metrics(_trace([0.1, math.pi / 2.0 + 0.1, 0.2]), _record())

    assert metrics.wrong_direction is True
    assert metrics.wrong_direction_fraction == pytest.approx(1.0 / 3.0)


def test_half_pi_boundary_and_small_errors_are_not_wrong_direction() -> None:
    metrics = compute_episode_metrics(_trace([0.1, math.pi / 2.0, 0.2]), _record())

    assert metrics.wrong_direction is False
    assert metrics.wrong_direction_fraction == 0.0


def test_negative_route_heading_errors_are_rejected(tmp_path: Path) -> None:
    arrays = _trace([0.1, 0.2])
    arrays["executed_route_heading_errors_rad"] = np.asarray([-0.1, 0.2])

    with pytest.raises(ValueError, match="route heading errors"):
        compute_episode_metrics(arrays, _record())
