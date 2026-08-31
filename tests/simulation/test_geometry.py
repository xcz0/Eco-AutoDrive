"""Research-level trajectory geometry contracts."""

from __future__ import annotations

import numpy as np
import pytest

from eco_planner.envs.contracts import PLANNER_HORIZON
from eco_planner.envs.domain.trajectory import to_world_trajectory
from eco_planner.envs.geometry import (
    local_points_to_world,
    rear_axle_position,
    shortest_angle_delta,
    world_points_to_local,
)


def _straight_trajectory(speed_mps: float = 5.0) -> np.ndarray:
    trajectory = np.zeros((PLANNER_HORIZON, 4), dtype=np.float32)
    trajectory[:, 0] = np.arange(1, PLANNER_HORIZON + 1, dtype=np.float32) * (speed_mps * 0.1)
    trajectory[:, 2] = 1.0
    return trajectory


@pytest.mark.smoke
def test_world_trajectory_uses_rear_axle_anchor_and_rotates_local_frame() -> None:
    result = to_world_trajectory(
        _straight_trajectory(),
        center_position=np.array([0.0, 1.4]),
        center_heading=np.pi / 2.0,
        rear_wheelbase=1.4,
        timestep_s=0.1,
    )

    np.testing.assert_allclose(result.centers[0], [0.0, 1.4], atol=1e-12)
    np.testing.assert_allclose(result.centers[1], [0.0, 1.9], atol=1e-6)
    np.testing.assert_allclose(result.velocities[0], [0.0, 5.0], atol=1e-6)
    assert result.headings[1] == pytest.approx(np.pi / 2.0)
    assert result.angular_velocities[0] == pytest.approx(0.0)


def test_world_local_transform_round_trip_under_translation_and_rotation() -> None:
    anchor = np.array([12.0, -4.0])
    heading = 1.2
    local = np.array([[0.0, 0.0], [5.0, -2.0], [-3.0, 7.0]])

    recovered = world_points_to_local(
        local_points_to_world(local, anchor, heading), anchor, heading
    )

    np.testing.assert_allclose(recovered, local, atol=1e-12)


def test_rear_axle_and_shortest_angle_contracts() -> None:
    np.testing.assert_allclose(
        rear_axle_position(np.array([2.0, 5.0]), np.pi / 2.0, 1.5),
        [2.0, 3.5],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        shortest_angle_delta(np.array([3.0 * np.pi, -3.0 * np.pi])),
        [-np.pi, -np.pi],
        atol=1e-12,
    )
