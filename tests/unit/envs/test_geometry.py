from __future__ import annotations

import numpy as np

from eco_planner.envs.geometry import (
    local_points_to_world,
    rear_axle_position,
    shortest_angle_delta,
    world_points_to_local,
)


def test_world_local_transform_round_trip_under_translation_and_rotation() -> None:
    anchor = np.array([12.0, -4.0])
    heading = 1.2
    local = np.array([[0.0, 0.0], [5.0, -2.0], [-3.0, 7.0]])

    world = local_points_to_world(local, anchor, heading)
    recovered = world_points_to_local(world, anchor, heading)

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
