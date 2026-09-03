"""Research-level trajectory geometry contracts."""

from __future__ import annotations

import numpy as np
import pytest

from eco_planner.contracts import (
    EVALUATION_EXECUTION_STEPS,
    METADRIVE_DECISION_REPEAT,
    METADRIVE_PHYSICS_STEP_S,
    PLANNER_HORIZON,
    ROLLOUT_EXECUTION_STEPS,
    SIMULATOR_STEP_S,
    TRAFFIC_HISTORY_FRAMES,
    TRAFFIC_HISTORY_WARMUP_STEPS,
    ExecutionMode,
    evaluation_plan_cycles,
    validate_metadrive_timestep,
)
from eco_planner.envs.domain import to_world_trajectory
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


def test_world_trajectory_preserves_rear_axle_velocity_and_wrapped_yaw_rate() -> None:
    center_position = np.array([2.0, 5.0])
    center_heading = 3.1
    rear_wheelbase = 1.5
    timestep_s = 0.1
    target_heading = -3.1
    relative_heading = target_heading - center_heading
    trajectory = np.zeros((PLANNER_HORIZON, 4), dtype=np.float32)
    trajectory[:, 0] = 1.0
    trajectory[:, 2] = np.cos(relative_heading)
    trajectory[:, 3] = np.sin(relative_heading)

    result = to_world_trajectory(
        trajectory,
        center_position=center_position,
        center_heading=center_heading,
        rear_wheelbase=rear_wheelbase,
        timestep_s=timestep_s,
    )

    rear_anchor = rear_axle_position(center_position, center_heading, rear_wheelbase)
    expected_rear_axle = local_points_to_world(
        np.array([[1.0, 0.0]]), rear_anchor, center_heading
    )[0]
    expected_center = expected_rear_axle + rear_wheelbase * np.array(
        [np.cos(target_heading), np.sin(target_heading)]
    )
    np.testing.assert_allclose(result.centers[1], expected_center, atol=1e-6)
    np.testing.assert_allclose(
        result.velocities[0], (expected_center - center_position) / timestep_s, atol=1e-6
    )
    assert result.angular_velocities[0] == pytest.approx(
        shortest_angle_delta(np.array([target_heading - center_heading]))[0] / timestep_s
    )


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


def test_execution_contracts_name_history_and_derive_every_timing_boundary() -> None:
    assert TRAFFIC_HISTORY_FRAMES == TRAFFIC_HISTORY_WARMUP_STEPS + 1
    assert ExecutionMode.ROLLOUT.steps == ROLLOUT_EXECUTION_STEPS
    assert ExecutionMode.EVALUATION.steps == EVALUATION_EXECUTION_STEPS
    assert validate_metadrive_timestep(
        METADRIVE_PHYSICS_STEP_S, METADRIVE_DECISION_REPEAT
    ) == pytest.approx(SIMULATOR_STEP_S)
    assert evaluation_plan_cycles(EVALUATION_EXECUTION_STEPS + 1) == 2

    with pytest.raises(ValueError, match="must equal"):
        validate_metadrive_timestep(METADRIVE_PHYSICS_STEP_S, 1)
