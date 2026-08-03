from __future__ import annotations

import numpy as np
import pytest

from eco_planner.envs.metadrive_env import (
    TRAJECTORY_HORIZON,
    _to_world_trajectory,
    _validate_trajectory,
)


def _straight_trajectory(speed_mps: float = 5.0) -> np.ndarray:
    trajectory = np.zeros((TRAJECTORY_HORIZON, 4), dtype=np.float32)
    trajectory[:, 0] = np.arange(1, TRAJECTORY_HORIZON + 1, dtype=np.float32) * (speed_mps * 0.1)
    trajectory[:, 2] = 1.0
    return trajectory


def test_world_trajectory_uses_rear_axle_anchor_and_rotates_local_frame() -> None:
    trajectory = _straight_trajectory()
    result = _to_world_trajectory(
        trajectory,
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


@pytest.mark.parametrize(
    ("trajectory", "error", "message"),
    [
        ([[0.0, 0.0, 1.0, 0.0]], TypeError, "numpy.ndarray"),
        (np.zeros((80, 4), dtype=np.float64), TypeError, "numpy.float32"),
        (np.zeros((79, 4), dtype=np.float32), ValueError, "shape"),
    ],
)
def test_trajectory_validation_rejects_invalid_contract(
    trajectory: object, error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        _validate_trajectory(trajectory, TRAJECTORY_HORIZON)


def test_trajectory_validation_rejects_nonfinite_values() -> None:
    trajectory = _straight_trajectory()
    trajectory[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        _validate_trajectory(trajectory, TRAJECTORY_HORIZON)


def test_trajectory_validation_rejects_zero_heading() -> None:
    trajectory = _straight_trajectory()
    trajectory[3, 2:4] = 0.0
    with pytest.raises(ValueError, match="non-zero"):
        _validate_trajectory(trajectory, TRAJECTORY_HORIZON)
