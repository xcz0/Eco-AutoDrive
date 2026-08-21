from __future__ import annotations

import pytest
import torch

from eco_planner.envs import collate_observations


def _observation(lane_value: float) -> dict[str, torch.Tensor]:
    return {
        "ego_current_state": torch.zeros(10, dtype=torch.float32),
        "neighbor_agents_past": torch.zeros((32, 21, 11), dtype=torch.float32),
        "static_objects": torch.zeros((5, 10), dtype=torch.float32),
        "lanes": torch.full((70, 20, 12), lane_value, dtype=torch.float32),
        "lanes_speed_limit": torch.zeros((70, 1), dtype=torch.float32),
        "lanes_has_speed_limit": torch.zeros((70, 1), dtype=torch.bool),
        "route_lanes": torch.zeros((25, 20, 12), dtype=torch.float32),
        "route_lanes_speed_limit": torch.zeros((25, 1), dtype=torch.float32),
        "route_lanes_has_speed_limit": torch.zeros((25, 1), dtype=torch.bool),
    }


def test_collate_observations_stacks_single_observations_without_changing_values() -> None:
    first = _observation(1.0)
    second = _observation(2.0)

    batch_one = collate_observations([first])
    batch_two = collate_observations([first, second])

    for name, value in first.items():
        torch.testing.assert_close(batch_one[name][0], value, rtol=0.0, atol=0.0)
    assert batch_two["lanes"].shape == (2, 70, 20, 12)
    assert batch_two["lanes"].dtype == torch.float32
    assert batch_two["lanes_has_speed_limit"].shape == (2, 70, 1)
    assert batch_two["lanes_has_speed_limit"].dtype == torch.bool
    torch.testing.assert_close(batch_two["lanes"][1], second["lanes"], rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        batch_two["lanes_has_speed_limit"][1],
        second["lanes_has_speed_limit"],
        rtol=0.0,
        atol=0.0,
    )


def test_collate_observations_rejects_empty_sequence() -> None:
    with pytest.raises(ValueError, match="empty"):
        collate_observations([])
