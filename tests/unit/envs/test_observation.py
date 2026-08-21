from __future__ import annotations

import pytest
import torch

from eco_planner.envs import collate_observations


def test_collate_observations_stacks_single_observations_without_changing_values() -> None:
    first = {
        "lanes": torch.tensor([[1.0, 0.0]], dtype=torch.float32),
        "lanes_has_speed_limit": torch.tensor([[True, False]]),
    }
    second = {
        "lanes": torch.tensor([[0.0, 2.0]], dtype=torch.float32),
        "lanes_has_speed_limit": torch.tensor([[False, True]]),
    }

    batch_one = collate_observations([first])
    batch_two = collate_observations([first, second])

    for name, value in first.items():
        torch.testing.assert_close(batch_one[name][0], value, rtol=0.0, atol=0.0)
    assert batch_two["lanes"].shape == (2, 1, 2)
    assert batch_two["lanes"].dtype == torch.float32
    assert batch_two["lanes_has_speed_limit"].shape == (2, 1, 2)
    assert batch_two["lanes_has_speed_limit"].dtype == torch.bool
    torch.testing.assert_close(batch_two["lanes"][1], second["lanes"], rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        batch_two["lanes_has_speed_limit"][1],
        second["lanes_has_speed_limit"],
        rtol=0.0,
        atol=0.0,
    )


def test_collate_observations_rejects_empty_or_mismatched_schemas() -> None:
    with pytest.raises(ValueError, match="empty"):
        collate_observations([])
    with pytest.raises(ValueError, match="schema"):
        collate_observations([{"lanes": torch.zeros(1)}, {"route_lanes": torch.zeros(1)}])
