"""Planner observation and traffic-history contract checks."""

from __future__ import annotations

import pytest
import torch

from eco_planner.envs import collate_observations
from eco_planner.envs.contracts import TRAFFIC_HISTORY_FRAMES
from eco_planner.envs.domain.traffic import TrafficFrame
from eco_planner.envs.observation import PlannerObservationSpec
from eco_planner.envs.observation.history import TrafficHistory


def _frame(step: int) -> TrafficFrame:
    return TrafficFrame(
        simulator_step=step,
        ego_center_xy_m=(0.0, 0.0),
        ego_heading_rad=0.0,
        ego_rear_wheelbase_m=1.4,
        participants=(),
        static_objects=(),
    )


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


def test_planner_observation_dimensions_are_fixed_contracts() -> None:
    assert PlannerObservationSpec().time_len == TRAFFIC_HISTORY_FRAMES
    with pytest.raises(ValueError, match="fixed"):
        PlannerObservationSpec(20, 11, 32, 10, 5, 20, 12, 70, 20, 12, 25)


def test_traffic_history_commits_only_consecutive_domain_frames() -> None:
    history = TrafficHistory()
    history.reset(_frame(0))
    history.append((_frame(1),))

    assert history.latest.simulator_step == 1
    with pytest.raises(ValueError, match="consecutive"):
        history.append((_frame(3),))


def test_collate_observations_preserves_values_and_stacks_batches() -> None:
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


def test_collate_observations_rejects_empty_sequence() -> None:
    with pytest.raises(ValueError, match="empty"):
        collate_observations([])
