from __future__ import annotations

import pytest

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
