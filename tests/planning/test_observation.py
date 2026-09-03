"""Planner observation and traffic-history contract checks."""

from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

from eco_planner.contracts import TRAFFIC_HISTORY_FRAMES
from eco_planner.envs.domain.traffic import (
    StaticTrafficObjectState,
    TrafficFrame,
    TrafficParticipantState,
)
from eco_planner.envs.observation import PlannerObservationSpec
from eco_planner.envs.observation.history import TrafficHistory
from eco_planner.envs.observation.scene import TrafficSceneEncoder


def _frame(
    step: int,
    *,
    participants: tuple[TrafficParticipantState, ...] = (),
    static_objects: tuple[StaticTrafficObjectState, ...] = (),
) -> TrafficFrame:
    return TrafficFrame(
        simulator_step=step,
        ego_center_xy_m=(0.0, 0.0),
        ego_heading_rad=0.0,
        ego_rear_wheelbase_m=1.4,
        participants=participants,
        static_objects=static_objects,
    )


def _observation(lane_value: float) -> TensorDict:
    return TensorDict(
        {
            "ego_current_state": torch.zeros(10, dtype=torch.float32),
            "neighbor_agents_past": torch.zeros((32, 21, 11), dtype=torch.float32),
            "static_objects": torch.zeros((5, 10), dtype=torch.float32),
            "lanes": torch.full((70, 20, 12), lane_value, dtype=torch.float32),
            "lanes_speed_limit": torch.zeros((70, 1), dtype=torch.float32),
            "lanes_has_speed_limit": torch.zeros((70, 1), dtype=torch.bool),
            "route_lanes": torch.zeros((25, 20, 12), dtype=torch.float32),
            "route_lanes_speed_limit": torch.zeros((25, 1), dtype=torch.float32),
            "route_lanes_has_speed_limit": torch.zeros((25, 1), dtype=torch.bool),
        },
        batch_size=[],
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


def test_scene_encoder_uses_official_type_features_and_current_to_past_backfill() -> None:
    vehicle = TrafficParticipantState(
        object_id="vehicle",
        kind="vehicle",
        position_xy_m=(2.0, 0.0),
        heading_rad=0.0,
        velocity_xy_mps=(1.0, 0.0),
        width_m=2.0,
        length_m=4.0,
    )
    pedestrian = TrafficParticipantState(
        object_id="pedestrian",
        kind="pedestrian",
        position_xy_m=(3.0, 0.0),
        heading_rad=0.0,
        velocity_xy_mps=(0.0, 1.0),
        width_m=0.5,
        length_m=0.5,
    )
    barrier = StaticTrafficObjectState(
        object_id="barrier",
        kind="barrier",
        position_xy_m=(4.0, 0.0),
        heading_rad=0.0,
        width_m=1.0,
        length_m=2.0,
    )
    history = TrafficHistory()
    history.reset(_frame(0, participants=(pedestrian,)))
    history.append(
        tuple(
            _frame(
                step,
                participants=(pedestrian,)
                if step < TRAFFIC_HISTORY_FRAMES - 1
                else (vehicle, pedestrian),
                static_objects=(barrier,) if step == TRAFFIC_HISTORY_FRAMES - 1 else (),
            )
            for step in range(1, TRAFFIC_HISTORY_FRAMES)
        )
    )

    neighbors, static_objects, audit = TrafficSceneEncoder(10.0).build(history)

    assert audit.selected_participant_ids == ("participant-000001", "participant-000000")
    torch.testing.assert_close(torch.from_numpy(neighbors[0, :, 0]), torch.full((21,), 3.4))
    torch.testing.assert_close(
        torch.from_numpy(neighbors[0, :, 8:11]), torch.tensor([[1.0, 0.0, 0.0]] * 21)
    )
    torch.testing.assert_close(
        torch.from_numpy(neighbors[1, :, 8:11]), torch.tensor([[0.0, 1.0, 0.0]] * 21)
    )
    torch.testing.assert_close(
        torch.from_numpy(static_objects[0, 6:]), torch.tensor([0.0, 1.0, 0.0, 0.0])
    )
    assert not neighbors[2:].any()
    assert not static_objects[1:].any()


def test_tensordict_stack_preserves_values_and_batch_shape() -> None:
    first = _observation(1.0)
    second = _observation(2.0)

    batch_one = torch.stack([first])
    batch_two = torch.stack([first, second])

    for name, value in first.items():
        torch.testing.assert_close(batch_one[name][0], value, rtol=0.0, atol=0.0)
    assert batch_two["lanes"].shape == (2, 70, 20, 12)
    assert batch_two["lanes"].dtype == torch.float32
    assert batch_two["lanes_has_speed_limit"].shape == (2, 70, 1)
    assert batch_two["lanes_has_speed_limit"].dtype == torch.bool
    torch.testing.assert_close(batch_two["lanes"][1], second["lanes"], rtol=0.0, atol=0.0)
