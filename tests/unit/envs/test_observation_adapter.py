from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from metadrive.component.static_object.traffic_object import TrafficCone

from eco_planner.envs.observation_adapter import (
    MetaDriveObservationAdapter,
    NoTrafficMetaDriveObservationAdapter,
)
from eco_planner.envs.traffic_state import (
    StaticTrafficObjectState,
    TrafficFrame,
    TrafficParticipantState,
)
from eco_planner.models.config import OfficialDiffusionPlannerConfig


def _map_observation(config: OfficialDiffusionPlannerConfig) -> dict[str, torch.Tensor]:
    return {
        "lanes": torch.zeros((1, config.lane_num, config.lane_len, 12)),
        "lanes_speed_limit": torch.zeros((1, config.lane_num, 1)),
        "lanes_has_speed_limit": torch.zeros((1, config.lane_num, 1), dtype=torch.bool),
        "route_lanes": torch.zeros((1, config.route_num, config.route_len, 12)),
        "route_lanes_speed_limit": torch.zeros((1, config.route_num, 1)),
        "route_lanes_has_speed_limit": torch.zeros((1, config.route_num, 1), dtype=torch.bool),
    }


class _Engine:
    def __init__(self, objects: dict[str, object]) -> None:
        self._objects = objects

    def get_objects(self) -> dict[str, object]:
        return self._objects


def _environment(objects: dict[str, object] | None = None) -> SimpleNamespace:
    ego = object()
    return SimpleNamespace(
        config={
            "traffic_density": 0.0,
            "random_traffic": False,
            "accident_prob": 0.0,
        },
        agent=ego,
        engine=_Engine({"ego": ego} if objects is None else objects),
    )


def test_no_traffic_adapter_builds_official_padding(
    official_model_config: OfficialDiffusionPlannerConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = NoTrafficMetaDriveObservationAdapter(official_model_config, 100.0)
    monkeypatch.setattr(
        adapter._map_adapter,
        "build",
        lambda env: _map_observation(official_model_config),
    )

    result = adapter.build(_environment())

    torch.testing.assert_close(
        result["ego_current_state"],
        torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
    )
    assert result["neighbor_agents_past"].shape == (1, 32, 21, 11)
    assert result["static_objects"].shape == (1, 5, 10)
    assert torch.count_nonzero(result["neighbor_agents_past"]).item() == 0
    assert torch.count_nonzero(result["static_objects"]).item() == 0


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("traffic_density", 0.1),
        ("random_traffic", True),
        ("accident_prob", 0.2),
    ],
)
def test_no_traffic_adapter_rejects_nonempty_configuration(
    official_model_config: OfficialDiffusionPlannerConfig,
    name: str,
    value: object,
) -> None:
    adapter = NoTrafficMetaDriveObservationAdapter(official_model_config, 100.0)
    env = _environment()
    env.config[name] = value

    with pytest.raises(ValueError, match=name):
        adapter.build(env)


def test_no_traffic_adapter_rejects_runtime_static_object(
    official_model_config: OfficialDiffusionPlannerConfig,
) -> None:
    adapter = NoTrafficMetaDriveObservationAdapter(official_model_config, 100.0)
    cone = object.__new__(TrafficCone)
    env = _environment({"cone": cone})

    with pytest.raises(RuntimeError, match="static=.*cone"):
        adapter.build(env)


def _traffic_frame(
    step: int,
    participants: tuple[TrafficParticipantState, ...],
    static_objects: tuple[StaticTrafficObjectState, ...] = (),
) -> TrafficFrame:
    return TrafficFrame(
        simulator_step=step,
        ego_center_xy_m=(10.0, 10.0),
        ego_heading_rad=np.pi / 2,
        ego_rear_wheelbase_m=1.0,
        participants=participants,
        static_objects=static_objects,
    )


def _participant(
    object_id: str,
    *,
    y: float,
    kind: str = "vehicle",
) -> TrafficParticipantState:
    return TrafficParticipantState(
        object_id=object_id,
        kind=kind,
        position_xy_m=(10.0, y),
        heading_rad=np.pi / 2,
        velocity_xy_mps=(0.0, 2.0),
        width_m=2.0,
        length_m=4.0,
    )


def test_traffic_adapter_builds_rotated_history_and_reverse_padding(
    official_model_config: OfficialDiffusionPlannerConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = MetaDriveObservationAdapter(official_model_config, 100.0)
    monkeypatch.setattr(
        adapter._map_adapter,
        "build",
        lambda env: _map_observation(official_model_config),
    )
    frames = []
    for step in range(21):
        participants = (_participant("vehicle-a", y=11.0 + 0.1 * step),) if step >= 10 else ()
        static = (
            StaticTrafficObjectState(
                object_id="cone",
                kind="traffic_cone",
                position_xy_m=(9.0, 11.0),
                heading_rad=np.pi / 2,
                width_m=0.2,
                length_m=0.2,
            ),
        )
        frames.append(_traffic_frame(step, participants, static))
    env = SimpleNamespace(engine=SimpleNamespace(episode_step=20))
    adapter.reset(frames[0])
    adapter.append_frames(tuple(frames[1:]))

    result = adapter.build(env)

    history = result["neighbor_agents_past"][0, 0]
    np.testing.assert_allclose(history[:11, 0].numpy(), 3.0, atol=1e-6)
    assert history[-1, 0].item() == pytest.approx(4.0)
    assert history[-1, 1].item() == pytest.approx(0.0, abs=1e-6)
    assert history[-1, 4].item() == pytest.approx(2.0)
    torch.testing.assert_close(history[-1, 8:], torch.tensor([1.0, 0.0, 0.0]))
    static = result["static_objects"][0, 0]
    assert static[0].item() == pytest.approx(2.0)
    assert static[1].item() == pytest.approx(1.0)
    torch.testing.assert_close(static[6:], torch.tensor([0.0, 0.0, 1.0, 0.0]))
    assert adapter.last_audit.selected_participant_ids == ("participant-000000",)


def test_traffic_adapter_sorts_and_truncates_current_participants(
    official_model_config: OfficialDiffusionPlannerConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = MetaDriveObservationAdapter(official_model_config, 100.0)
    monkeypatch.setattr(
        adapter._map_adapter,
        "build",
        lambda env: _map_observation(official_model_config),
    )
    participants = tuple(
        _participant(f"vehicle-{index:02d}", y=11.0 + index) for index in range(33)
    )
    frames = tuple(_traffic_frame(step, participants) for step in range(21))
    adapter.reset(frames[0])
    adapter.append_frames(frames[1:])

    adapter.build(SimpleNamespace(engine=SimpleNamespace(episode_step=20)))

    assert len(adapter.last_audit.selected_participant_ids) == 32
    assert adapter.last_audit.selected_participant_ids[0] == "participant-000000"
    assert adapter.last_audit.selected_participant_ids[-1] == "participant-000031"
    assert adapter.last_audit.participant_count_in_radius == 33


def test_traffic_adapter_artifact_ids_do_not_depend_on_metadrive_uuid(
    official_model_config: OfficialDiffusionPlannerConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audits = []
    observations = []
    for object_id in ("random-uuid-a", "random-uuid-b"):
        adapter = MetaDriveObservationAdapter(official_model_config, 100.0)
        monkeypatch.setattr(
            adapter._map_adapter,
            "build",
            lambda env: _map_observation(official_model_config),
        )
        frames = tuple(
            _traffic_frame(step, (_participant(object_id, y=11.0 + 0.1 * step),))
            for step in range(21)
        )
        adapter.reset(frames[0])
        adapter.append_frames(frames[1:])
        observations.append(adapter.build(SimpleNamespace(engine=SimpleNamespace(episode_step=20))))
        audits.append(adapter.last_audit)

    torch.testing.assert_close(
        observations[0]["neighbor_agents_past"],
        observations[1]["neighbor_agents_past"],
        rtol=0,
        atol=0,
    )
    assert (
        audits[0].selected_participant_ids
        == audits[1].selected_participant_ids
        == ("participant-000000",)
    )


def test_traffic_adapter_requires_consecutive_complete_history(
    official_model_config: OfficialDiffusionPlannerConfig,
) -> None:
    adapter = MetaDriveObservationAdapter(official_model_config, 100.0)
    adapter.reset(_traffic_frame(0, ()))
    with pytest.raises(ValueError, match="consecutive"):
        adapter.append_frames((_traffic_frame(2, ()),))
    with pytest.raises(RuntimeError, match="exactly 21"):
        adapter.build(SimpleNamespace(engine=SimpleNamespace(episode_step=0)))


def test_traffic_adapter_rejects_nonconsecutive_batch_atomically(
    official_model_config: OfficialDiffusionPlannerConfig,
) -> None:
    adapter = MetaDriveObservationAdapter(official_model_config, 100.0)
    adapter.reset(_traffic_frame(0, ()))
    frames = (
        _traffic_frame(1, (_participant("new-id", y=12.0),)),
        _traffic_frame(3, (_participant("new-id", y=12.0),)),
    )

    with pytest.raises(ValueError, match="consecutive"):
        adapter.append_frames(frames)

    assert tuple(frame.simulator_step for frame in adapter._history) == (0,)
    assert adapter._artifact_participant_ids == {}
