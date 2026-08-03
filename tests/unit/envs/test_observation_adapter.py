from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from metadrive.component.static_object.traffic_object import TrafficCone

from eco_planner.envs.observation_adapter import NoTrafficMetaDriveObservationAdapter
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
        lambda env, device: _map_observation(official_model_config),
    )

    result = adapter.build(_environment(), torch.device("cpu"))

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
        adapter.build(env, torch.device("cpu"))


def test_no_traffic_adapter_rejects_runtime_static_object(
    official_model_config: OfficialDiffusionPlannerConfig,
) -> None:
    adapter = NoTrafficMetaDriveObservationAdapter(official_model_config, 100.0)
    cone = object.__new__(TrafficCone)
    env = _environment({"cone": cone})

    with pytest.raises(RuntimeError, match="static=.*cone"):
        adapter.build(env, torch.device("cpu"))
