from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pytest
import torch

from eco_planner.envs.lane_speed import model_lane_speed_limit_mps
from eco_planner.envs.map_adapter import MetaDriveMapAdapter
from eco_planner.models.config import OfficialDiffusionPlannerConfig


class _StubLane:
    def __init__(
        self,
        lane_index: tuple[str, str, int],
        *,
        start: tuple[float, float],
        heading: float,
        length: float,
        width: float,
        speed_limit: float | None,
        distance: float,
    ) -> None:
        self.index = lane_index
        self.start = np.asarray(start, dtype=np.float64)
        self.heading = heading
        self.length = length
        self.width = width
        self.speed_limit = speed_limit
        self._distance = distance
        self.position_calls = 0
        self.width_calls = 0
        self.distance_calls = 0

    def position(self, longitudinal: float, lateral: float) -> np.ndarray:
        self.position_calls += 1
        forward = np.array([math.cos(self.heading), math.sin(self.heading)])
        right = np.array([math.sin(self.heading), -math.cos(self.heading)])
        return self.start + longitudinal * forward + lateral * right

    def width_at(self, longitudinal: float) -> float:
        self.width_calls += 1
        assert 0.0 <= longitudinal <= self.length
        return self.width

    def distance(self, position: np.ndarray) -> float:
        self.distance_calls += 1
        assert position.shape == (2,)
        return self._distance


class _StubRoadNetwork:
    def __init__(self, lanes: list[_StubLane]) -> None:
        self._lanes = lanes

    def get_all_lanes(self) -> list[_StubLane]:
        return self._lanes


@dataclass
class _StubMap:
    road_network: _StubRoadNetwork


@dataclass
class _StubNavigation:
    checkpoints: list[str]


class _StubAgent:
    REAR_WHEELBASE = 1.4

    def __init__(self, navigation: _StubNavigation | None = None) -> None:
        self.position = np.array([1.4, 0.0])
        self.heading_theta = 0.0
        self.navigation = navigation


class _StubEnv:
    def __init__(self, lanes: list[_StubLane], checkpoints: list[str]) -> None:
        self.current_map = _StubMap(_StubRoadNetwork(lanes))
        self.agent = _StubAgent(_StubNavigation(checkpoints))


def _lane(
    lane_number: int,
    *,
    road: tuple[str, str] = ("A", "B"),
    distance: float = 0.0,
    speed_limit: float | None = 36.0,
) -> _StubLane:
    return _StubLane(
        (road[0], road[1], lane_number),
        start=(0.0, lane_number * 4.0),
        heading=0.0,
        length=19.0,
        width=4.0,
        speed_limit=speed_limit,
        distance=distance,
    )


def test_map_adapter_builds_official_raw_tensor_contract(
    official_model_config: OfficialDiffusionPlannerConfig,
) -> None:
    env = _StubEnv([_lane(1, distance=1.0), _lane(0, distance=1.0)], ["A", "B"])
    adapter = MetaDriveMapAdapter(official_model_config, query_radius_m=100.0)

    result = adapter.build(env)

    assert result["lanes"].shape == (1, 70, 20, 12)
    assert result["lanes_speed_limit"].shape == (1, 70, 1)
    assert result["lanes_has_speed_limit"].shape == (1, 70, 1)
    assert result["route_lanes"].shape == (1, 25, 20, 12)
    assert result["route_lanes_speed_limit"].shape == (1, 25, 1)
    assert result["route_lanes_has_speed_limit"].shape == (1, 25, 1)
    assert result["lanes"].dtype == torch.float32
    assert result["lanes_has_speed_limit"].dtype == torch.bool
    assert result["lanes_speed_limit"][0, 0, 0].item() == pytest.approx(10.0)
    assert result["lanes_has_speed_limit"][0, 0, 0].item() is True

    first_lane = result["lanes"][0, 0]
    torch.testing.assert_close(first_lane[0, :2], torch.tensor([0.0, 0.0]))
    torch.testing.assert_close(first_lane[0, 4:6], torch.tensor([0.0, 2.0]))
    torch.testing.assert_close(first_lane[0, 6:8], torch.tensor([0.0, -2.0]))
    torch.testing.assert_close(first_lane[0, 8:12], torch.tensor([0.0, 0.0, 0.0, 1.0]))
    torch.testing.assert_close(first_lane[-1, 2:4], torch.zeros(2))
    assert torch.count_nonzero(result["lanes"][0, 2:]) == 0
    torch.testing.assert_close(result["route_lanes"][0, :2], result["lanes"][0, :2])


def test_map_adapter_caches_world_geometry_but_recomputes_exact_distance(
    official_model_config: OfficialDiffusionPlannerConfig,
) -> None:
    lane = _lane(0, distance=1.0)
    env = _StubEnv([lane], ["A", "B"])
    adapter = MetaDriveMapAdapter(official_model_config, query_radius_m=100.0)

    adapter.reset(env)
    first = adapter.build(env)
    geometry_calls = (lane.position_calls, lane.width_calls)
    second = adapter.build(env)

    assert geometry_calls == (60, 20)
    assert (lane.position_calls, lane.width_calls) == geometry_calls
    assert lane.distance_calls == 2
    for name in first:
        torch.testing.assert_close(first[name], second[name], rtol=0, atol=0)


def test_map_adapter_marks_missing_speed_limit_explicitly(
    official_model_config: OfficialDiffusionPlannerConfig,
) -> None:
    env = _StubEnv([_lane(0, speed_limit=None)], ["A", "B"])
    result = MetaDriveMapAdapter(official_model_config, 100.0).build(env)

    assert result["lanes_speed_limit"][0, 0, 0].item() == 0.0
    assert result["lanes_has_speed_limit"][0, 0, 0].item() is False


@pytest.mark.parametrize(
    ("speed_limit_kmh", "expected_mps", "has_speed_limit"),
    [(None, 0.0, False), (0.0, 0.0, False), (36.0, 10.0, True)],
)
def test_speed_limit_conversion_contract(
    speed_limit_kmh: float | None, expected_mps: float, has_speed_limit: bool
) -> None:
    value, valid = model_lane_speed_limit_mps(_lane(0, speed_limit=speed_limit_kmh))

    assert value == pytest.approx(expected_mps)
    assert valid is has_speed_limit


def test_speed_limit_conversion_accepts_finite_numpy_scalar() -> None:
    value, valid = model_lane_speed_limit_mps(_lane(0, speed_limit=np.float32(36.0)))

    assert value == pytest.approx(10.0)
    assert valid is True


def test_speed_limit_conversion_rejects_unconfigured_programmatic_sentinel() -> None:
    with pytest.raises(RuntimeError, match="programmatic lane speed limit was not configured"):
        model_lane_speed_limit_mps(_lane(0, speed_limit=1000.0))


def test_map_adapter_preserves_mixed_explicit_speed_limit_encoding(
    official_model_config: OfficialDiffusionPlannerConfig,
) -> None:
    env = _StubEnv(
        [_lane(0, speed_limit=50.0), _lane(1, speed_limit=20.0)],
        ["A", "B"],
    )

    result = MetaDriveMapAdapter(official_model_config, 100.0).build(env)

    np.testing.assert_allclose(
        result["lanes_speed_limit"][0, :2, 0].numpy(), [50.0 / 3.6, 20.0 / 3.6], atol=1e-6
    )
    assert result["lanes_has_speed_limit"][0, :2, 0].tolist() == [True, True]


def test_map_adapter_filters_radius_and_keeps_connected_route(
    official_model_config: OfficialDiffusionPlannerConfig,
) -> None:
    lanes = [
        _lane(0, road=("A", "B"), distance=2.0),
        _lane(1, road=("B", "C"), distance=3.0),
        _lane(2, road=("D", "E"), distance=4.0),
        _lane(3, road=("C", "D"), distance=101.0),
    ]
    env = _StubEnv(lanes, ["A", "B", "C", "D", "E"])
    result = MetaDriveMapAdapter(official_model_config, 100.0).build(env)

    assert torch.count_nonzero(result["lanes"][0, :, 0, 11]).item() == 3
    assert torch.count_nonzero(result["route_lanes"][0, :, 0, 11]).item() == 2


def test_map_adapter_truncates_lane_and_route_capacity(
    official_model_config: OfficialDiffusionPlannerConfig,
) -> None:
    lanes = [_lane(index, distance=float(index)) for index in range(75)]
    for index, lane in enumerate(lanes):
        lane.start[1] = float(index)
    env = _StubEnv(lanes, ["A", "B"])
    result = MetaDriveMapAdapter(official_model_config, 100.0).build(env)

    assert torch.count_nonzero(result["lanes"][0, :, 0, 11]).item() == 70
    assert torch.count_nonzero(result["route_lanes"][0, :, 0, 11]).item() == 25


def test_map_adapter_rejects_missing_navigation(
    official_model_config: OfficialDiffusionPlannerConfig,
) -> None:
    env = _StubEnv([_lane(0)], ["A", "B"])
    env.agent.navigation = None
    with pytest.raises(RuntimeError, match="navigation"):
        MetaDriveMapAdapter(official_model_config, 100.0).build(env)
