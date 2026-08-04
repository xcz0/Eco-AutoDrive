"""MetaDrive vector-map conversion for the official Diffusion Planner contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from eco_planner.envs.geometry import rear_axle_position, world_points_to_local
from eco_planner.envs.lane_speed import model_lane_speed_limit_mps
from eco_planner.models.config import OfficialDiffusionPlannerConfig

_LANE_FEATURE_DIM = 12
_TRAFFIC_LIGHT_UNKNOWN = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)


@dataclass(frozen=True)
class _LaneRecord:
    lane: Any
    distance: float
    stable_id: str
    road: tuple[Any, Any]


class MetaDriveMapAdapter:
    """Build raw, unnormalized Diffusion Planner map tensors from a PGMap."""

    def __init__(
        self,
        model_config: OfficialDiffusionPlannerConfig,
        query_radius_m: float,
    ) -> None:
        if not isinstance(model_config, OfficialDiffusionPlannerConfig):
            raise TypeError("model_config must be an OfficialDiffusionPlannerConfig")
        if type(query_radius_m) not in {int, float}:
            raise TypeError("query_radius_m must be numeric")
        if not np.isfinite(query_radius_m) or query_radius_m <= 0.0:
            raise ValueError("query_radius_m must be finite and positive")
        if model_config.lane_state_dim != _LANE_FEATURE_DIM:
            raise ValueError(f"lane_state_dim must be {_LANE_FEATURE_DIM}")
        if model_config.route_state_dim != _LANE_FEATURE_DIM:
            raise ValueError(f"route_state_dim must be {_LANE_FEATURE_DIM}")
        if model_config.route_len != model_config.lane_len:
            raise ValueError("route_len must equal lane_len")
        self._config = model_config
        self._query_radius_m = float(query_radius_m)

    def build(self, env: Any, device: torch.device) -> dict[str, torch.Tensor]:
        if not isinstance(device, torch.device):
            raise TypeError("device must be a torch.device")
        current_map = getattr(env, "current_map", None)
        if current_map is None:
            raise RuntimeError("MetaDrive environment has no current map; call reset() first")
        road_network = getattr(current_map, "road_network", None)
        if road_network is None or not hasattr(road_network, "get_all_lanes"):
            raise RuntimeError("current map does not expose a lane road network")
        ego = getattr(env, "agent", None)
        if ego is None:
            raise RuntimeError("MetaDrive environment has no active ego agent")
        navigation = getattr(ego, "navigation", None)
        if navigation is None:
            raise RuntimeError("ego agent has no navigation route")

        center_position = np.asarray(ego.position, dtype=np.float64)
        if center_position.shape != (2,) or not np.isfinite(center_position).all():
            raise ValueError("ego center position must be a finite two-dimensional vector")
        center_heading = float(ego.heading_theta)
        if not np.isfinite(center_heading):
            raise ValueError("ego heading must be finite")
        rear_wheelbase = getattr(ego, "REAR_WHEELBASE", None)
        if rear_wheelbase is None or not np.isfinite(rear_wheelbase) or rear_wheelbase <= 0.0:
            raise ValueError("ego vehicle must define a finite positive REAR_WHEELBASE")
        rear_axle = rear_axle_position(center_position, center_heading, float(rear_wheelbase))

        lane_records = self._select_lanes(road_network.get_all_lanes(), rear_axle)
        route_records = self._select_route_lanes(lane_records, navigation)
        arrays = self._allocate_arrays()
        for index, record in enumerate(lane_records):
            features, speed_limit, has_speed_limit = self._encode_lane(
                record.lane, rear_axle, center_heading
            )
            arrays["lanes"][index] = features
            arrays["lanes_speed_limit"][index, 0] = speed_limit
            arrays["lanes_has_speed_limit"][index, 0] = has_speed_limit
        for index, record in enumerate(route_records):
            features, speed_limit, has_speed_limit = self._encode_lane(
                record.lane, rear_axle, center_heading
            )
            arrays["route_lanes"][index] = features
            arrays["route_lanes_speed_limit"][index, 0] = speed_limit
            arrays["route_lanes_has_speed_limit"][index, 0] = has_speed_limit
        return {
            name: torch.as_tensor(value, device=device).unsqueeze(0)
            for name, value in arrays.items()
        }

    def _select_lanes(self, lanes: list[Any], rear_axle: np.ndarray) -> list[_LaneRecord]:
        records: list[_LaneRecord] = []
        for lane in lanes:
            lane_index = getattr(lane, "index", None)
            if not isinstance(lane_index, tuple) or len(lane_index) < 2:
                raise ValueError(f"lane has invalid index: {lane_index!r}")
            distance = float(lane.distance(rear_axle))
            if not np.isfinite(distance) or distance < 0.0:
                raise ValueError(f"lane {lane_index!r} returned invalid distance {distance!r}")
            if distance <= self._query_radius_m:
                records.append(
                    _LaneRecord(
                        lane=lane,
                        distance=distance,
                        stable_id=repr(lane_index),
                        road=(lane_index[0], lane_index[1]),
                    )
                )
        records.sort(key=lambda record: (record.distance, record.stable_id))
        selected = records[: self._config.lane_num]
        if not selected:
            raise RuntimeError("no lanes were found within the configured map query radius")
        return selected

    def _select_route_lanes(
        self, lane_records: list[_LaneRecord], navigation: Any
    ) -> list[_LaneRecord]:
        checkpoints = getattr(navigation, "checkpoints", None)
        if not isinstance(checkpoints, (list, tuple)) or len(checkpoints) < 2:
            raise RuntimeError("navigation checkpoints must contain at least one road")
        route_roads = list(zip(checkpoints[:-1], checkpoints[1:]))
        local_roads = {record.road for record in lane_records}
        connected_local_route: list[tuple[Any, Any]] = []
        route_started = False
        for road in route_roads:
            if road in local_roads:
                connected_local_route.append(road)
                route_started = True
            elif route_started:
                break
        if not connected_local_route:
            raise RuntimeError("no connected navigation route lanes exist in the local lane set")
        route_road_set = set(connected_local_route)
        selected = [record for record in lane_records if record.road in route_road_set]
        if not selected:
            raise RuntimeError("navigation route did not resolve to a selected local lane")
        return selected[: self._config.route_num]

    def _allocate_arrays(self) -> dict[str, np.ndarray]:
        config = self._config
        return {
            "lanes": np.zeros(
                (config.lane_num, config.lane_len, config.lane_state_dim), dtype=np.float32
            ),
            "lanes_speed_limit": np.zeros((config.lane_num, 1), dtype=np.float32),
            "lanes_has_speed_limit": np.zeros((config.lane_num, 1), dtype=np.bool_),
            "route_lanes": np.zeros(
                (config.route_num, config.route_len, config.route_state_dim), dtype=np.float32
            ),
            "route_lanes_speed_limit": np.zeros((config.route_num, 1), dtype=np.float32),
            "route_lanes_has_speed_limit": np.zeros((config.route_num, 1), dtype=np.bool_),
        }

    def _encode_lane(
        self, lane: Any, rear_axle: np.ndarray, ego_heading: float
    ) -> tuple[np.ndarray, float, bool]:
        length = float(lane.length)
        if not np.isfinite(length) or length <= 0.0:
            raise ValueError(f"lane {lane.index!r} must have a finite positive length")
        longitudinal = np.linspace(0.0, length, self._config.lane_len, dtype=np.float64)
        centers = []
        left_boundary = []
        right_boundary = []
        for value in longitudinal:
            width = float(lane.width_at(float(value)))
            if not np.isfinite(width) or width <= 0.0:
                raise ValueError(f"lane {lane.index!r} returned invalid width {width!r}")
            centers.append(lane.position(float(value), 0.0))
            # MetaDrive's positive lateral coordinate points right; the model expects left first.
            left_boundary.append(lane.position(float(value), -width / 2.0))
            right_boundary.append(lane.position(float(value), width / 2.0))
        centers_local = world_points_to_local(centers, rear_axle, ego_heading).astype(np.float32)
        left_local = world_points_to_local(left_boundary, rear_axle, ego_heading).astype(np.float32)
        right_local = world_points_to_local(right_boundary, rear_axle, ego_heading).astype(
            np.float32
        )

        features = np.zeros((self._config.lane_len, _LANE_FEATURE_DIM), dtype=np.float32)
        features[:, :2] = centers_local
        features[:-1, 2:4] = np.diff(centers_local, axis=0)
        features[:, 4:6] = left_local - centers_local
        features[:, 6:8] = right_local - centers_local
        features[:, 8:12] = _TRAFFIC_LIGHT_UNKNOWN
        speed_limit, has_speed_limit = model_lane_speed_limit_mps(lane)
        return features, speed_limit, has_speed_limit
