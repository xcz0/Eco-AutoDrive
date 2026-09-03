"""MetaDrive vector-map conversion for the official Diffusion Planner contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from shapely import LineString, box
from shapely.strtree import STRtree

from ..array_types import (
    EncodedLaneArray,
    LaneGeometryArray,
    NumpyMapObservation,
    WorldVectorArray,
)
from ..geometry import rear_axle_position, world_points_to_local
from ..observation import PlannerObservationSpec
from .lane_speed import model_lane_speed_limit_mps

_LANE_FEATURE_DIM = 12
_TRAFFIC_LIGHT_UNKNOWN = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)


@dataclass(frozen=True, slots=True)
class _LaneSnapshot:
    lane: Any
    stable_id: str
    road: tuple[Any, Any]
    centers_world: LaneGeometryArray
    left_boundary_world: LaneGeometryArray
    right_boundary_world: LaneGeometryArray
    speed_limit_mps: float
    has_speed_limit: bool


def navigation_route_roads(navigation: Any) -> tuple[tuple[Any, Any], ...]:
    """Return the complete ordered navigation route as MetaDrive road edges."""

    checkpoints = getattr(navigation, "checkpoints", None)
    if not isinstance(checkpoints, (list, tuple)) or len(checkpoints) < 2:
        raise RuntimeError("navigation checkpoints must contain at least one road")
    return tuple(zip(checkpoints[:-1], checkpoints[1:], strict=True))


class MetaDriveMapAdapter:
    """Build raw, unnormalized Diffusion Planner map arrays from a PGMap."""

    def __init__(
        self,
        observation_spec: PlannerObservationSpec,
        query_radius_m: float,
    ) -> None:
        if not np.isfinite(query_radius_m) or query_radius_m <= 0.0:
            raise ValueError("query_radius_m must be finite and positive")
        if observation_spec.lane_state_dim != _LANE_FEATURE_DIM:
            raise ValueError(f"lane_state_dim must be {_LANE_FEATURE_DIM}")
        if observation_spec.route_state_dim != _LANE_FEATURE_DIM:
            raise ValueError(f"route_state_dim must be {_LANE_FEATURE_DIM}")
        if observation_spec.route_len != observation_spec.lane_len:
            raise ValueError("route_len must equal lane_len")
        self._config = observation_spec
        self._query_radius_m = float(query_radius_m)
        self._map_identity: int | None = None
        self._lane_snapshots: tuple[_LaneSnapshot, ...] | None = None
        self._lane_tree: STRtree | None = None
        self._maximum_sample_interval_m: float | None = None

    def reset(self, env: Any) -> None:
        """Capture immutable lane geometry and metadata for the current map."""

        current_map, road_network, _, _ = self._environment_parts(env)
        snapshots = tuple(self._snapshot_lane(lane) for lane in road_network.get_all_lanes())
        if not snapshots:
            raise RuntimeError("current map does not contain any lanes")
        self._map_identity = id(current_map)
        self._lane_snapshots = snapshots
        self._lane_tree = STRtree([LineString(snapshot.centers_world) for snapshot in snapshots])
        self._maximum_sample_interval_m = max(
            float(np.linalg.norm(np.diff(snapshot.centers_world, axis=0), axis=1).max())
            for snapshot in snapshots
        )

    def build_arrays(self, env: Any, *, allow_empty_route: bool = False) -> NumpyMapObservation:
        """Build raw NumPy map fields before the common observation output boundary."""

        current_map, _, ego, navigation = self._environment_parts(env)
        if self._lane_snapshots is None:
            self.reset(env)
        if self._map_identity != id(current_map) or self._lane_snapshots is None:
            raise RuntimeError("map snapshot does not match the current reset map")

        center_position = np.asarray(ego.position, dtype=np.float64)
        center_heading = float(ego.heading_theta)
        rear_axle = rear_axle_position(center_position, center_heading, float(ego.REAR_WHEELBASE))

        lane_snapshots = self._select_lanes(self._candidate_snapshots(rear_axle), rear_axle)
        route_snapshots = self._select_route_lanes(
            lane_snapshots, navigation, allow_empty_route=allow_empty_route
        )
        arrays = self._allocate_arrays()
        encoded = self._encode_lanes(lane_snapshots, rear_axle, center_heading)
        encoded_by_id: dict[str, tuple[EncodedLaneArray, float, bool]] = {}
        for index, (snapshot, values) in enumerate(zip(lane_snapshots, encoded, strict=True)):
            features, speed_limit, has_speed_limit = values
            arrays["lanes"][index] = features
            arrays["lanes_speed_limit"][index, 0] = speed_limit
            arrays["lanes_has_speed_limit"][index, 0] = has_speed_limit
            encoded_by_id[snapshot.stable_id] = values
        for index, snapshot in enumerate(route_snapshots):
            features, speed_limit, has_speed_limit = encoded_by_id[snapshot.stable_id]
            arrays["route_lanes"][index] = features
            arrays["route_lanes_speed_limit"][index, 0] = speed_limit
            arrays["route_lanes_has_speed_limit"][index, 0] = has_speed_limit
        return arrays

    @staticmethod
    def _environment_parts(env: Any) -> tuple[Any, Any, Any, Any]:
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
        return current_map, road_network, ego, navigation

    def _snapshot_lane(self, lane: Any) -> _LaneSnapshot:
        lane_index = getattr(lane, "index", None)
        if not isinstance(lane_index, tuple) or len(lane_index) < 2:
            raise ValueError(f"lane has invalid index: {lane_index!r}")
        length = float(lane.length)
        if not np.isfinite(length) or length <= 0.0:
            raise ValueError(f"lane {lane_index!r} must have a finite positive length")
        longitudinal = np.linspace(0.0, length, self._config.lane_len, dtype=np.float64)
        centers = np.empty((self._config.lane_len, 2), dtype=np.float64)
        left = np.empty_like(centers)
        right = np.empty_like(centers)
        for index, value in enumerate(longitudinal):
            width = float(lane.width_at(float(value)))
            if not np.isfinite(width) or width <= 0.0:
                raise ValueError(f"lane {lane_index!r} returned invalid width {width!r}")
            centers[index] = lane.position(float(value), 0.0)
            left[index] = lane.position(float(value), -width / 2.0)
            right[index] = lane.position(float(value), width / 2.0)
        speed_limit, has_speed_limit = model_lane_speed_limit_mps(lane)
        for values in (centers, left, right):
            values.setflags(write=False)
        return _LaneSnapshot(
            lane=lane,
            stable_id=repr(lane_index),
            road=(lane_index[0], lane_index[1]),
            centers_world=centers,
            left_boundary_world=left,
            right_boundary_world=right,
            speed_limit_mps=speed_limit,
            has_speed_limit=has_speed_limit,
        )

    def _select_lanes(
        self, snapshots: tuple[_LaneSnapshot, ...], rear_axle: WorldVectorArray
    ) -> list[_LaneSnapshot]:
        candidates: list[tuple[float, _LaneSnapshot]] = []
        for snapshot in snapshots:
            distance = float(snapshot.lane.distance(rear_axle))
            if not np.isfinite(distance) or distance < 0.0:
                raise ValueError(
                    f"lane {snapshot.stable_id} returned invalid distance {distance!r}"
                )
            if distance <= self._query_radius_m:
                candidates.append((distance, snapshot))
        candidates.sort(key=lambda candidate: (candidate[0], candidate[1].stable_id))
        selected = [snapshot for _, snapshot in candidates[: self._config.lane_num]]
        if not selected:
            raise RuntimeError("no lanes were found within the configured map query radius")
        return selected

    def _candidate_snapshots(self, rear_axle: WorldVectorArray) -> tuple[_LaneSnapshot, ...]:
        if (
            self._lane_snapshots is None
            or self._lane_tree is None
            or self._maximum_sample_interval_m is None
        ):
            raise RuntimeError("map spatial index is unavailable before reset")
        radius = self._query_radius_m + self._maximum_sample_interval_m
        bounds = box(
            rear_axle[0] - radius,
            rear_axle[1] - radius,
            rear_axle[0] + radius,
            rear_axle[1] + radius,
        )
        indexes = np.asarray(self._lane_tree.query(bounds), dtype=np.intp)
        return tuple(self._lane_snapshots[index] for index in indexes)

    def _select_route_lanes(
        self,
        lane_snapshots: list[_LaneSnapshot],
        navigation: Any,
        *,
        allow_empty_route: bool,
    ) -> list[_LaneSnapshot]:
        route_roads = navigation_route_roads(navigation)
        local_roads = {snapshot.road for snapshot in lane_snapshots}
        connected_local_route: list[tuple[Any, Any]] = []
        route_started = False
        for road in route_roads:
            if road in local_roads:
                connected_local_route.append(road)
                route_started = True
            elif route_started:
                break
        if not connected_local_route:
            if allow_empty_route:
                return []
            raise RuntimeError("no connected navigation route lanes exist in the local lane set")
        route_road_set = set(connected_local_route)
        selected = [snapshot for snapshot in lane_snapshots if snapshot.road in route_road_set]
        if not selected:
            raise RuntimeError("navigation route did not resolve to a selected local lane")
        return selected[: self._config.route_num]

    def _allocate_arrays(self) -> NumpyMapObservation:
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

    def _encode_lanes(
        self,
        snapshots: list[_LaneSnapshot],
        rear_axle: WorldVectorArray,
        ego_heading: float,
    ) -> list[tuple[EncodedLaneArray, float, bool]]:
        centers_world = np.stack([snapshot.centers_world for snapshot in snapshots])
        left_world = np.stack([snapshot.left_boundary_world for snapshot in snapshots])
        right_world = np.stack([snapshot.right_boundary_world for snapshot in snapshots])
        shape = centers_world.shape
        centers_local = (
            world_points_to_local(centers_world.reshape(-1, 2), rear_axle, ego_heading)
            .reshape(shape)
            .astype(np.float32)
        )
        left_local = (
            world_points_to_local(left_world.reshape(-1, 2), rear_axle, ego_heading)
            .reshape(shape)
            .astype(np.float32)
        )
        right_local = (
            world_points_to_local(right_world.reshape(-1, 2), rear_axle, ego_heading)
            .reshape(shape)
            .astype(np.float32)
        )
        results: list[tuple[EncodedLaneArray, float, bool]] = []
        for index, snapshot in enumerate(snapshots):
            features = np.zeros((self._config.lane_len, _LANE_FEATURE_DIM), dtype=np.float32)
            features[:, :2] = centers_local[index]
            features[:-1, 2:4] = np.diff(centers_local[index], axis=0)
            features[:, 4:6] = left_local[index] - centers_local[index]
            features[:, 6:8] = right_local[index] - centers_local[index]
            features[:, 8:12] = _TRAFFIC_LIGHT_UNKNOWN
            results.append((features, snapshot.speed_limit_mps, snapshot.has_speed_limit))
        return results
