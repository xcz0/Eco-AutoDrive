"""MetaDrive vector-map conversion for the official Diffusion Planner contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from shapely import LineString, box
from shapely.strtree import STRtree

from eco_planner.envs.geometry import rear_axle_position, world_points_to_local
from eco_planner.envs.lane_speed import (
    MAX_LANE_SPEED_LIMIT_KMH,
    PROGRAMMATIC_SPEED_LIMIT_SENTINEL_KMH,
    model_lane_speed_limit_mps,
    validated_programmatic_speed_limit_kmh,
)
from eco_planner.envs.observation import to_cpu_torch_observation
from eco_planner.envs.validation import is_real_scalar
from eco_planner.models.config import OfficialDiffusionPlannerConfig

_LANE_FEATURE_DIM = 12
_TRAFFIC_LIGHT_UNKNOWN = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)


@dataclass(slots=True)
class ProgrammaticLaneSpeedAdapter:
    """Replace only PGMap's unset-speed sentinel after a map has been created."""

    configured_speed_limit_kmh: float
    _sentinel_lane_ids: frozenset[str] | None = field(default=None, init=False)
    _map: object | None = field(default=None, init=False)
    _audit: dict[str, object] | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.configured_speed_limit_kmh = validated_programmatic_speed_limit_kmh(
            self.configured_speed_limit_kmh
        )

    @property
    def audit(self) -> dict[str, object]:
        if self._audit is None:
            raise RuntimeError("programmatic lane speed limits are unavailable before reset")
        return dict(self._audit)

    def clear_audit(self) -> None:
        """Invalidate the previous reset's audit before beginning a new lifecycle."""

        self._audit = None

    def apply(self, current_map: object) -> None:
        """Configure the current map and retain its verified speed-limit audit."""

        road_network = getattr(current_map, "road_network", None)
        if road_network is None or not hasattr(road_network, "get_all_lanes"):
            raise RuntimeError("current map does not expose a lane road network")
        lanes = road_network.get_all_lanes()
        if not isinstance(lanes, list) or not lanes:
            raise RuntimeError("current map road network has no lanes")

        lane_ids = [repr(getattr(lane, "index", None)) for lane in lanes]
        if len(set(lane_ids)) != len(lane_ids):
            raise RuntimeError("current map exposes duplicate lane identifiers")
        if current_map is not self._map:
            self._sentinel_lane_ids = frozenset(
                lane_id
                for lane_id, lane in zip(lane_ids, lanes, strict=True)
                if getattr(lane, "speed_limit", None) == PROGRAMMATIC_SPEED_LIMIT_SENTINEL_KMH
            )
            self._map = current_map
        if self._sentinel_lane_ids is None:
            raise RuntimeError("programmatic sentinel state was not initialized")

        replaced_lane_ids: set[str] = set()
        preserved_speed_limits_kmh: dict[str, float] = {}
        for lane_id, lane in zip(lane_ids, lanes, strict=True):
            speed_limit = getattr(lane, "speed_limit", None)
            if not is_real_scalar(speed_limit) or not np.isfinite(speed_limit):
                raise ValueError(f"lane {lane_id} speed limit must be a finite numeric km/h value")
            original_kmh = float(speed_limit)
            if lane_id in self._sentinel_lane_ids:
                if original_kmh not in {
                    self.configured_speed_limit_kmh,
                    PROGRAMMATIC_SPEED_LIMIT_SENTINEL_KMH,
                }:
                    raise RuntimeError(
                        f"lane {lane_id} no longer has the configured programmatic speed limit"
                    )
                setter = getattr(lane, "set_speed_limit", None)
                if not callable(setter):
                    raise RuntimeError(f"lane {lane_id} cannot set its programmatic speed limit")
                setter(self.configured_speed_limit_kmh)
                replaced_lane_ids.add(lane_id)
            elif 0.0 < original_kmh <= MAX_LANE_SPEED_LIMIT_KMH:
                preserved_speed_limits_kmh[lane_id] = original_kmh
            else:
                raise ValueError(
                    f"lane {lane_id} speed limit {original_kmh} km/h is neither the "
                    "programmatic unset sentinel nor a legal explicit speed limit"
                )

        final_speed_limits_kmh: list[float] = []
        for lane_id, lane in zip(lane_ids, lanes, strict=True):
            speed_limit = getattr(lane, "speed_limit", None)
            if not is_real_scalar(speed_limit) or not np.isfinite(speed_limit):
                raise RuntimeError(f"lane {lane_id} returned an invalid configured speed limit")
            final_kmh = float(speed_limit)
            if final_kmh == PROGRAMMATIC_SPEED_LIMIT_SENTINEL_KMH:
                raise RuntimeError(f"lane {lane_id} retained the programmatic speed-limit sentinel")
            if lane_id in replaced_lane_ids and final_kmh != self.configured_speed_limit_kmh:
                raise RuntimeError(f"lane {lane_id} did not retain the configured speed limit")
            if (
                lane_id in preserved_speed_limits_kmh
                and final_kmh != preserved_speed_limits_kmh[lane_id]
            ):
                raise RuntimeError(f"lane {lane_id} explicit speed limit was unexpectedly modified")
            final_speed_limits_kmh.append(final_kmh)

        counts: dict[str, int] = {}
        for speed_limit_kmh in final_speed_limits_kmh:
            label = f"{speed_limit_kmh:g}"
            counts[label] = counts.get(label, 0) + 1
        self._audit = {
            "speed_limit_sentinel_replaced_count": len(replaced_lane_ids),
            "speed_limit_existing_preserved_count": len(preserved_speed_limits_kmh),
            "configured_programmatic_lane_speed_limit_kmh": self.configured_speed_limit_kmh,
            "lane_speed_limit_kmh_counts": dict(
                sorted(counts.items(), key=lambda item: float(item[0]))
            ),
        }


@dataclass(frozen=True, slots=True)
class _LaneSnapshot:
    lane: Any
    stable_id: str
    road: tuple[Any, Any]
    centers_world: np.ndarray
    left_boundary_world: np.ndarray
    right_boundary_world: np.ndarray
    speed_limit_mps: float
    has_speed_limit: bool


@dataclass(frozen=True, slots=True)
class _LaneRecord:
    snapshot: _LaneSnapshot
    distance: float


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
        self._map_identity: int | None = None
        self._lane_snapshots: tuple[_LaneSnapshot, ...] | None = None
        self._lane_centers_world: np.ndarray | None = None
        self._lane_left_world: np.ndarray | None = None
        self._lane_right_world: np.ndarray | None = None
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
        self._lane_centers_world = np.stack([snapshot.centers_world for snapshot in snapshots])
        self._lane_left_world = np.stack([snapshot.left_boundary_world for snapshot in snapshots])
        self._lane_right_world = np.stack([snapshot.right_boundary_world for snapshot in snapshots])
        self._lane_tree = STRtree([LineString(snapshot.centers_world) for snapshot in snapshots])
        self._maximum_sample_interval_m = max(
            float(np.linalg.norm(np.diff(snapshot.centers_world, axis=0), axis=1).max())
            for snapshot in snapshots
        )

    def build(self, env: Any) -> dict[str, torch.Tensor]:
        """Build one CPU-tensor map observation for direct adapter use."""

        return to_cpu_torch_observation(self.build_arrays(env))

    def build_arrays(self, env: Any) -> dict[str, np.ndarray]:
        """Build raw NumPy map fields before the common observation output boundary."""

        current_map, _, ego, navigation = self._environment_parts(env)
        if self._lane_snapshots is None:
            self.reset(env)
        if self._map_identity != id(current_map) or self._lane_snapshots is None:
            raise RuntimeError("map snapshot does not match the current reset map")

        center_position = np.asarray(ego.position, dtype=np.float64)
        center_heading = float(ego.heading_theta)
        rear_axle = rear_axle_position(center_position, center_heading, float(ego.REAR_WHEELBASE))

        lane_records = self._select_lanes(self._candidate_snapshots(rear_axle), rear_axle)
        route_records = self._select_route_lanes(lane_records, navigation)
        arrays = self._allocate_arrays()
        encoded = self._encode_lanes(lane_records, rear_axle, center_heading)
        encoded_by_id: dict[str, tuple[np.ndarray, float, bool]] = {}
        for index, (record, values) in enumerate(zip(lane_records, encoded, strict=True)):
            features, speed_limit, has_speed_limit = values
            arrays["lanes"][index] = features
            arrays["lanes_speed_limit"][index, 0] = speed_limit
            arrays["lanes_has_speed_limit"][index, 0] = has_speed_limit
            encoded_by_id[record.snapshot.stable_id] = values
        for index, record in enumerate(route_records):
            features, speed_limit, has_speed_limit = encoded_by_id[record.snapshot.stable_id]
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
        self, snapshots: tuple[_LaneSnapshot, ...], rear_axle: np.ndarray
    ) -> list[_LaneRecord]:
        records: list[_LaneRecord] = []
        for snapshot in snapshots:
            distance = float(snapshot.lane.distance(rear_axle))
            if not np.isfinite(distance) or distance < 0.0:
                raise ValueError(
                    f"lane {snapshot.stable_id} returned invalid distance {distance!r}"
                )
            if distance <= self._query_radius_m:
                records.append(_LaneRecord(snapshot=snapshot, distance=distance))
        records.sort(key=lambda record: (record.distance, record.snapshot.stable_id))
        selected = records[: self._config.lane_num]
        if not selected:
            raise RuntimeError("no lanes were found within the configured map query radius")
        return selected

    def _candidate_snapshots(self, rear_axle: np.ndarray) -> tuple[_LaneSnapshot, ...]:
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
        self, lane_records: list[_LaneRecord], navigation: Any
    ) -> list[_LaneRecord]:
        checkpoints = getattr(navigation, "checkpoints", None)
        if not isinstance(checkpoints, (list, tuple)) or len(checkpoints) < 2:
            raise RuntimeError("navigation checkpoints must contain at least one road")
        route_roads = list(zip(checkpoints[:-1], checkpoints[1:], strict=True))
        local_roads = {record.snapshot.road for record in lane_records}
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
        selected = [record for record in lane_records if record.snapshot.road in route_road_set]
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

    def _encode_lanes(
        self,
        records: list[_LaneRecord],
        rear_axle: np.ndarray,
        ego_heading: float,
    ) -> list[tuple[np.ndarray, float, bool]]:
        if (
            self._lane_snapshots is None
            or self._lane_centers_world is None
            or self._lane_left_world is None
            or self._lane_right_world is None
        ):
            raise RuntimeError("map geometry cache is unavailable before reset")
        snapshot_indexes = {
            snapshot.stable_id: index for index, snapshot in enumerate(self._lane_snapshots)
        }
        indexes = np.fromiter(
            (snapshot_indexes[record.snapshot.stable_id] for record in records),
            dtype=np.intp,
            count=len(records),
        )
        centers_world = self._lane_centers_world[indexes]
        left_world = self._lane_left_world[indexes]
        right_world = self._lane_right_world[indexes]
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
        results: list[tuple[np.ndarray, float, bool]] = []
        for index, record in enumerate(records):
            features = np.zeros((self._config.lane_len, _LANE_FEATURE_DIM), dtype=np.float32)
            features[:, :2] = centers_local[index]
            features[:-1, 2:4] = np.diff(centers_local[index], axis=0)
            features[:, 4:6] = left_local[index] - centers_local[index]
            features[:, 6:8] = right_local[index] - centers_local[index]
            features[:, 8:12] = _TRAFFIC_LIGHT_UNKNOWN
            snapshot = record.snapshot
            results.append((features, snapshot.speed_limit_mps, snapshot.has_speed_limit))
        return results
