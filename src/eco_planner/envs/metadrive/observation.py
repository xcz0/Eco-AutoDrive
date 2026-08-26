"""MetaDrive observations for official Diffusion Planner inference."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from metadrive.component.static_object.traffic_object import TrafficObject
from metadrive.component.traffic_participants.base_traffic_participant import (
    BaseTrafficParticipant,
)
from metadrive.component.vehicle.base_vehicle import BaseVehicle

from eco_planner.envs.array_types import (
    EgoStateArray,
    EncodedParticipantHistoryArray,
    NeighborAgentsArray,
    NumpyObservation,
    ParticipantHistoryArray,
    ParticipantRowsArray,
    SingleObservation,
    StaticObjectArray,
    StaticObjectsArray,
    WorldVectorArray,
)
from eco_planner.envs.domain.traffic import (
    StaticTrafficObjectState,
    TrafficFrame,
)
from eco_planner.envs.geometry import rear_axle_position, world_vectors_to_local
from eco_planner.envs.metadrive.map import MetaDriveMapAdapter
from eco_planner.envs.observation import PlannerObservationSpec

_EGO_CURRENT_STATE: EgoStateArray = np.array(
    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32
)


@dataclass(frozen=True, slots=True)
class TrafficObservationAudit:
    """Selection metadata returned with one traffic observation."""

    selected_participant_ids: tuple[str, ...]
    participant_count_in_radius: int
    static_object_count_in_radius: int
    nearest_participant_distance_m: float | None


@dataclass(frozen=True, slots=True)
class _EncodedParticipantFrame:
    ids: tuple[str, ...]
    rows: ParticipantRowsArray
    index_by_id: dict[str, int]


def _to_cpu_tensors(arrays: NumpyObservation) -> SingleObservation:
    return {
        "ego_current_state": torch.from_numpy(arrays["ego_current_state"]),
        "neighbor_agents_past": torch.from_numpy(arrays["neighbor_agents_past"]),
        "static_objects": torch.from_numpy(arrays["static_objects"]),
        "lanes": torch.from_numpy(arrays["lanes"]),
        "lanes_speed_limit": torch.from_numpy(arrays["lanes_speed_limit"]),
        "lanes_has_speed_limit": torch.from_numpy(arrays["lanes_has_speed_limit"]),
        "route_lanes": torch.from_numpy(arrays["route_lanes"]),
        "route_lanes_speed_limit": torch.from_numpy(arrays["route_lanes_speed_limit"]),
        "route_lanes_has_speed_limit": torch.from_numpy(arrays["route_lanes_has_speed_limit"]),
    }


class MetaDriveObservationAdapter:
    """Build official observations from a strict 21-frame MetaDrive traffic history."""

    def __init__(
        self,
        observation_spec: PlannerObservationSpec,
        query_radius_m: float,
    ) -> None:
        self._config = observation_spec
        self._query_radius_m = float(query_radius_m)
        self._map_adapter = MetaDriveMapAdapter(observation_spec, self._query_radius_m)
        self._latest_frame: TrafficFrame | None = None
        self._encoded_history: deque[_EncodedParticipantFrame] = deque(
            maxlen=observation_spec.time_len
        )
        self._artifact_participant_ids: dict[str, str] = {}

    def reset(self, env: Any, initial_frame: TrafficFrame) -> None:
        """Reset map state and seed traffic history with the simulator reset frame."""

        self._latest_frame = initial_frame
        self._encoded_history.clear()
        self._encoded_history.append(_encode_participants(initial_frame))
        self._artifact_participant_ids.clear()
        _register_artifact_participant_ids(initial_frame, self._artifact_participant_ids)
        self._map_adapter.reset(env)

    def append_frames(self, frames: tuple[TrafficFrame, ...]) -> None:
        """Append consecutive 10 Hz frames returned by one trajectory action."""

        if self._latest_frame is None:
            raise RuntimeError("traffic history is unavailable before reset")
        previous_step = self._latest_frame.simulator_step
        encoded_frames: list[_EncodedParticipantFrame] = []
        staged_artifact_ids = dict(self._artifact_participant_ids)
        for frame in frames:
            if frame.simulator_step != previous_step + 1:
                raise ValueError(
                    "traffic history simulator steps must be consecutive: "
                    f"expected {previous_step + 1}, got {frame.simulator_step}"
                )
            _register_artifact_participant_ids(frame, staged_artifact_ids)
            encoded_frames.append(_encode_participants(frame))
            previous_step = frame.simulator_step
        self._artifact_participant_ids = staged_artifact_ids
        self._encoded_history.extend(encoded_frames)
        if frames:
            self._latest_frame = frames[-1]

    def build(self, env: Any) -> tuple[SingleObservation, TrafficObservationAudit]:
        """Return one observation and its traffic-selection audit."""

        if len(self._encoded_history) != self._config.time_len:
            raise RuntimeError(
                f"traffic history must contain exactly {self._config.time_len} frames; "
                f"received {len(self._encoded_history)}"
            )
        if self._latest_frame is None:
            raise RuntimeError("traffic history is unavailable before reset")
        current_step = getattr(getattr(env, "engine", None), "episode_step", None)
        latest = self._latest_frame
        if current_step != latest.simulator_step:
            raise RuntimeError("latest traffic frame does not match the current simulator step")

        neighbor_agents, neighbor_audit = self._build_neighbor_agents(latest)
        static_objects, static_count = self._build_static_objects(latest)
        audit = TrafficObservationAudit(
            selected_participant_ids=neighbor_audit.selected_participant_ids,
            participant_count_in_radius=neighbor_audit.participant_count_in_radius,
            static_object_count_in_radius=static_count,
            nearest_participant_distance_m=neighbor_audit.nearest_participant_distance_m,
        )
        observation: NumpyObservation = {
            "ego_current_state": _EGO_CURRENT_STATE.copy(),
            "neighbor_agents_past": neighbor_agents,
            "static_objects": static_objects,
            **self._map_adapter.build_arrays(env),
        }
        return _to_cpu_tensors(observation), audit

    def _build_neighbor_agents(
        self, latest: TrafficFrame
    ) -> tuple[NeighborAgentsArray, TrafficObservationAudit]:
        anchor_xy, anchor_heading = _rear_axle_anchor(latest)
        current = self._encoded_history[-1]
        distances_array = np.linalg.norm(current.rows[:, :2] - anchor_xy, axis=1)
        distances = dict(zip(current.ids, distances_array, strict=True))
        in_radius = [
            object_id
            for object_id, distance in zip(current.ids, distances_array, strict=True)
            if distance <= self._query_radius_m
        ]
        in_radius.sort(key=lambda object_id: (distances[object_id], object_id))
        selected = in_radius[: self._config.agent_num]
        result = np.zeros(
            (self._config.agent_num, self._config.time_len, self._config.agent_state_dim),
            dtype=np.float32,
        )
        histories = tuple(self._encoded_history)
        selected_rows = np.empty((len(selected), len(histories), 8), dtype=np.float64)
        for output_index, object_id in enumerate(selected):
            filled = current.rows[current.index_by_id[object_id]]
            for reverse_index, frame in enumerate(reversed(histories)):
                row_index = frame.index_by_id.get(object_id)
                if row_index is not None:
                    filled = frame.rows[row_index]
                selected_rows[output_index, len(histories) - reverse_index - 1] = filled
        if selected:
            result[: len(selected)] = _encoded_participant_history_features(
                selected_rows, anchor_xy, anchor_heading
            )
        nearest = min((distances[object_id] for object_id in in_radius), default=None)
        return result, TrafficObservationAudit(
            selected_participant_ids=tuple(
                self._artifact_participant_ids[object_id] for object_id in selected
            ),
            participant_count_in_radius=len(in_radius),
            static_object_count_in_radius=0,
            nearest_participant_distance_m=nearest,
        )

    def _build_static_objects(self, latest: TrafficFrame) -> tuple[StaticObjectsArray, int]:
        anchor_xy, anchor_heading = _rear_axle_anchor(latest)
        unique = {state.object_id: state for state in latest.static_objects}
        distances = {
            object_id: float(np.linalg.norm(np.asarray(state.position_xy_m) - anchor_xy))
            for object_id, state in unique.items()
        }
        in_radius = [
            object_id
            for object_id, distance in distances.items()
            if distance <= self._query_radius_m
        ]
        in_radius.sort(key=lambda object_id: (distances[object_id], object_id))
        result = np.zeros(
            (self._config.static_objects_num, self._config.static_objects_state_dim),
            dtype=np.float32,
        )
        for output_index, object_id in enumerate(in_radius[: self._config.static_objects_num]):
            result[output_index] = _static_object_features(
                unique[object_id], anchor_xy, anchor_heading
            )
        return result, len(in_radius)


def _encode_participants(frame: TrafficFrame) -> _EncodedParticipantFrame:
    ids = tuple(state.object_id for state in frame.participants)
    rows = np.empty((len(ids), 8), dtype=np.float64)
    kind_codes = {"vehicle": 0.0, "pedestrian": 1.0, "bicycle": 2.0}
    for index, state in enumerate(frame.participants):
        rows[index] = (
            *state.position_xy_m,
            state.heading_rad,
            *state.velocity_xy_mps,
            state.width_m,
            state.length_m,
            kind_codes[state.kind],
        )
    rows.setflags(write=False)
    return _EncodedParticipantFrame(
        ids=ids,
        rows=rows,
        index_by_id={object_id: index for index, object_id in enumerate(ids)},
    )


def _register_artifact_participant_ids(frame: TrafficFrame, artifact_ids: dict[str, str]) -> None:
    unseen = [state for state in frame.participants if state.object_id not in artifact_ids]
    keyed = sorted(
        (
            (
                state.kind,
                *state.position_xy_m,
                state.heading_rad,
                *state.velocity_xy_mps,
                state.width_m,
                state.length_m,
            ),
            state.object_id,
        )
        for state in unseen
    )
    for previous, current in zip(keyed, keyed[1:], strict=False):
        if previous[0] == current[0]:
            raise RuntimeError(
                "new traffic participants have indistinguishable physical identity keys"
            )
    for _, object_id in keyed:
        artifact_ids[object_id] = f"participant-{len(artifact_ids):06d}"


def _rear_axle_anchor(frame: TrafficFrame) -> tuple[WorldVectorArray, float]:
    heading = frame.ego_heading_rad
    center = np.asarray(frame.ego_center_xy_m, dtype=np.float64)
    return rear_axle_position(center, heading, frame.ego_rear_wheelbase_m), heading


def _encoded_participant_history_features(
    histories: ParticipantHistoryArray,
    anchor_xy: WorldVectorArray,
    anchor_heading: float,
) -> EncodedParticipantHistoryArray:
    shape = histories.shape[:2]
    positions = histories[..., :2]
    velocities = histories[..., 3:5]
    local_positions = world_vectors_to_local(
        (positions - anchor_xy).reshape(-1, 2), anchor_heading
    ).reshape((*shape, 2))
    local_velocities = world_vectors_to_local(velocities.reshape(-1, 2), anchor_heading).reshape(
        (*shape, 2)
    )
    result = np.zeros((*shape, 11), dtype=np.float32)
    result[..., :2] = local_positions
    relative_headings = histories[..., 2] - anchor_heading
    result[..., 2] = np.cos(relative_headings)
    result[..., 3] = np.sin(relative_headings)
    result[..., 4:6] = local_velocities
    result[..., 6:8] = histories[..., 5:7]
    type_codes = histories[..., 7].astype(np.intp)
    result[..., 8:11] = np.eye(3, dtype=np.float32)[type_codes]
    return result


def _static_object_features(
    state: StaticTrafficObjectState, anchor_xy: WorldVectorArray, anchor_heading: float
) -> StaticObjectArray:
    position = world_vectors_to_local(
        (np.asarray(state.position_xy_m) - anchor_xy)[None], anchor_heading
    )[0]
    relative_heading = state.heading_rad - anchor_heading
    type_features = {
        "barrier": (0.0, 1.0, 0.0, 0.0),
        "traffic_cone": (0.0, 0.0, 1.0, 0.0),
        "generic": (0.0, 0.0, 0.0, 1.0),
    }[state.kind]
    return np.asarray(
        [
            *position,
            np.cos(relative_heading),
            np.sin(relative_heading),
            state.width_m,
            state.length_m,
            *type_features,
        ],
        dtype=np.float32,
    )


class NoTrafficMetaDriveObservationAdapter:
    """Build the official model input for a strictly empty MetaDrive traffic scene."""

    def __init__(
        self,
        observation_spec: PlannerObservationSpec,
        query_radius_m: float,
    ) -> None:
        self._config = observation_spec
        self._map_adapter = MetaDriveMapAdapter(observation_spec, query_radius_m)

    def reset(self, env: Any) -> None:
        """Capture immutable map geometry for the reset no-traffic episode."""

        self._map_adapter.reset(env)

    def build(self, env: Any) -> SingleObservation:
        """Return one observation and reject any non-empty traffic scene."""

        self._validate_environment_config(env)
        self._validate_scene_is_empty(env)

        config = self._config
        observation: NumpyObservation = {
            "ego_current_state": _EGO_CURRENT_STATE.copy(),
            "neighbor_agents_past": np.zeros(
                (config.agent_num, config.time_len, config.agent_state_dim), dtype=np.float32
            ),
            "static_objects": np.zeros(
                (config.static_objects_num, config.static_objects_state_dim), dtype=np.float32
            ),
            **self._map_adapter.build_arrays(env),
        }
        return _to_cpu_tensors(observation)

    @staticmethod
    def _validate_environment_config(env: Any) -> None:
        config = getattr(env, "config", None)
        if config is None:
            raise RuntimeError("MetaDrive environment does not expose its configuration")
        required = {
            "traffic_density": 0.0,
            "random_traffic": False,
            "accident_prob": 0.0,
        }
        missing = sorted(set(required) - set(config))
        if missing:
            raise ValueError(f"MetaDrive no-traffic configuration is missing: {missing}")
        for name, expected in required.items():
            actual = config[name]
            if isinstance(expected, bool):
                valid = type(actual) is bool and actual is expected
            else:
                valid = type(actual) in {int, float} and float(actual) == expected
            if not valid:
                raise ValueError(f"{name} must be explicitly configured as {expected!r}")

    @staticmethod
    def _validate_scene_is_empty(env: Any) -> None:
        ego = getattr(env, "agent", None)
        engine = getattr(env, "engine", None)
        if ego is None or engine is None:
            raise RuntimeError("MetaDrive environment must be reset before building observations")
        objects = engine.get_objects()
        if not isinstance(objects, dict):
            raise RuntimeError("MetaDrive engine objects must be exposed as a dictionary")
        dynamic = [
            name
            for name, value in objects.items()
            if value is not ego and isinstance(value, (BaseVehicle, BaseTrafficParticipant))
        ]
        static = [name for name, value in objects.items() if isinstance(value, TrafficObject)]
        if dynamic or static:
            raise RuntimeError(
                "no-traffic observation received unsupported scene objects: "
                f"dynamic={sorted(dynamic)}, static={sorted(static)}"
            )
