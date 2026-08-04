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

from eco_planner.envs.map_adapter import MetaDriveMapAdapter
from eco_planner.envs.traffic_state import (
    StaticTrafficObjectState,
    TrafficFrame,
    TrafficParticipantState,
)
from eco_planner.models.config import OfficialDiffusionPlannerConfig
from eco_planner.models.contracts import validate_official_observation

_EGO_CURRENT_STATE = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


@dataclass(frozen=True)
class TrafficObservationAudit:
    """Selection metadata for the most recently built traffic observation."""

    selected_participant_ids: tuple[str, ...]
    participant_count_in_radius: int
    static_object_count_in_radius: int
    nearest_participant_distance_m: float | None


class MetaDriveObservationAdapter:
    """Build official observations from a strict 21-frame MetaDrive traffic history."""

    def __init__(
        self,
        model_config: OfficialDiffusionPlannerConfig,
        query_radius_m: float,
    ) -> None:
        if not isinstance(model_config, OfficialDiffusionPlannerConfig):
            raise TypeError("model_config must be an OfficialDiffusionPlannerConfig")
        if type(query_radius_m) not in {int, float}:
            raise TypeError("query_radius_m must be a numeric value")
        if not np.isfinite(query_radius_m) or float(query_radius_m) <= 0.0:
            raise ValueError("query_radius_m must be finite and positive")
        self._config = model_config
        self._query_radius_m = float(query_radius_m)
        self._map_adapter = MetaDriveMapAdapter(model_config, self._query_radius_m)
        self._history: deque[TrafficFrame] = deque(maxlen=model_config.time_len)
        self._last_audit: TrafficObservationAudit | None = None

    @property
    def last_audit(self) -> TrafficObservationAudit:
        """Return selection metadata from the most recent successful build."""

        if self._last_audit is None:
            raise RuntimeError("traffic observation audit is unavailable before build")
        return self._last_audit

    def reset(self, initial_frame: TrafficFrame) -> None:
        """Clear prior episode state and seed history with the reset frame."""

        if not isinstance(initial_frame, TrafficFrame):
            raise TypeError("initial_frame must be a TrafficFrame")
        self._history.clear()
        self._history.append(initial_frame)
        self._last_audit = None

    def append_frames(self, frames: tuple[TrafficFrame, ...]) -> None:
        """Append consecutive 10 Hz frames returned by one trajectory action."""

        if not isinstance(frames, tuple) or not frames:
            raise TypeError("frames must be a non-empty tuple of TrafficFrame values")
        previous_step = self._history[-1].simulator_step if self._history else None
        for frame in frames:
            if not isinstance(frame, TrafficFrame):
                raise TypeError("frames must contain only TrafficFrame values")
            if previous_step is not None and frame.simulator_step != previous_step + 1:
                raise ValueError(
                    "traffic history simulator steps must be consecutive: "
                    f"expected {previous_step + 1}, got {frame.simulator_step}"
                )
            self._history.append(frame)
            previous_step = frame.simulator_step
        self._last_audit = None

    def build(self, env: Any, device: torch.device) -> dict[str, torch.Tensor]:
        """Return a batch-one official observation anchored at the latest ego rear axle."""

        if not isinstance(device, torch.device):
            raise TypeError("device must be a torch.device")
        if len(self._history) != self._config.time_len:
            raise RuntimeError(
                f"traffic history must contain exactly {self._config.time_len} frames; "
                f"received {len(self._history)}"
            )
        current_step = getattr(getattr(env, "engine", None), "episode_step", None)
        latest = self._history[-1]
        if current_step != latest.simulator_step:
            raise RuntimeError("latest traffic frame does not match the current simulator step")

        neighbor_agents, neighbor_audit = self._build_neighbor_agents(latest)
        static_objects, static_count = self._build_static_objects(latest)
        self._last_audit = TrafficObservationAudit(
            selected_participant_ids=neighbor_audit.selected_participant_ids,
            participant_count_in_radius=neighbor_audit.participant_count_in_radius,
            static_object_count_in_radius=static_count,
            nearest_participant_distance_m=neighbor_audit.nearest_participant_distance_m,
        )
        observation = {
            "ego_current_state": torch.tensor(
                [_EGO_CURRENT_STATE], dtype=torch.float32, device=device
            ),
            "neighbor_agents_past": torch.from_numpy(neighbor_agents)[None].to(device),
            "static_objects": torch.from_numpy(static_objects)[None].to(device),
        }
        observation.update(self._map_adapter.build(env, device))
        validate_official_observation(observation, device)
        return observation

    def _build_neighbor_agents(
        self, latest: TrafficFrame
    ) -> tuple[np.ndarray, TrafficObservationAudit]:
        anchor_xy, anchor_heading = _rear_axle_anchor(latest)
        current_by_id = _unique_participants(latest)
        distances = {
            object_id: float(np.linalg.norm(np.asarray(state.position_xy_m) - anchor_xy))
            for object_id, state in current_by_id.items()
        }
        in_radius = [
            object_id
            for object_id, distance in distances.items()
            if distance <= self._query_radius_m
        ]
        in_radius.sort(key=lambda object_id: (distances[object_id], object_id))
        selected = in_radius[: self._config.agent_num]
        result = np.zeros(
            (self._config.agent_num, self._config.time_len, self._config.agent_state_dim),
            dtype=np.float32,
        )
        histories = [_unique_participants(frame) for frame in self._history]
        for output_index, object_id in enumerate(selected):
            filled: TrafficParticipantState | None = None
            resolved: list[TrafficParticipantState] = []
            for frame_states in reversed(histories):
                state = frame_states.get(object_id)
                if state is not None:
                    if state.kind != current_by_id[object_id].kind:
                        raise RuntimeError(
                            f"traffic participant {object_id!r} changed type within history"
                        )
                    filled = state
                if filled is None:
                    raise RuntimeError(
                        f"current traffic participant {object_id!r} is absent from current frame"
                    )
                resolved.append(filled)
            for time_index, state in enumerate(reversed(resolved)):
                result[output_index, time_index] = _participant_features(
                    state, anchor_xy, anchor_heading
                )
        nearest = min((distances[object_id] for object_id in in_radius), default=None)
        return result, TrafficObservationAudit(
            selected_participant_ids=tuple(selected),
            participant_count_in_radius=len(in_radius),
            static_object_count_in_radius=0,
            nearest_participant_distance_m=nearest,
        )

    def _build_static_objects(self, latest: TrafficFrame) -> tuple[np.ndarray, int]:
        anchor_xy, anchor_heading = _rear_axle_anchor(latest)
        unique: dict[str, StaticTrafficObjectState] = {}
        for state in latest.static_objects:
            if state.object_id in unique:
                raise RuntimeError(f"duplicate static traffic object id: {state.object_id!r}")
            unique[state.object_id] = state
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


def _unique_participants(frame: TrafficFrame) -> dict[str, TrafficParticipantState]:
    result: dict[str, TrafficParticipantState] = {}
    for state in frame.participants:
        if state.object_id in result:
            raise RuntimeError(f"duplicate traffic participant id: {state.object_id!r}")
        result[state.object_id] = state
    return result


def _rear_axle_anchor(frame: TrafficFrame) -> tuple[np.ndarray, float]:
    heading = frame.ego_heading_rad
    direction = np.array([np.cos(heading), np.sin(heading)], dtype=np.float64)
    center = np.asarray(frame.ego_center_xy_m, dtype=np.float64)
    return center - frame.ego_rear_wheelbase_m * direction, heading


def _to_local_vector(vector: np.ndarray, anchor_heading: float) -> np.ndarray:
    cosine = np.cos(anchor_heading)
    sine = np.sin(anchor_heading)
    return np.array(
        [cosine * vector[0] + sine * vector[1], -sine * vector[0] + cosine * vector[1]],
        dtype=np.float64,
    )


def _participant_features(
    state: TrafficParticipantState, anchor_xy: np.ndarray, anchor_heading: float
) -> np.ndarray:
    position = _to_local_vector(np.asarray(state.position_xy_m) - anchor_xy, anchor_heading)
    velocity = _to_local_vector(np.asarray(state.velocity_xy_mps), anchor_heading)
    relative_heading = state.heading_rad - anchor_heading
    type_features = {
        "vehicle": (1.0, 0.0, 0.0),
        "pedestrian": (0.0, 1.0, 0.0),
        "bicycle": (0.0, 0.0, 1.0),
    }[state.kind]
    return np.asarray(
        [
            *position,
            np.cos(relative_heading),
            np.sin(relative_heading),
            *velocity,
            state.width_m,
            state.length_m,
            *type_features,
        ],
        dtype=np.float32,
    )


def _static_object_features(
    state: StaticTrafficObjectState, anchor_xy: np.ndarray, anchor_heading: float
) -> np.ndarray:
    position = _to_local_vector(np.asarray(state.position_xy_m) - anchor_xy, anchor_heading)
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
        model_config: OfficialDiffusionPlannerConfig,
        query_radius_m: float,
    ) -> None:
        if not isinstance(model_config, OfficialDiffusionPlannerConfig):
            raise TypeError("model_config must be an OfficialDiffusionPlannerConfig")
        self._config = model_config
        self._map_adapter = MetaDriveMapAdapter(model_config, query_radius_m)

    def build(self, env: Any, device: torch.device) -> dict[str, torch.Tensor]:
        """Return a batch-one observation and reject any non-empty traffic scene."""

        if not isinstance(device, torch.device):
            raise TypeError("device must be a torch.device")
        self._validate_environment_config(env)
        self._validate_scene_is_empty(env)

        config = self._config
        observation = {
            "ego_current_state": torch.tensor(
                [_EGO_CURRENT_STATE],
                dtype=torch.float32,
                device=device,
            ),
            "neighbor_agents_past": torch.zeros(
                (1, config.agent_num, config.time_len, config.agent_state_dim),
                dtype=torch.float32,
                device=device,
            ),
            "static_objects": torch.zeros(
                (1, config.static_objects_num, config.static_objects_state_dim),
                dtype=torch.float32,
                device=device,
            ),
        }
        observation.update(self._map_adapter.build(env, device))
        validate_official_observation(observation, device)
        return observation

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
