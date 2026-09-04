"""Traffic selection and local-frame encoding over domain snapshots."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from eco_planner.contracts import (
    AGENT_COUNT,
    AGENT_HISTORY_DIM,
    STATIC_OBJECT_COUNT,
    STATIC_OBJECT_DIM,
)

from ..domain.geometry import rear_axle_position, world_vectors_to_local
from .arrays import NeighborAgentsArray, StaticObjectsArray
from .history import TrafficHistory


@dataclass(frozen=True, slots=True)
class TrafficObservationAudit:
    selected_participant_ids: tuple[str, ...]
    participant_count_in_radius: int
    static_object_count_in_radius: int
    nearest_participant_distance_m: float | None


class TrafficSceneEncoder:
    """Encode current-scene objects from a history of project-owned snapshots."""

    def __init__(self, query_radius_m: float) -> None:
        self._query_radius_m = query_radius_m

    def build(
        self, history: TrafficHistory
    ) -> tuple[NeighborAgentsArray, StaticObjectsArray, TrafficObservationAudit]:
        frames = history.require_full()
        latest = frames[-1]
        anchor = rear_axle_position(
            np.asarray(latest.ego_center_xy_m, dtype=np.float64),
            latest.ego_heading_rad,
            latest.ego_rear_wheelbase_m,
        )
        participant_distances = [
            (_distance(state.position_xy_m, anchor), state) for state in latest.participants
        ]
        participant_distances.sort(key=lambda item: (item[0], item[1].object_id))
        in_radius = [
            state for distance, state in participant_distances if distance <= self._query_radius_m
        ]
        history_by_id = [
            {state.object_id: state for state in frame.participants} for frame in frames
        ]
        neighbor_agents = np.zeros((AGENT_COUNT, len(frames), AGENT_HISTORY_DIM), dtype=np.float32)
        for output_index, state in enumerate(in_radius[:AGENT_COUNT]):
            historical_states = [state] * len(frames)
            filled = state
            for history_index in range(len(frames) - 1, -1, -1):
                historical = history_by_id[history_index].get(state.object_id)
                if historical is not None:
                    filled = historical
                historical_states[history_index] = filled
            for history_index, historical in enumerate(historical_states):
                local_position = world_vectors_to_local(
                    np.asarray([historical.position_xy_m], dtype=np.float64) - anchor,
                    latest.ego_heading_rad,
                )[0]
                local_velocity = world_vectors_to_local(
                    np.asarray([historical.velocity_xy_mps], dtype=np.float64),
                    latest.ego_heading_rad,
                )[0]
                heading = historical.heading_rad - latest.ego_heading_rad
                neighbor_agents[output_index, history_index] = (
                    local_position[0],
                    local_position[1],
                    np.cos(heading),
                    np.sin(heading),
                    local_velocity[0],
                    local_velocity[1],
                    historical.width_m,
                    historical.length_m,
                    *_participant_kind_features(historical.kind),
                )
        static_objects = np.zeros((STATIC_OBJECT_COUNT, STATIC_OBJECT_DIM), dtype=np.float32)
        static_distances = [
            (_distance(state.position_xy_m, anchor), state) for state in latest.static_objects
        ]
        static_distances.sort(key=lambda item: (item[0], item[1].object_id))
        static_in_radius = [
            state for distance, state in static_distances if distance <= self._query_radius_m
        ]
        for output_index, state in enumerate(static_in_radius[:STATIC_OBJECT_COUNT]):
            local = world_vectors_to_local(
                np.asarray([state.position_xy_m], dtype=np.float64) - anchor,
                latest.ego_heading_rad,
            )[0]
            heading = state.heading_rad - latest.ego_heading_rad
            static_objects[output_index] = (
                local[0],
                local[1],
                np.cos(heading),
                np.sin(heading),
                state.width_m,
                state.length_m,
                *_static_kind_features(state.kind),
            )
        distances = [
            distance for distance, _ in participant_distances if distance <= self._query_radius_m
        ]
        return (
            neighbor_agents,
            static_objects,
            TrafficObservationAudit(
                selected_participant_ids=tuple(
                    history.artifact_participant_id(state.object_id)
                    for state in in_radius[:AGENT_COUNT]
                ),
                participant_count_in_radius=len(in_radius),
                static_object_count_in_radius=len(static_in_radius),
                nearest_participant_distance_m=min(distances, default=None),
            ),
        )


def _distance(position_xy_m: tuple[float, float], anchor: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(position_xy_m, dtype=np.float64) - anchor))


def _participant_kind_features(kind: str) -> tuple[float, float, float]:
    return {
        "vehicle": (1.0, 0.0, 0.0),
        "pedestrian": (0.0, 1.0, 0.0),
        "bicycle": (0.0, 0.0, 1.0),
    }[kind]


def _static_kind_features(kind: str) -> tuple[float, float, float, float]:
    return {
        "barrier": (0.0, 1.0, 0.0, 0.0),
        "traffic_cone": (0.0, 0.0, 1.0, 0.0),
        "generic": (0.0, 0.0, 0.0, 1.0),
    }[kind]
