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
from eco_planner.envs.array_types import NeighborAgentsArray, StaticObjectsArray
from eco_planner.envs.domain.traffic import TrafficFrame
from eco_planner.envs.geometry import rear_axle_position, world_vectors_to_local


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
        self, frames: tuple[TrafficFrame, ...]
    ) -> tuple[NeighborAgentsArray, StaticObjectsArray, TrafficObservationAudit]:
        latest = frames[-1]
        anchor = rear_axle_position(
            np.asarray(latest.ego_center_xy_m, dtype=np.float64),
            latest.ego_heading_rad,
            latest.ego_rear_wheelbase_m,
        )
        participants = sorted(
            latest.participants,
            key=lambda state: (_distance(state.position_xy_m, anchor), state.object_id),
        )
        in_radius = [
            state
            for state in participants
            if _distance(state.position_xy_m, anchor) <= self._query_radius_m
        ]
        neighbor_agents = np.zeros((AGENT_COUNT, len(frames), AGENT_HISTORY_DIM), dtype=np.float32)
        for output_index, state in enumerate(in_radius[:AGENT_COUNT]):
            for history_index, frame in enumerate(frames):
                historical = next(
                    (item for item in frame.participants if item.object_id == state.object_id),
                    state,
                )
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
                    _participant_kind_code(historical.kind),
                    0.0,
                    0.0,
                )
        static_objects = np.zeros((STATIC_OBJECT_COUNT, STATIC_OBJECT_DIM), dtype=np.float32)
        static = sorted(
            latest.static_objects,
            key=lambda state: (_distance(state.position_xy_m, anchor), state.object_id),
        )
        static_in_radius = [
            state
            for state in static
            if _distance(state.position_xy_m, anchor) <= self._query_radius_m
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
                _static_kind_code(state.kind),
                0.0,
                0.0,
                0.0,
            )
        distances = [_distance(state.position_xy_m, anchor) for state in in_radius]
        return (
            neighbor_agents,
            static_objects,
            TrafficObservationAudit(
                selected_participant_ids=tuple(
                    state.object_id for state in in_radius[:AGENT_COUNT]
                ),
                participant_count_in_radius=len(in_radius),
                static_object_count_in_radius=len(static_in_radius),
                nearest_participant_distance_m=min(distances, default=None),
            ),
        )


def _distance(position_xy_m: tuple[float, float], anchor: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(position_xy_m, dtype=np.float64) - anchor))


def _participant_kind_code(kind: str) -> float:
    return {"vehicle": 0.0, "pedestrian": 1.0, "bicycle": 2.0}[kind]


def _static_kind_code(kind: str) -> float:
    return {"barrier": 0.0, "traffic_cone": 1.0, "generic": 2.0}[kind]
