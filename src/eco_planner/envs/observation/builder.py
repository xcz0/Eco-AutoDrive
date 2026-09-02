"""Combine pure scene and map arrays at the planner observation boundary."""

from __future__ import annotations

import numpy as np
import torch
from tensordict import TensorDict

from eco_planner.contracts import (
    AGENT_COUNT,
    AGENT_HISTORY_DIM,
    STATIC_OBJECT_COUNT,
    STATIC_OBJECT_DIM,
    TRAFFIC_HISTORY_FRAMES,
)
from eco_planner.envs.array_types import SingleObservation
from eco_planner.envs.observation.history import TrafficHistory
from eco_planner.envs.observation.map import MapSnapshot
from eco_planner.envs.observation.scene import TrafficObservationAudit, TrafficSceneEncoder

_EGO_CURRENT_STATE = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)


class ObservationBuilder:
    """Build one raw CPU planner observation from immutable scene and map inputs."""

    def __init__(self, scene_encoder: TrafficSceneEncoder) -> None:
        self._scene_encoder = scene_encoder

    def build(
        self, history: TrafficHistory, map_snapshot: MapSnapshot
    ) -> tuple[SingleObservation, TrafficObservationAudit]:
        neighbors, static_objects, audit = self._scene_encoder.build(history)
        return TensorDict(
            {
            "ego_current_state": torch.from_numpy(_EGO_CURRENT_STATE.copy()),
            "neighbor_agents_past": torch.from_numpy(neighbors),
            "static_objects": torch.from_numpy(static_objects),
            "lanes": torch.from_numpy(map_snapshot.arrays["lanes"]),
            "lanes_speed_limit": torch.from_numpy(map_snapshot.arrays["lanes_speed_limit"]),
            "lanes_has_speed_limit": torch.from_numpy(map_snapshot.arrays["lanes_has_speed_limit"]),
            "route_lanes": torch.from_numpy(map_snapshot.arrays["route_lanes"]),
            "route_lanes_speed_limit": torch.from_numpy(
                map_snapshot.arrays["route_lanes_speed_limit"]
            ),
            "route_lanes_has_speed_limit": torch.from_numpy(
                map_snapshot.arrays["route_lanes_has_speed_limit"]
            ),
            },
            batch_size=[],
        ), audit

    def build_empty_scene(self, map_snapshot: MapSnapshot) -> SingleObservation:
        """Build the no-traffic variant through the same TensorDict/map boundary."""

        arrays = map_snapshot.arrays
        return TensorDict(
            {
                "ego_current_state": torch.from_numpy(_EGO_CURRENT_STATE.copy()),
                "neighbor_agents_past": torch.zeros(
                    (AGENT_COUNT, TRAFFIC_HISTORY_FRAMES, AGENT_HISTORY_DIM), dtype=torch.float32
                ),
                "static_objects": torch.zeros(
                    (STATIC_OBJECT_COUNT, STATIC_OBJECT_DIM), dtype=torch.float32
                ),
                "lanes": torch.from_numpy(arrays["lanes"]),
                "lanes_speed_limit": torch.from_numpy(arrays["lanes_speed_limit"]),
                "lanes_has_speed_limit": torch.from_numpy(arrays["lanes_has_speed_limit"]),
                "route_lanes": torch.from_numpy(arrays["route_lanes"]),
                "route_lanes_speed_limit": torch.from_numpy(arrays["route_lanes_speed_limit"]),
                "route_lanes_has_speed_limit": torch.from_numpy(
                    arrays["route_lanes_has_speed_limit"]
                ),
            },
            batch_size=[],
        )
