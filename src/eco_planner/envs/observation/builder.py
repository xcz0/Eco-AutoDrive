"""Combine pure scene and map arrays at the planner observation boundary."""

from __future__ import annotations

import numpy as np
import torch

from eco_planner.envs.array_types import NumpyMapObservation, SingleObservation
from eco_planner.envs.domain.traffic import TrafficFrame
from eco_planner.envs.observation.scene import TrafficObservationAudit, TrafficSceneEncoder

_EGO_CURRENT_STATE = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)


class ObservationBuilder:
    """Build one raw CPU planner observation from immutable scene and map inputs."""

    def __init__(self, scene_encoder: TrafficSceneEncoder) -> None:
        self._scene_encoder = scene_encoder

    def build(
        self, frames: tuple[TrafficFrame, ...], map_arrays: NumpyMapObservation
    ) -> tuple[SingleObservation, TrafficObservationAudit]:
        neighbors, static_objects, audit = self._scene_encoder.build(frames)
        return {
            "ego_current_state": torch.from_numpy(_EGO_CURRENT_STATE.copy()),
            "neighbor_agents_past": torch.from_numpy(neighbors),
            "static_objects": torch.from_numpy(static_objects),
            "lanes": torch.from_numpy(map_arrays["lanes"]),
            "lanes_speed_limit": torch.from_numpy(map_arrays["lanes_speed_limit"]),
            "lanes_has_speed_limit": torch.from_numpy(map_arrays["lanes_has_speed_limit"]),
            "route_lanes": torch.from_numpy(map_arrays["route_lanes"]),
            "route_lanes_speed_limit": torch.from_numpy(map_arrays["route_lanes_speed_limit"]),
            "route_lanes_has_speed_limit": torch.from_numpy(
                map_arrays["route_lanes_has_speed_limit"]
            ),
        }, audit
