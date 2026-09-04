"""Combine pure scene and map arrays at the planner observation boundary."""

from __future__ import annotations

import numpy as np
import torch
from tensordict import TensorDict, TensorDictBase

from .arrays import MapObservationArrays
from .history import TrafficHistory
from .scene import TrafficObservationAudit, TrafficSceneEncoder
from .schema import PLANNER_OBSERVATION_FIELDS

_EGO_CURRENT_STATE = np.array(
    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    dtype=PLANNER_OBSERVATION_FIELDS["ego_current_state"][1],
)


class ObservationBuilder:
    """Build one raw CPU planner observation from immutable scene and map inputs."""

    def __init__(self, scene_encoder: TrafficSceneEncoder) -> None:
        self._scene_encoder = scene_encoder

    def build(
        self, history: TrafficHistory, map_arrays: MapObservationArrays
    ) -> tuple[TensorDictBase, TrafficObservationAudit]:
        neighbors, static_objects, audit = self._scene_encoder.build(history)
        return self._assemble(map_arrays, neighbors, static_objects), audit

    def build_empty_scene(self, map_arrays: MapObservationArrays) -> TensorDictBase:
        """Build the no-traffic variant through the same TensorDict/map boundary."""

        return self._assemble(
            map_arrays,
            np.zeros(*PLANNER_OBSERVATION_FIELDS["neighbor_agents_past"]),
            np.zeros(*PLANNER_OBSERVATION_FIELDS["static_objects"]),
        )

    @staticmethod
    def _assemble(
        map_arrays: MapObservationArrays,
        neighbors: np.ndarray,
        static_objects: np.ndarray,
    ) -> TensorDictBase:
        arrays = {
            "ego_current_state": _EGO_CURRENT_STATE.copy(),
            "neighbor_agents_past": neighbors,
            "static_objects": static_objects,
            **map_arrays,
        }
        return TensorDict(
            {name: torch.from_numpy(arrays[name]) for name in PLANNER_OBSERVATION_FIELDS},
            batch_size=[],
        )
