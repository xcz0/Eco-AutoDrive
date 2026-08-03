"""No-traffic MetaDrive observations for official Diffusion Planner inference."""

from __future__ import annotations

from typing import Any

import torch
from metadrive.component.static_object.traffic_object import TrafficObject
from metadrive.component.traffic_participants.base_traffic_participant import (
    BaseTrafficParticipant,
)
from metadrive.component.vehicle.base_vehicle import BaseVehicle

from eco_planner.envs.map_adapter import MetaDriveMapAdapter
from eco_planner.models.config import OfficialDiffusionPlannerConfig
from eco_planner.models.contracts import validate_official_observation


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
                [[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
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
