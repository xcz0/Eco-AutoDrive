"""MetaDrive snapshot and map glue for the canonical observation pipeline."""

from __future__ import annotations

from typing import Any

from metadrive.component.static_object.traffic_object import TrafficObject
from metadrive.component.traffic_participants.base_traffic_participant import (
    BaseTrafficParticipant,
)
from metadrive.component.vehicle.base_vehicle import BaseVehicle

from ..domain.traffic import TrafficFrame
from ..observation import MapSnapshot, PlannerObservationSpec, TrafficHistory
from .map import MetaDriveMapAdapter


class MetaDriveObservationSource:
    """Adapt MetaDrive state into domain history and map snapshots without encoding features."""

    def __init__(self, observation_spec: PlannerObservationSpec, query_radius_m: float) -> None:
        self._history = TrafficHistory()
        self._map_adapter = MetaDriveMapAdapter(observation_spec, query_radius_m)

    @property
    def history(self) -> TrafficHistory:
        """Return the canonical traffic-history state machine."""

        return self._history

    def reset(self, env: Any, initial_frame: TrafficFrame) -> None:
        """Seed immutable traffic history and capture reset-time map topology."""

        self._history.reset(initial_frame)
        self._map_adapter.reset(env)

    def append_frames(self, frames: tuple[TrafficFrame, ...]) -> None:
        """Commit consecutive simulator snapshots from one executed trajectory prefix."""

        self._history.append(frames)

    def map_snapshot(self, env: Any) -> MapSnapshot:
        """Extract current local map arrays at the simulator/domain boundary."""

        latest = self._history.latest
        current_step = getattr(getattr(env, "engine", None), "episode_step", None)
        if current_step != latest.simulator_step:
            raise RuntimeError("latest traffic frame does not match the current simulator step")
        return MapSnapshot(
            self._map_adapter.build_arrays(env, allow_empty_route=env.is_out_of_road_terminal)
        )


class NoTrafficMetaDriveObservationSource:
    """Extract map snapshots after proving the configured MetaDrive scene is empty."""

    def __init__(self, observation_spec: PlannerObservationSpec, query_radius_m: float) -> None:
        self._map_adapter = MetaDriveMapAdapter(observation_spec, query_radius_m)

    def reset(self, env: Any) -> None:
        """Capture immutable map geometry for the reset no-traffic episode."""

        self._map_adapter.reset(env)

    def map_snapshot(self, env: Any) -> MapSnapshot:
        """Validate no-traffic conditions and return local map arrays."""

        self._validate_environment_config(env)
        self._validate_scene_is_empty(env)
        return MapSnapshot(
            self._map_adapter.build_arrays(env, allow_empty_route=env.is_out_of_road_terminal)
        )

    @staticmethod
    def _validate_environment_config(env: Any) -> None:
        config = getattr(env, "config", None)
        if config is None:
            raise RuntimeError("MetaDrive environment does not expose its configuration")
        required = {"traffic_density": 0.0, "random_traffic": False, "accident_prob": 0.0}
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
