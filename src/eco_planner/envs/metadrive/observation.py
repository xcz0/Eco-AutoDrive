"""MetaDrive sources for the canonical planner observation pipeline."""

from __future__ import annotations

from typing import Any, Protocol

from tensordict import TensorDictBase

from ..domain.traffic import TrafficFrame
from ..observation.builder import ObservationBuilder
from ..observation.history import TrafficHistory
from ..observation.scene import TrafficObservationAudit, TrafficSceneEncoder
from .map import MetaDriveMapAdapter


class MetaDriveObservationPipeline(Protocol):
    """Common state and output boundary for traffic and no-traffic observations."""

    def reset(self, env: Any, initial_frame: TrafficFrame) -> None: ...

    def append_frames(self, frames: tuple[TrafficFrame, ...]) -> None: ...

    def build(self, env: Any) -> tuple[TensorDictBase, TrafficObservationAudit | None]: ...


class TrafficMetaDriveObservationPipeline:
    """Build planner observations from traffic history and current local map state."""

    def __init__(self, query_radius_m: float) -> None:
        self._history = TrafficHistory()
        self._map_adapter = MetaDriveMapAdapter(query_radius_m)
        self._builder = ObservationBuilder(TrafficSceneEncoder(query_radius_m))

    def reset(self, env: Any, initial_frame: TrafficFrame) -> None:
        self._history.reset(initial_frame)
        self._map_adapter.reset(env)

    def append_frames(self, frames: tuple[TrafficFrame, ...]) -> None:
        self._history.append(frames)

    def build(self, env: Any) -> tuple[TensorDictBase, TrafficObservationAudit]:
        self._validate_current_step(env, self._history.latest.simulator_step)
        return self._builder.build(
            self._history,
            self._map_adapter.build_arrays(env, allow_empty_route=env.is_out_of_road_terminal),
        )

    @staticmethod
    def _validate_current_step(env: Any, expected_step: int) -> None:
        current_step = getattr(getattr(env, "engine", None), "episode_step", None)
        if current_step != expected_step:
            raise RuntimeError("latest traffic frame does not match the current simulator step")


class NoTrafficMetaDriveObservationPipeline:
    """Build empty-scene planner observations after validating captured frames."""

    def __init__(self, query_radius_m: float) -> None:
        self._map_adapter = MetaDriveMapAdapter(query_radius_m)
        self._builder = ObservationBuilder(TrafficSceneEncoder(query_radius_m))
        self._simulator_step: int | None = None

    def reset(self, env: Any, initial_frame: TrafficFrame) -> None:
        self._validate_environment_config(env)
        self._validate_empty_frame(initial_frame)
        self._simulator_step = initial_frame.simulator_step
        self._map_adapter.reset(env)

    def append_frames(self, frames: tuple[TrafficFrame, ...]) -> None:
        if self._simulator_step is None:
            raise RuntimeError("no-traffic observation pipeline is unavailable before reset")
        expected_step = self._simulator_step
        for frame in frames:
            expected_step += 1
            if frame.simulator_step != expected_step:
                raise ValueError(
                    "no-traffic simulator steps must be consecutive: "
                    f"expected {expected_step}, got {frame.simulator_step}"
                )
            self._validate_empty_frame(frame)
        self._simulator_step = expected_step

    def build(self, env: Any) -> tuple[TensorDictBase, None]:
        if self._simulator_step is None:
            raise RuntimeError("no-traffic observation pipeline is unavailable before reset")
        TrafficMetaDriveObservationPipeline._validate_current_step(env, self._simulator_step)
        return (
            self._builder.build_empty_scene(
                self._map_adapter.build_arrays(env, allow_empty_route=env.is_out_of_road_terminal)
            ),
            None,
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
    def _validate_empty_frame(frame: TrafficFrame) -> None:
        if frame.participants or frame.static_objects:
            raise RuntimeError(
                "no-traffic observation received unsupported scene objects: "
                f"dynamic={[state.object_id for state in frame.participants]}, "
                f"static={[state.object_id for state in frame.static_objects]}"
            )
