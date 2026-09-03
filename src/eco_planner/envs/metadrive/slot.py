"""Shared single-process MetaDrive environment lifecycle."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from tensordict import TensorDictBase

from eco_planner.contracts import PLANNER_HORIZON, TRAFFIC_HISTORY_WARMUP_STEPS

from ..array_types import TrajectoryArray
from ..domain import MetaDriveFuelProxyProvider, TrajectoryExecutionResult
from ..observation import (
    ObservationBuilder,
    PlannerObservationSpec,
    TrafficObservationAudit,
    TrafficSceneEncoder,
)
from .execution import TrajectoryExecutor, execution_steps_from_config
from .observation import MetaDriveObservationSource, NoTrafficMetaDriveObservationSource
from .simulator import MetaDriveBackend

ObservationMode = Literal["traffic", "no_traffic"]


@dataclass(frozen=True, slots=True)
class EnvSlotReset:
    """Metadata captured at reset before optional traffic warmup."""

    route_completion: float
    route_length_m: float
    warmup_initial_state: np.ndarray
    programmatic_lane_speed_limit_audit: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class EnvSlotObservation:
    """One raw planner observation and its optional traffic-selection audit."""

    observation: TensorDictBase
    traffic_audit: TrafficObservationAudit | None


class MetaDriveEnvSlot:
    """Compose one simulator backend, trajectory executor, and planner observation source."""

    def __init__(
        self,
        env_config: Mapping[str, Any],
        *,
        mode: ObservationMode,
        observation_spec: PlannerObservationSpec,
        map_query_radius_m: float,
        history_warmup_steps: int,
    ) -> None:
        if mode not in {"traffic", "no_traffic"}:
            raise ValueError("mode must be either 'traffic' or 'no_traffic'")
        if type(map_query_radius_m) not in {int, float} or map_query_radius_m <= 0.0:
            raise ValueError("map_query_radius_m must be a positive real scalar")
        expected_warmup = TRAFFIC_HISTORY_WARMUP_STEPS if mode == "traffic" else 0
        if history_warmup_steps != expected_warmup:
            raise ValueError(f"{mode} environments require history_warmup_steps={expected_warmup}")

        self._env_config = dict(env_config)
        self._execution_steps = execution_steps_from_config(self._env_config)
        self._mode = mode
        self._observation_spec = observation_spec
        self._map_query_radius_m = float(map_query_radius_m)
        self._history_warmup_steps = history_warmup_steps
        self._observation_source = self._create_observation_source()
        self._observation_builder = ObservationBuilder(
            TrafficSceneEncoder(self._map_query_radius_m)
        )
        self._backend = self._create_backend()
        self._executor = self._create_executor()

    @property
    def backend(self) -> MetaDriveBackend:
        """Return the owned simulator backend for rendering and MetaDrive-native audits."""

        return self._backend

    @property
    def route_completion(self) -> float:
        """Return the backend's current finite route-completion fraction."""

        return self._backend.route_completion

    @property
    def vehicle_state(self) -> np.ndarray:
        """Return the current seven-value world-frame ego state."""

        velocity = np.asarray(self._backend.agent.velocity, dtype=np.float64)
        return np.array(
            [
                *np.asarray(self._backend.agent.position, dtype=np.float64),
                float(self._backend.agent.heading_theta),
                *velocity,
                float(self._backend.agent.speed),
                0.0,
            ],
            dtype=np.float64,
        )

    def reset(self, *, map_name: str, seed: int) -> EnvSlotReset:
        """Reset this slot for one map and seed before traffic warmup."""

        if map_name != self._backend.config["map"]:
            self._replace_environment(map_name)
        self._backend.reset(seed=seed)
        initial_traffic_frame = self._executor.reset()
        if isinstance(self._observation_source, MetaDriveObservationSource):
            self._observation_source.reset(self._backend, initial_traffic_frame)
        else:
            self._observation_source.reset(self._backend)
        return EnvSlotReset(
            route_completion=self._backend.route_completion,
            route_length_m=self._backend.route_length_m,
            warmup_initial_state=self.vehicle_state,
            programmatic_lane_speed_limit_audit=(self._backend.programmatic_lane_speed_limit_audit),
        )

    def warmup(self) -> Iterator[TrajectoryExecutionResult]:
        """Advance stationary ego while yielding every completed warmup execution."""

        if self._mode == "no_traffic":
            return
        initial_position = self.vehicle_state[:2].copy()
        collected = 0
        while collected < self._history_warmup_steps:
            result = self.step(_stationary_trajectory())
            execution = result.execution
            yield result
            if result.terminated or result.truncated:
                raise RuntimeError("traffic history warmup ended before the required frame count")
            if (
                float(
                    np.linalg.norm(
                        execution.substep_states[:, :2] - initial_position,
                        axis=1,
                    ).max()
                )
                >= 1e-3
            ):
                raise RuntimeError("ego moved during stationary traffic history warmup")
            collected += execution.substep_states.shape[0]
        if collected != self._history_warmup_steps:
            raise RuntimeError("traffic history warmup overshot the required frame count")

    def observe(self) -> EnvSlotObservation:
        """Build the current raw CPU planner observation."""

        if isinstance(self._observation_source, MetaDriveObservationSource):
            observation, audit = self._observation_builder.build(
                self._observation_source.history,
                self._observation_source.map_snapshot(self._backend),
            )
            return EnvSlotObservation(observation, audit)
        return EnvSlotObservation(
            self._observation_builder.build_empty_scene(
                self._observation_source.map_snapshot(self._backend)
            ),
            None,
        )

    def step(self, trajectory: TrajectoryArray) -> TrajectoryExecutionResult:
        """Execute one trajectory prefix and commit its traffic frames."""

        result = self._executor.execute(trajectory)
        if isinstance(self._observation_source, MetaDriveObservationSource):
            self._observation_source.append_frames(result.execution.traffic_frames)
        return result

    def close(self) -> None:
        self._backend.close()

    def __enter__(self) -> MetaDriveEnvSlot:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _create_observation_source(
        self,
    ) -> MetaDriveObservationSource | NoTrafficMetaDriveObservationSource:
        if self._mode == "traffic":
            return MetaDriveObservationSource(self._observation_spec, self._map_query_radius_m)
        return NoTrafficMetaDriveObservationSource(self._observation_spec, self._map_query_radius_m)

    def _create_backend(self) -> MetaDriveBackend:
        backend_config = dict(self._env_config)
        for name in ("execution_mode", "trajectory_horizon", "trajectory_execution_steps"):
            backend_config.pop(name, None)
        return MetaDriveBackend(backend_config)

    def _create_executor(self) -> TrajectoryExecutor:
        return TrajectoryExecutor(
            self._backend,
            self._execution_steps,
            MetaDriveFuelProxyProvider(),
        )

    def recreate_environment(self) -> None:
        """Close and recreate the MetaDrive environment with the current map.

        MetaDrive's navigation state can corrupt after many same-map resets; this
        method rebuilds the environment from scratch without changing the config.
        """

        self._replace_environment(self._env_config["map"])

    def _replace_environment(self, map_name: str) -> None:
        self._backend.close()
        self._env_config["map"] = map_name
        self._observation_source = self._create_observation_source()
        self._backend = self._create_backend()
        self._executor = self._create_executor()


def _stationary_trajectory() -> TrajectoryArray:
    trajectory = np.zeros((PLANNER_HORIZON, 4), dtype=np.float32)
    trajectory[:, 2] = 1.0
    return trajectory
