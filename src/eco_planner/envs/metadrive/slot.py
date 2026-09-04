"""Atomic single-process MetaDrive environment lifecycle."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Literal

import numpy as np
from tensordict import TensorDictBase

from eco_planner.contracts import (
    PLANNER_HORIZON,
    TRAFFIC_HISTORY_WARMUP_STEPS,
    ExecutionMode,
)

from ..domain.arrays import TrajectoryArray
from ..domain.energy import MetaDriveFuelProxyProvider
from ..domain.execution import TrajectoryExecutionResult
from ..observation.scene import TrafficObservationAudit
from .execution import TrajectoryExecutor
from .observation import (
    MetaDriveObservationPipeline,
    NoTrafficMetaDriveObservationPipeline,
    TrafficMetaDriveObservationPipeline,
)
from .simulator import MetaDriveBackend

ObservationMode = Literal["traffic", "no_traffic"]


@dataclass(frozen=True, slots=True)
class EnvSlotState:
    """Planner-visible state captured at one simulator decision boundary."""

    observation: TensorDictBase
    traffic_audit: TrafficObservationAudit | None
    vehicle_state: np.ndarray
    route_completion: float


@dataclass(frozen=True, slots=True)
class EnvSlotTiming:
    """Simulator and observation service time for one atomic operation."""

    environment_s: float
    observation_s: float


@dataclass(frozen=True, slots=True)
class EnvSlotReset:
    """Complete reset, warmup, initial observation, and audit result."""

    state: EnvSlotState
    route_length_m: float
    warmup_initial_state: np.ndarray
    warmup_steps: tuple[TrajectoryExecutionResult, ...]
    programmatic_lane_speed_limit_audit: Mapping[str, object]
    timing: EnvSlotTiming


@dataclass(frozen=True, slots=True)
class EnvSlotStep:
    """One trajectory execution and the next planner-visible state."""

    state: EnvSlotState
    execution: TrajectoryExecutionResult
    timing: EnvSlotTiming


class MetaDriveEnvSlot:
    """Compose one simulator, trajectory executor, and observation pipeline."""

    def __init__(
        self,
        env_config: Mapping[str, Any],
        *,
        mode: ObservationMode,
        execution_mode: ExecutionMode,
        map_query_radius_m: float,
        history_warmup_steps: int,
    ) -> None:
        if mode not in {"traffic", "no_traffic"}:
            raise ValueError("mode must be either 'traffic' or 'no_traffic'")
        if not isinstance(execution_mode, ExecutionMode):
            raise TypeError("execution_mode must be an ExecutionMode")
        if type(map_query_radius_m) not in {int, float} or map_query_radius_m <= 0.0:
            raise ValueError("map_query_radius_m must be a positive real scalar")
        expected_warmup = TRAFFIC_HISTORY_WARMUP_STEPS if mode == "traffic" else 0
        if history_warmup_steps != expected_warmup:
            raise ValueError(f"{mode} environments require history_warmup_steps={expected_warmup}")

        self._env_config = dict(env_config)
        self._execution_steps = execution_mode.steps
        self._mode = mode
        self._map_query_radius_m = float(map_query_radius_m)
        self._history_warmup_steps = history_warmup_steps
        self._observation_pipeline = self._create_observation_pipeline()
        self._backend = self._create_backend()
        self._executor = self._create_executor()

    @property
    def backend(self) -> MetaDriveBackend:
        """Return the owned simulator backend for rendering and native audits."""

        return self._backend

    def reset(self, *, map_name: str, seed: int) -> EnvSlotReset:
        """Reset, warm up, and return the first complete planner state."""

        environment_started = perf_counter()
        if map_name != self._backend.config["map"]:
            self._replace_environment(map_name)
        self._backend.reset(seed=seed)
        warmup_initial_state = self._vehicle_state()
        initial_traffic_frame = self._executor.reset()
        self._observation_pipeline.reset(self._backend, initial_traffic_frame)
        warmup_steps = tuple(self._warmup())
        environment_s = perf_counter() - environment_started

        observation_started = perf_counter()
        state = self._state()
        observation_s = perf_counter() - observation_started
        return EnvSlotReset(
            state=state,
            route_length_m=self._backend.route_length_m,
            warmup_initial_state=warmup_initial_state,
            warmup_steps=warmup_steps,
            programmatic_lane_speed_limit_audit=self._backend.programmatic_lane_speed_limit_audit,
            timing=EnvSlotTiming(environment_s, observation_s),
        )

    def step(self, trajectory: TrajectoryArray) -> EnvSlotStep:
        """Execute one trajectory prefix and return the next complete planner state."""

        environment_started = perf_counter()
        execution = self._execute(trajectory)
        environment_s = perf_counter() - environment_started
        observation_started = perf_counter()
        state = self._state()
        observation_s = perf_counter() - observation_started
        return EnvSlotStep(
            state=state,
            execution=execution,
            timing=EnvSlotTiming(environment_s, observation_s),
        )

    def close(self) -> None:
        self._backend.close()

    def __enter__(self) -> MetaDriveEnvSlot:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def recreate_environment(self) -> None:
        """Recreate the environment after MetaDrive navigation corruption."""

        self._replace_environment(self._env_config["map"])

    def _warmup(self) -> Iterator[TrajectoryExecutionResult]:
        if self._mode == "no_traffic":
            return
        initial_position = self._vehicle_state()[:2]
        collected = 0
        while collected < self._history_warmup_steps:
            result = self._execute(_stationary_trajectory())
            yield result
            if result.terminated or result.truncated:
                raise RuntimeError("traffic history warmup ended before the required frame count")
            if (
                float(
                    np.linalg.norm(
                        result.execution.substep_states[:, :2] - initial_position,
                        axis=1,
                    ).max()
                )
                >= 1e-3
            ):
                raise RuntimeError("ego moved during stationary traffic history warmup")
            collected += result.execution.substep_states.shape[0]
        if collected != self._history_warmup_steps:
            raise RuntimeError("traffic history warmup overshot the required frame count")

    def _execute(self, trajectory: TrajectoryArray) -> TrajectoryExecutionResult:
        result = self._executor.execute(trajectory)
        self._observation_pipeline.append_frames(result.execution.traffic_frames)
        return result

    def _state(self) -> EnvSlotState:
        observation, traffic_audit = self._observation_pipeline.build(self._backend)
        return EnvSlotState(
            observation=observation,
            traffic_audit=traffic_audit,
            vehicle_state=self._vehicle_state(),
            route_completion=self._backend.route_completion,
        )

    def _vehicle_state(self) -> np.ndarray:
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

    def _create_observation_pipeline(self) -> MetaDriveObservationPipeline:
        if self._mode == "traffic":
            return TrafficMetaDriveObservationPipeline(self._map_query_radius_m)
        return NoTrafficMetaDriveObservationPipeline(self._map_query_radius_m)

    def _create_backend(self) -> MetaDriveBackend:
        return MetaDriveBackend(dict(self._env_config))

    def _create_executor(self) -> TrajectoryExecutor:
        return TrajectoryExecutor(
            self._backend,
            self._execution_steps,
            MetaDriveFuelProxyProvider(),
        )

    def _replace_environment(self, map_name: str) -> None:
        self._backend.close()
        self._env_config["map"] = map_name
        self._observation_pipeline = self._create_observation_pipeline()
        self._backend = self._create_backend()
        self._executor = self._create_executor()


def _stationary_trajectory() -> TrajectoryArray:
    trajectory = np.zeros((PLANNER_HORIZON, 4), dtype=np.float32)
    trajectory[:, 2] = 1.0
    return trajectory
