"""Shared single-process MetaDrive environment lifecycle."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from eco_planner.envs.array_types import SingleObservation, TrajectoryArray
from eco_planner.envs.metadrive.execution import TrajectoryExecutionRecord
from eco_planner.envs.metadrive.observation import (
    MetaDriveObservationAdapter,
    NoTrafficMetaDriveObservationAdapter,
    TrafficObservationAudit,
)
from eco_planner.envs.metadrive.reward import RewardProfileConfig
from eco_planner.envs.metadrive.simulator import TrajectoryMetaDriveEnv
from eco_planner.envs.observation import PlannerObservationSpec
from eco_planner.execution_contracts import PLANNER_FUTURE_STEPS

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

    observation: SingleObservation
    traffic_audit: TrafficObservationAudit | None


@dataclass(frozen=True, slots=True)
class EnvSlotStep:
    """One completed trajectory-prefix transition."""

    reward: float
    terminated: bool
    truncated: bool
    execution: TrajectoryExecutionRecord


class MetaDriveEnvSlot:
    """Own one MetaDrive environment, observation adapter, and traffic history."""

    def __init__(
        self,
        env_config: Mapping[str, Any],
        *,
        mode: ObservationMode,
        observation_spec: PlannerObservationSpec,
        map_query_radius_m: float,
        history_warmup_steps: int,
        reward_profile: RewardProfileConfig | None = None,
    ) -> None:
        if mode not in {"traffic", "no_traffic"}:
            raise ValueError("mode must be either 'traffic' or 'no_traffic'")
        if not isinstance(observation_spec, PlannerObservationSpec):
            raise TypeError("observation_spec must be a PlannerObservationSpec")
        if type(map_query_radius_m) not in {int, float} or map_query_radius_m <= 0.0:
            raise ValueError("map_query_radius_m must be a positive real scalar")
        expected_warmup = observation_spec.time_len - 1 if mode == "traffic" else 0
        if history_warmup_steps != expected_warmup:
            raise ValueError(f"{mode} environments require history_warmup_steps={expected_warmup}")

        self._env_config = dict(env_config)
        self._mode = mode
        self._observation_spec = observation_spec
        self._map_query_radius_m = float(map_query_radius_m)
        self._history_warmup_steps = history_warmup_steps
        self._reward_profile = reward_profile
        self._adapter = self._create_adapter()
        self._env = self._create_environment()

    @property
    def env(self) -> TrajectoryMetaDriveEnv:
        """Return the owned low-level environment for rendering and audits."""

        return self._env

    @property
    def vehicle_state(self) -> np.ndarray:
        """Return the current seven-value world-frame ego state."""

        velocity = np.asarray(self._env.agent.velocity, dtype=np.float64)
        return np.array(
            [
                *np.asarray(self._env.agent.position, dtype=np.float64),
                float(self._env.agent.heading_theta),
                *velocity,
                float(self._env.agent.speed),
                0.0,
            ],
            dtype=np.float64,
        )

    def reset(self, *, map_name: str, seed: int) -> EnvSlotReset:
        """Reset this slot for one map and seed before traffic warmup."""

        if map_name != self._env.config["map"]:
            self._replace_environment(map_name)
        self._env.reset(seed=seed)
        if isinstance(self._adapter, MetaDriveObservationAdapter):
            self._adapter.reset(self._env, self._env.initial_traffic_frame)
        else:
            self._adapter.reset(self._env)
        return EnvSlotReset(
            route_completion=self._env.route_completion,
            route_length_m=self._env.route_length_m,
            warmup_initial_state=self.vehicle_state,
            programmatic_lane_speed_limit_audit=self._env.programmatic_lane_speed_limit_audit,
        )

    def warmup(self) -> Iterator[TrajectoryExecutionRecord]:
        """Advance stationary ego while yielding every completed warmup execution."""

        if self._mode == "no_traffic":
            return
        initial_position = self.vehicle_state[:2].copy()
        collected = 0
        while collected < self._history_warmup_steps:
            result = self.step(_stationary_trajectory())
            execution = result.execution
            yield execution
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

        if isinstance(self._adapter, MetaDriveObservationAdapter):
            observation, audit = self._adapter.build(self._env)
            return EnvSlotObservation(observation, audit)
        return EnvSlotObservation(self._adapter.build(self._env), None)

    def step(self, trajectory: TrajectoryArray) -> EnvSlotStep:
        """Execute one trajectory prefix and commit its traffic frames."""

        _, reward, terminated, truncated, info = self._env.step(trajectory)
        execution = info["trajectory_execution"]
        if not isinstance(execution, TrajectoryExecutionRecord):
            raise RuntimeError("TrajectoryMetaDriveEnv did not return a TrajectoryExecutionRecord")
        if isinstance(self._adapter, MetaDriveObservationAdapter):
            self._adapter.append_frames(execution.traffic_frames)
        return EnvSlotStep(
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            execution=execution,
        )

    def close(self) -> None:
        self._env.close()

    def __enter__(self) -> MetaDriveEnvSlot:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _create_adapter(
        self,
    ) -> MetaDriveObservationAdapter | NoTrafficMetaDriveObservationAdapter:
        if self._mode == "traffic":
            return MetaDriveObservationAdapter(self._observation_spec, self._map_query_radius_m)
        return NoTrafficMetaDriveObservationAdapter(
            self._observation_spec, self._map_query_radius_m
        )

    def _create_environment(self) -> TrajectoryMetaDriveEnv:
        if self._reward_profile is None:
            return TrajectoryMetaDriveEnv(self._env_config)
        return TrajectoryMetaDriveEnv(self._env_config, reward_profile=self._reward_profile)

    def _replace_environment(self, map_name: str) -> None:
        self._env.close()
        self._env_config["map"] = map_name
        self._adapter = self._create_adapter()
        self._env = self._create_environment()


def _stationary_trajectory() -> TrajectoryArray:
    trajectory = np.zeros((PLANNER_FUTURE_STEPS, 4), dtype=np.float32)
    trajectory[:, 2] = 1.0
    return trajectory
