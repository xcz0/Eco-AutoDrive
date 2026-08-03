"""Trajectory-level MetaDrive environment and kinematic execution policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from metadrive.engine.engine_utils import get_global_config
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.policy.base_policy import BasePolicy
from metadrive.utils import Config

TRAJECTORY_HORIZON = 80
TRAJECTORY_EXECUTION_STEPS = 5
TRAJECTORY_TIMESTEP_S = 0.1
_MIN_HEADING_NORM = 1e-6


@dataclass(frozen=True)
class _WorldTrajectory:
    centers: np.ndarray
    headings: np.ndarray
    velocities: np.ndarray
    angular_velocities: np.ndarray


def _validate_trajectory(trajectory: object, horizon: int) -> np.ndarray:
    if not isinstance(trajectory, np.ndarray):
        raise TypeError("trajectory must be a numpy.ndarray")
    if trajectory.dtype != np.float32:
        raise TypeError("trajectory must use numpy.float32")
    if trajectory.shape != (horizon, 4):
        raise ValueError(f"trajectory must have shape ({horizon}, 4)")
    if not np.isfinite(trajectory).all():
        raise ValueError("trajectory must contain only finite values")
    heading_norms = np.linalg.norm(trajectory[:, 2:4], axis=1)
    if np.any(heading_norms <= _MIN_HEADING_NORM):
        raise ValueError("trajectory heading vectors must be non-zero")
    return trajectory.copy()


def _rear_axle_position(
    center_position: np.ndarray, heading: float, rear_wheelbase: float
) -> np.ndarray:
    direction = np.array([np.cos(heading), np.sin(heading)], dtype=np.float64)
    return center_position - rear_wheelbase * direction


def _to_world_trajectory(
    trajectory: np.ndarray,
    *,
    center_position: np.ndarray,
    center_heading: float,
    rear_wheelbase: float,
    timestep_s: float,
) -> _WorldTrajectory:
    if center_position.shape != (2,) or not np.isfinite(center_position).all():
        raise ValueError("center_position must be a finite two-dimensional vector")
    if not np.isfinite(center_heading):
        raise ValueError("center_heading must be finite")
    if not np.isfinite(rear_wheelbase) or rear_wheelbase <= 0.0:
        raise ValueError("rear_wheelbase must be finite and positive")
    if not np.isfinite(timestep_s) or timestep_s <= 0.0:
        raise ValueError("timestep_s must be finite and positive")

    anchor_rear_axle = _rear_axle_position(
        center_position.astype(np.float64), center_heading, rear_wheelbase
    )
    cosine = np.cos(center_heading)
    sine = np.sin(center_heading)
    rotation = np.array([[cosine, -sine], [sine, cosine]], dtype=np.float64)
    future_rear_axles = trajectory[:, :2].astype(np.float64) @ rotation.T + anchor_rear_axle

    relative_headings = np.arctan2(trajectory[:, 3], trajectory[:, 2]).astype(np.float64)
    future_headings = center_heading + relative_headings
    future_directions = np.column_stack((np.cos(future_headings), np.sin(future_headings)))
    future_centers = future_rear_axles + rear_wheelbase * future_directions

    centers = np.vstack((center_position.astype(np.float64), future_centers))
    headings = np.concatenate(([center_heading], future_headings))
    velocities = np.diff(centers, axis=0) / timestep_s
    heading_deltas = (np.diff(headings) + np.pi) % (2.0 * np.pi) - np.pi
    angular_velocities = heading_deltas / timestep_s
    return _WorldTrajectory(centers, headings, velocities, angular_velocities)


class KinematicTrajectoryPolicy(BasePolicy):
    """Execute an ego-local rear-axle trajectory by directly updating vehicle state."""

    def __init__(self, obj: Any, seed: int) -> None:
        super().__init__(control_object=obj, random_seed=seed)
        config = self.engine.global_config
        self._horizon = _require_positive_int(config, "trajectory_horizon")
        self._execution_steps = _require_positive_int(config, "trajectory_execution_steps")
        self._timestep_s = _validated_timestep(config)
        self._trajectory: _WorldTrajectory | None = None
        self._cache_last_update: int | None = None

    @classmethod
    def get_input_space(cls) -> gym.spaces.Box:
        horizon = _require_positive_int(get_global_config(), "trajectory_horizon")
        return gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(horizon, 4),
            dtype=np.float32,
        )

    def reset(self) -> None:
        self._trajectory = None
        self._cache_last_update = None
        super().reset()

    def act(self, agent_id: str) -> None:
        external_action = self.engine.external_actions[agent_id]
        if external_action is not None:
            if self._trajectory is not None:
                raise RuntimeError(
                    "a new trajectory was supplied before the cached prefix finished"
                )
            trajectory = _validate_trajectory(external_action, self._horizon)
            rear_wheelbase = self.control_object.REAR_WHEELBASE
            if rear_wheelbase is None:
                raise RuntimeError("controlled vehicle does not define REAR_WHEELBASE")
            self._trajectory = _to_world_trajectory(
                trajectory,
                center_position=np.asarray(self.control_object.position, dtype=np.float64),
                center_heading=float(self.control_object.heading_theta),
                rear_wheelbase=float(rear_wheelbase),
                timestep_s=self._timestep_s,
            )
            self._cache_last_update = self.engine.episode_step
        elif self._trajectory is None or self._cache_last_update is None:
            raise RuntimeError("trajectory continuation requested without a cached trajectory")

        if self._trajectory is None or self._cache_last_update is None:
            raise RuntimeError("trajectory cache was not initialized")
        index = self.engine.episode_step - self._cache_last_update
        if index < 0 or index >= self._execution_steps:
            raise RuntimeError(
                f"trajectory cache index {index} is outside execution prefix "
                f"[0, {self._execution_steps})"
            )

        trajectory = self._trajectory
        self.control_object.set_position(trajectory.centers[index])
        self.control_object.set_heading_theta(float(trajectory.headings[index]))
        self.control_object.set_velocity(trajectory.velocities[index])
        self.control_object.set_angular_velocity(float(trajectory.angular_velocities[index]))
        self.action_info["trajectory_index"] = index
        self.action_info["trajectory_target_position"] = trajectory.centers[index + 1].copy()
        self.action_info["trajectory_target_heading"] = float(trajectory.headings[index + 1])

        if index == self._execution_steps - 1:
            self._trajectory = None
            self._cache_last_update = None
        return None


class TrajectoryMetaDriveEnv(MetaDriveEnv):
    """Single-agent MetaDrive environment with a 0.5 s trajectory action."""

    @classmethod
    def default_config(cls) -> Config:
        config = super().default_config()
        config.update(
            {
                "trajectory_horizon": None,
                "trajectory_execution_steps": None,
            },
            allow_add_new_key=True,
        )
        return config

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        if config is None:
            raise ValueError("TrajectoryMetaDriveEnv requires an explicit configuration")
        required = {
            "physics_world_step_size",
            "decision_repeat",
            "trajectory_horizon",
            "trajectory_execution_steps",
        }
        missing = sorted(required - set(config))
        if missing:
            raise ValueError(f"TrajectoryMetaDriveEnv configuration is missing: {missing}")
        supplied_policy = config.get("agent_policy")
        if supplied_policy is not None and supplied_policy is not KinematicTrajectoryPolicy:
            raise ValueError("agent_policy must be KinematicTrajectoryPolicy")
        configured = dict(config)
        configured["agent_policy"] = KinematicTrajectoryPolicy
        super().__init__(configured)
        if self.config["is_multi_agent"]:
            raise ValueError("TrajectoryMetaDriveEnv supports only single-agent operation")
        if self.config["manual_control"]:
            raise ValueError("TrajectoryMetaDriveEnv does not support manual control")
        horizon = _require_positive_int(self.config, "trajectory_horizon")
        execution_steps = _require_positive_int(self.config, "trajectory_execution_steps")
        if horizon != TRAJECTORY_HORIZON:
            raise ValueError(f"trajectory_horizon must be {TRAJECTORY_HORIZON}")
        if execution_steps != TRAJECTORY_EXECUTION_STEPS:
            raise ValueError(f"trajectory_execution_steps must be {TRAJECTORY_EXECUTION_STEPS}")
        _validated_timestep(self.config)

    def step(self, trajectory: np.ndarray) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        validated = _validate_trajectory(trajectory, TRAJECTORY_HORIZON)
        rear_wheelbase = self.agent.REAR_WHEELBASE
        if rear_wheelbase is None:
            raise RuntimeError("controlled vehicle does not define REAR_WHEELBASE")
        world_trajectory = _to_world_trajectory(
            validated,
            center_position=np.asarray(self.agent.position, dtype=np.float64),
            center_heading=float(self.agent.heading_theta),
            rear_wheelbase=float(rear_wheelbase),
            timestep_s=TRAJECTORY_TIMESTEP_S,
        )
        total_reward = 0.0
        final_result: tuple[Any, float, bool, bool, dict[str, Any]] | None = None
        executed_steps = 0
        substep_states: list[np.ndarray] = []
        substep_rewards: list[float] = []
        substep_terminated: list[bool] = []
        substep_truncated: list[bool] = []
        for index in range(TRAJECTORY_EXECUTION_STEPS):
            action = validated if index == 0 else None
            observation, reward, terminated, truncated, info = super().step(action)
            total_reward += float(reward)
            executed_steps += 1
            velocity = np.asarray(self.agent.velocity, dtype=np.float64)
            if velocity.shape != (2,) or not np.isfinite(velocity).all():
                raise RuntimeError("controlled vehicle returned an invalid velocity")
            substep_states.append(
                np.array(
                    [
                        *np.asarray(self.agent.position, dtype=np.float64),
                        float(self.agent.heading_theta),
                        *velocity,
                        float(self.agent.speed),
                        float(world_trajectory.angular_velocities[index]),
                    ],
                    dtype=np.float64,
                )
            )
            substep_rewards.append(float(reward))
            substep_terminated.append(bool(terminated))
            substep_truncated.append(bool(truncated))
            final_result = (observation, total_reward, terminated, truncated, info)
            if terminated or truncated:
                break
        if final_result is None:
            raise RuntimeError("trajectory execution completed without a simulator step")
        observation, _, terminated, truncated, final_info = final_result
        result_info = dict(final_info)
        result_info["trajectory_execution_steps"] = executed_steps
        result_info["trajectory_reward_sum"] = total_reward
        result_info["trajectory_world_centers"] = world_trajectory.centers[1:].copy()
        result_info["trajectory_world_headings"] = world_trajectory.headings[1:].copy()
        result_info["trajectory_substep_states"] = np.stack(substep_states)
        result_info["trajectory_substep_rewards"] = np.asarray(substep_rewards, dtype=np.float64)
        result_info["trajectory_substep_terminated"] = np.asarray(
            substep_terminated, dtype=np.bool_
        )
        result_info["trajectory_substep_truncated"] = np.asarray(substep_truncated, dtype=np.bool_)
        return observation, total_reward, terminated, truncated, result_info


def _require_positive_int(config: Any, name: str) -> int:
    value = config[name]
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validated_timestep(config: Any) -> float:
    physics_step = config["physics_world_step_size"]
    decision_repeat = config["decision_repeat"]
    if type(physics_step) not in {int, float} or physics_step <= 0.0:
        raise ValueError("physics_world_step_size must be positive")
    if type(decision_repeat) is not int or decision_repeat <= 0:
        raise ValueError("decision_repeat must be a positive integer")
    timestep = float(physics_step) * decision_repeat
    if not np.isclose(timestep, TRAJECTORY_TIMESTEP_S, rtol=0.0, atol=1e-12):
        raise ValueError(
            f"physics_world_step_size * decision_repeat must equal {TRAJECTORY_TIMESTEP_S} seconds"
        )
    return timestep
