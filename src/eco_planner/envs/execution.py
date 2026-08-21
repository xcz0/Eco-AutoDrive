"""Trajectory validation, kinematic execution, and typed MetaDrive results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from metadrive.policy.base_policy import BasePolicy
from metadrive.policy.replay_policy import ReplayTrafficParticipantPolicy

from eco_planner.envs.geometry import (
    local_points_to_world,
    rear_axle_position,
    shortest_angle_delta,
)
from eco_planner.envs.traffic_state import TrafficFrame
from eco_planner.envs.validation import is_real_scalar

TRAJECTORY_HORIZON = 80
TRAJECTORY_EXECUTION_STEPS = 5
ROLLOUT_EXECUTION_STEPS = 1
TRAJECTORY_TIMESTEP_S = 0.1

_ALLOWED_EXECUTION_STEPS = frozenset({ROLLOUT_EXECUTION_STEPS, TRAJECTORY_EXECUTION_STEPS})
_MIN_HEADING_NORM = 1e-6


@dataclass(frozen=True, slots=True)
class TrajectoryExecutionRecord:
    start_center: np.ndarray
    start_heading: float
    world_centers: np.ndarray
    world_headings: np.ndarray
    substep_states: np.ndarray
    target_centers: np.ndarray
    target_headings: np.ndarray
    position_errors_m: np.ndarray
    heading_errors_rad: np.ndarray
    substep_rewards: np.ndarray
    substep_dense_rewards: np.ndarray
    substep_terminated: np.ndarray
    substep_truncated: np.ndarray
    traffic_frames: tuple[TrafficFrame, ...]
    route_completion: float
    arrive_dest: bool
    out_of_road: bool
    crash_vehicle: bool
    crash_object: bool
    crash_building: bool
    crash_human: bool
    max_step: bool


@dataclass(frozen=True, slots=True)
class WorldTrajectory:
    centers: np.ndarray
    headings: np.ndarray
    velocities: np.ndarray
    angular_velocities: np.ndarray


@dataclass(slots=True)
class TrajectoryExecutionRecorder:
    states: np.ndarray
    rewards: np.ndarray
    dense_rewards: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    traffic_frames: list[TrafficFrame]
    count: int

    @classmethod
    def empty(cls, execution_steps: int) -> TrajectoryExecutionRecorder:
        return cls(
            states=np.empty((execution_steps, 7), dtype=np.float64),
            rewards=np.empty(execution_steps, dtype=np.float64),
            dense_rewards=np.empty(execution_steps, dtype=np.float64),
            terminated=np.empty(execution_steps, dtype=np.bool_),
            truncated=np.empty(execution_steps, dtype=np.bool_),
            traffic_frames=[],
            count=0,
        )

    def append(
        self,
        agent: Any,
        reward: float,
        dense_reward: float,
        terminated: bool,
        truncated: bool,
        angular_velocity: float,
        traffic_frame: TrafficFrame,
    ) -> None:
        index = self.count
        self.states[index, :2] = np.asarray(agent.position, dtype=np.float64)
        self.states[index, 2] = float(agent.heading_theta)
        self.states[index, 3:5] = agent.velocity
        self.states[index, 5] = float(agent.speed)
        self.states[index, 6] = angular_velocity
        self.rewards[index] = reward
        self.dense_rewards[index] = dense_reward
        self.terminated[index] = terminated
        self.truncated[index] = truncated
        self.traffic_frames.append(traffic_frame)
        self.count += 1

    def update_info(
        self,
        final_info: dict[str, Any],
        world_trajectory: WorldTrajectory,
        total_reward: float,
    ) -> dict[str, Any]:
        executed_steps = self.count
        state_array = self.states[:executed_steps].copy()
        target_centers = world_trajectory.centers[1 : executed_steps + 1]
        target_headings = world_trajectory.headings[1 : executed_steps + 1]
        execution = TrajectoryExecutionRecord(
            start_center=world_trajectory.centers[0].copy(),
            start_heading=float(world_trajectory.headings[0]),
            world_centers=world_trajectory.centers[1:].copy(),
            world_headings=world_trajectory.headings[1:].copy(),
            substep_states=state_array,
            target_centers=target_centers.copy(),
            target_headings=target_headings.copy(),
            position_errors_m=np.linalg.norm(state_array[:, :2] - target_centers, axis=1),
            heading_errors_rad=np.abs(shortest_angle_delta(state_array[:, 2] - target_headings)),
            substep_rewards=self.rewards[:executed_steps].copy(),
            substep_dense_rewards=self.dense_rewards[:executed_steps].copy(),
            substep_terminated=self.terminated[:executed_steps].copy(),
            substep_truncated=self.truncated[:executed_steps].copy(),
            traffic_frames=tuple(self.traffic_frames),
            route_completion=finite_info_scalar(final_info, "route_completion"),
            arrive_dest=bool(final_info["arrive_dest"]),
            out_of_road=bool(final_info["out_of_road"]),
            crash_vehicle=bool(final_info["crash_vehicle"]),
            crash_object=bool(final_info["crash_object"]),
            crash_building=bool(final_info["crash_building"]),
            crash_human=bool(final_info["crash_human"]),
            max_step=bool(final_info["max_step"]),
        )
        result = dict(final_info)
        result["trajectory_execution_steps"] = executed_steps
        result["trajectory_reward_sum"] = total_reward
        result["trajectory_execution"] = execution
        return result


def validate_trajectory(trajectory: object, horizon: int) -> np.ndarray:
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


def to_world_trajectory(
    trajectory: np.ndarray,
    *,
    center_position: np.ndarray,
    center_heading: float,
    rear_wheelbase: float,
    timestep_s: float,
) -> WorldTrajectory:
    anchor_rear_axle = rear_axle_position(
        center_position.astype(np.float64), center_heading, rear_wheelbase
    )
    future_rear_axles = local_points_to_world(
        trajectory[:, :2].astype(np.float64), anchor_rear_axle, center_heading
    )

    relative_headings = np.arctan2(trajectory[:, 3], trajectory[:, 2]).astype(np.float64)
    future_headings = center_heading + relative_headings
    future_directions = np.column_stack((np.cos(future_headings), np.sin(future_headings)))
    future_centers = future_rear_axles + rear_wheelbase * future_directions

    centers = np.vstack((center_position.astype(np.float64), future_centers))
    headings = np.concatenate(([center_heading], future_headings))
    velocities = np.diff(centers, axis=0) / timestep_s
    heading_deltas = shortest_angle_delta(np.diff(headings))
    angular_velocities = heading_deltas / timestep_s
    return WorldTrajectory(centers, headings, velocities, angular_velocities)


class KinematicTrajectoryPolicy(ReplayTrafficParticipantPolicy):
    """Execute an ego-local rear-axle trajectory by directly updating vehicle state."""

    def __init__(self, obj: Any, seed: int) -> None:
        # ReplayTrafficParticipantPolicy marks this policy for MetaDrive's after_step phase. That
        # phase runs after physics integration but before BaseEnv samples observation, reward, and
        # termination, so the externally recorded state is the requested trajectory waypoint.
        BasePolicy.__init__(self, control_object=obj, random_seed=seed)
        self._execution_steps = execution_steps_from_config(self.engine.global_config)
        self._trajectory: WorldTrajectory | None = None
        self._cache_last_update: int | None = None

    @classmethod
    def get_input_space(cls) -> gym.spaces.Box:
        return gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(TRAJECTORY_HORIZON, 4),
            dtype=np.float32,
        )

    def reset(self) -> None:
        self._trajectory = None
        self._cache_last_update = None
        super().reset()

    def act(self, agent_id: str) -> None:
        external_actions = self.engine.external_actions
        if external_actions is None:
            if self._trajectory is None:
                return None
            raise RuntimeError(
                "trajectory cache survived MetaDrive reset without an external action"
            )
        if agent_id not in external_actions:
            raise RuntimeError(f"MetaDrive did not provide an external action for {agent_id!r}")
        external_action = external_actions[agent_id]
        if external_action is not None:
            if self._trajectory is not None:
                raise RuntimeError(
                    "a new trajectory was supplied before the cached prefix finished"
                )
            self._trajectory = external_action
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
        self.control_object.set_position(trajectory.centers[index + 1])
        self.control_object.set_heading_theta(float(trajectory.headings[index + 1]))
        self.control_object.set_velocity(trajectory.velocities[index])
        self.control_object.set_angular_velocity(float(trajectory.angular_velocities[index]))
        self.action_info["trajectory_index"] = index
        self.action_info["trajectory_target_position"] = trajectory.centers[index + 1].copy()
        self.action_info["trajectory_target_heading"] = float(trajectory.headings[index + 1])

        if index == self._execution_steps - 1:
            self._trajectory = None
            self._cache_last_update = None
        return None


def execution_steps_from_config(config: Any) -> int:
    horizon = _require_positive_int(config, "trajectory_horizon")
    execution_steps = _require_positive_int(config, "trajectory_execution_steps")
    if horizon != TRAJECTORY_HORIZON:
        raise ValueError(f"trajectory_horizon must be {TRAJECTORY_HORIZON}")
    if execution_steps not in _ALLOWED_EXECUTION_STEPS:
        raise ValueError(
            f"trajectory_execution_steps must be one of {sorted(_ALLOWED_EXECUTION_STEPS)}"
        )
    _validated_timestep(config)
    return execution_steps


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


def finite_info_scalar(info: dict[str, Any], name: str) -> float:
    value = info.get(name)
    if not is_real_scalar(value):
        raise TypeError(f"{name} must be a finite numeric scalar")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result
