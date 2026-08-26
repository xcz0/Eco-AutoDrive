"""MetaDrive callback policy for executing project world trajectories."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from metadrive.policy.base_policy import BasePolicy
from metadrive.policy.replay_policy import ReplayTrafficParticipantPolicy

from eco_planner.envs.contracts import PLANNER_HORIZON, ExecutionMode
from eco_planner.envs.domain.trajectory import WorldTrajectory


class KinematicTrajectoryPolicy(ReplayTrafficParticipantPolicy):
    """Apply a fixed world trajectory inside MetaDrive's callback lifecycle."""

    def __init__(self, obj: Any, seed: int) -> None:
        BasePolicy.__init__(self, control_object=obj, random_seed=seed)
        self._execution_steps = ExecutionMode(self.engine.global_config["execution_mode"]).steps
        self._trajectory: WorldTrajectory | None = None
        self._cache_last_update: int | None = None

    @classmethod
    def get_input_space(cls) -> gym.spaces.Box:
        return gym.spaces.Box(-np.inf, np.inf, shape=(PLANNER_HORIZON, 4), dtype=np.float32)

    def reset(self) -> None:
        self._trajectory = None
        self._cache_last_update = None
        super().reset()

    def act(self, agent_id: str) -> None:
        actions = self.engine.external_actions
        if actions is None:
            if self._trajectory is None:
                return None
            raise RuntimeError(
                "trajectory cache survived MetaDrive reset without an external action"
            )
        if agent_id not in actions:
            raise RuntimeError(f"MetaDrive did not provide an external action for {agent_id!r}")
        action = actions[agent_id]
        if action is not None:
            if self._trajectory is not None:
                raise RuntimeError(
                    "a new trajectory was supplied before the cached prefix finished"
                )
            self._trajectory = action
            self._cache_last_update = self.engine.episode_step
        elif self._trajectory is None or self._cache_last_update is None:
            if self.engine.episode_step == 0:
                return None
            raise RuntimeError("trajectory continuation requested without a cached trajectory")
        assert self._trajectory is not None and self._cache_last_update is not None
        index = self.engine.episode_step - self._cache_last_update
        if not 0 <= index < self._execution_steps:
            raise RuntimeError("trajectory cache index is outside the execution prefix")
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
