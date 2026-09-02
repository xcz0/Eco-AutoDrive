"""MetaDrive callback policy for executing project world trajectories."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import gymnasium as gym
import numpy as np
from metadrive.policy.base_policy import BasePolicy
from metadrive.policy.replay_policy import ReplayTrafficParticipantPolicy

from eco_planner.contracts import PLANNER_HORIZON
from eco_planner.envs.domain.trajectory import WorldTrajectory


class KinematicTrajectoryPolicy(ReplayTrafficParticipantPolicy):
    """Apply a fixed world trajectory inside MetaDrive's callback lifecycle."""

    def __init__(self, obj: Any, seed: int) -> None:
        BasePolicy.__init__(self, control_object=obj, random_seed=seed)
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
        engine = cast(Any, self.engine)
        actions = cast(Mapping[str, WorldTrajectory | None] | None, engine.external_actions)
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
            self._trajectory = action
            self._cache_last_update = engine.episode_step
        elif self._trajectory is None or self._cache_last_update is None:
            if engine.episode_step == 0:
                return None
            raise RuntimeError("trajectory continuation requested without a cached trajectory")
        assert self._trajectory is not None and self._cache_last_update is not None
        index = engine.episode_step - self._cache_last_update
        if not 0 <= index < PLANNER_HORIZON:
            raise RuntimeError("trajectory cache index is outside the planner horizon")
        trajectory = self._trajectory
        control_object = cast(Any, self.control_object)
        control_object.set_position(trajectory.centers[index + 1])
        control_object.set_heading_theta(float(trajectory.headings[index + 1]))
        control_object.set_velocity(trajectory.velocities[index])
        control_object.set_angular_velocity(float(trajectory.angular_velocities[index]))
        self.action_info["trajectory_index"] = index
        self.action_info["trajectory_target_position"] = trajectory.centers[index + 1].copy()
        self.action_info["trajectory_target_heading"] = float(trajectory.headings[index + 1])
        return None
