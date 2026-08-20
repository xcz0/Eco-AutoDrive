"""Trajectory-level MetaDrive environment lifecycle and step orchestration."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from metadrive.constants import DEFAULT_AGENT, TerminationState
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.obs.observation_base import BaseObservation
from metadrive.utils import Config, concat_step_infos, merge_dicts

from eco_planner.envs.execution import (
    TRAJECTORY_HORIZON,
    TRAJECTORY_TIMESTEP_S,
    KinematicTrajectoryPolicy,
    TrajectoryExecutionRecorder,
    WorldTrajectory,
    execution_steps_from_config,
    finite_info_scalar,
    to_world_trajectory,
    validate_trajectory,
)
from eco_planner.envs.lane_speed import (
    MAX_LANE_SPEED_LIMIT_KMH,
    PROGRAMMATIC_SPEED_LIMIT_SENTINEL_KMH,
    is_real_scalar,
    validated_programmatic_speed_limit_kmh,
)
from eco_planner.envs.traffic_state import TrafficFrame, capture_traffic_frame

_PLANNER_ONLY_OBSERVATION = np.zeros(1, dtype=np.float32)
_ZERO_VELOCITY = np.zeros(2, dtype=np.float64)


class PlannerOnlyObservation(BaseObservation):
    """Minimal Gym observation for an environment observed through project adapters."""

    @property
    def observation_space(self) -> gym.spaces.Box:
        return gym.spaces.Box(0.0, 0.0, shape=(1,), dtype=np.float32)

    def observe(self, *args: Any, **kwargs: Any) -> np.ndarray:
        return _PLANNER_ONLY_OBSERVATION.copy()


class TrajectoryMetaDriveEnv(MetaDriveEnv):
    """Single-agent MetaDrive environment with an explicit 0.1 s or 0.5 s trajectory prefix."""

    @classmethod
    def default_config(cls) -> Config:
        config = super().default_config()
        config.update(
            {
                "trajectory_horizon": None,
                "trajectory_execution_steps": None,
                "programmatic_lane_speed_limit_kmh": None,
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
            "programmatic_lane_speed_limit_kmh",
        }
        missing = sorted(required - set(config))
        if missing:
            raise ValueError(f"TrajectoryMetaDriveEnv configuration is missing: {missing}")
        if config.get("image_observation", False) is not False:
            raise ValueError("TrajectoryMetaDriveEnv requires image_observation=false")
        if config.get("agent_observation") is not None:
            raise ValueError("TrajectoryMetaDriveEnv does not accept a custom agent_observation")
        supplied_policy = config.get("agent_policy")
        if supplied_policy is not None and supplied_policy is not KinematicTrajectoryPolicy:
            raise ValueError("agent_policy must be KinematicTrajectoryPolicy")
        configured = dict(config)
        configured["agent_policy"] = KinematicTrajectoryPolicy
        configured["agent_observation"] = PlannerOnlyObservation
        super().__init__(configured)
        if self.config["is_multi_agent"]:
            raise ValueError("TrajectoryMetaDriveEnv supports only single-agent operation")
        if self.config["manual_control"]:
            raise ValueError("TrajectoryMetaDriveEnv does not support manual control")
        self._execution_steps = execution_steps_from_config(self.config)
        self._programmatic_lane_speed_limit_kmh = validated_programmatic_speed_limit_kmh(
            self.config["programmatic_lane_speed_limit_kmh"]
        )
        self._programmatic_lane_speed_limit_audit: dict[str, object] | None = None
        self._programmatic_sentinel_lane_ids: frozenset[str] | None = None
        self._programmatic_sentinel_map: object | None = None
        self._initial_traffic_frame: TrafficFrame | None = None

    def _post_process_config(self, config: Config) -> Config:
        processed = super()._post_process_config(config)
        if not processed["use_render"]:
            processed["sensors"] = (
                {"lidar": processed["sensors"]["lidar"]}
                if float(processed["traffic_density"]) > 0.0
                else {}
            )
        return processed

    @property
    def programmatic_lane_speed_limit_audit(self) -> dict[str, object]:
        """Return the verified lane-speed metadata written during the latest reset."""

        if self._programmatic_lane_speed_limit_audit is None:
            raise RuntimeError("programmatic lane speed limits are unavailable before reset")
        return dict(self._programmatic_lane_speed_limit_audit)

    @property
    def initial_traffic_frame(self) -> TrafficFrame:
        """Return the traffic snapshot captured immediately after the latest reset."""

        if self._initial_traffic_frame is None:
            raise RuntimeError("initial traffic frame is unavailable before reset")
        return self._initial_traffic_frame

    @property
    def route_completion(self) -> float:
        """Return the current finite route-completion fraction."""

        value = float(self.agent.navigation.route_completion)
        if not np.isfinite(value):
            raise RuntimeError("MetaDrive returned non-finite route completion")
        return value

    def reset(self, *args: Any, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        self._initial_traffic_frame = None
        self._programmatic_lane_speed_limit_audit = None
        result = super().reset(*args, **kwargs)
        self._configure_programmatic_lane_speed_limits()
        self._initial_traffic_frame = capture_traffic_frame(self)
        return result

    def _configure_programmatic_lane_speed_limits(self) -> None:
        """Replace only PGMap's explicit 1000 km/h unset-speed sentinel after map creation."""

        current_map = self.current_map
        if current_map is None:
            raise RuntimeError("MetaDrive did not create a current map during reset")
        road_network = getattr(current_map, "road_network", None)
        if road_network is None or not hasattr(road_network, "get_all_lanes"):
            raise RuntimeError("current map does not expose a lane road network")
        lanes = road_network.get_all_lanes()
        if not isinstance(lanes, list) or not lanes:
            raise RuntimeError("current map road network has no lanes")

        configured_kmh = self._programmatic_lane_speed_limit_kmh
        lane_ids = [repr(getattr(lane, "index", None)) for lane in lanes]
        if len(set(lane_ids)) != len(lane_ids):
            raise RuntimeError("current map exposes duplicate lane identifiers")
        if current_map is not self._programmatic_sentinel_map:
            self._programmatic_sentinel_lane_ids = frozenset(
                repr(getattr(lane, "index", None))
                for lane in lanes
                if getattr(lane, "speed_limit", None) == PROGRAMMATIC_SPEED_LIMIT_SENTINEL_KMH
            )
            self._programmatic_sentinel_map = current_map
        if self._programmatic_sentinel_lane_ids is None:
            raise RuntimeError("programmatic sentinel state was not initialized")
        replaced_lane_ids: set[str] = set()
        preserved_speed_limits_kmh: dict[str, float] = {}
        for lane in lanes:
            lane_id = repr(getattr(lane, "index", None))
            speed_limit = getattr(lane, "speed_limit", None)
            if not is_real_scalar(speed_limit) or not np.isfinite(speed_limit):
                raise ValueError(f"lane {lane_id} speed limit must be a finite numeric km/h value")
            original_kmh = float(speed_limit)
            if lane_id in self._programmatic_sentinel_lane_ids:
                if original_kmh not in {
                    configured_kmh,
                    PROGRAMMATIC_SPEED_LIMIT_SENTINEL_KMH,
                }:
                    raise RuntimeError(
                        f"lane {lane_id} no longer has the configured programmatic speed limit"
                    )
                setter = getattr(lane, "set_speed_limit", None)
                if not callable(setter):
                    raise RuntimeError(f"lane {lane_id} cannot set its programmatic speed limit")
                setter(configured_kmh)
                replaced_lane_ids.add(lane_id)
            elif 0.0 < original_kmh <= MAX_LANE_SPEED_LIMIT_KMH:
                preserved_speed_limits_kmh[lane_id] = original_kmh
            else:
                raise ValueError(
                    f"lane {lane_id} speed limit {original_kmh} km/h is neither the "
                    "programmatic unset sentinel nor a legal explicit speed limit"
                )

        final_speed_limits_kmh: list[float] = []
        for lane in lanes:
            lane_id = repr(getattr(lane, "index", None))
            speed_limit = getattr(lane, "speed_limit", None)
            if not is_real_scalar(speed_limit) or not np.isfinite(speed_limit):
                raise RuntimeError(f"lane {lane_id} returned an invalid configured speed limit")
            final_kmh = float(speed_limit)
            if final_kmh == PROGRAMMATIC_SPEED_LIMIT_SENTINEL_KMH:
                raise RuntimeError(f"lane {lane_id} retained the programmatic speed-limit sentinel")
            if lane_id in replaced_lane_ids and final_kmh != configured_kmh:
                raise RuntimeError(f"lane {lane_id} did not retain the configured speed limit")
            if lane_id in preserved_speed_limits_kmh:
                if final_kmh != preserved_speed_limits_kmh[lane_id]:
                    raise RuntimeError(
                        f"lane {lane_id} explicit speed limit was unexpectedly modified"
                    )
            final_speed_limits_kmh.append(final_kmh)

        counts: dict[str, int] = {}
        for speed_limit_kmh in final_speed_limits_kmh:
            label = f"{speed_limit_kmh:g}"
            counts[label] = counts.get(label, 0) + 1
        self._programmatic_lane_speed_limit_audit = {
            "speed_limit_sentinel_replaced_count": len(replaced_lane_ids),
            "speed_limit_existing_preserved_count": len(preserved_speed_limits_kmh),
            "configured_programmatic_lane_speed_limit_kmh": configured_kmh,
            "lane_speed_limit_kmh_counts": dict(
                sorted(counts.items(), key=lambda item: float(item[0]))
            ),
        }

    def step(self, trajectory: np.ndarray) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        validated = validate_trajectory(trajectory, TRAJECTORY_HORIZON)
        rear_wheelbase = self.agent.REAR_WHEELBASE
        if rear_wheelbase is None:
            raise RuntimeError("controlled vehicle does not define REAR_WHEELBASE")
        world_trajectory = to_world_trajectory(
            validated,
            center_position=np.asarray(self.agent.position, dtype=np.float64),
            center_heading=float(self.agent.heading_theta),
            rear_wheelbase=float(rear_wheelbase),
            timestep_s=TRAJECTORY_TIMESTEP_S,
        )
        total_reward = 0.0
        step_record = TrajectoryExecutionRecorder.empty(self._execution_steps)
        for index in range(self._execution_steps):
            action = world_trajectory if index == 0 else None
            # KinematicTrajectoryPolicy applies the exact waypoint in MetaDrive's after_step
            # phase. Prevent the previous waypoint's velocity from moving the vehicle during
            # this intervening physics phase, including when a new trajectory is supplied.
            self.agent.set_velocity(_ZERO_VELOCITY)
            self.agent.set_angular_velocity(0.0)
            observation, reward, terminated, truncated, info = self._step_once(action)
            total_reward += float(reward)
            step_record.append(
                self.agent,
                float(reward),
                finite_info_scalar(info, "step_reward"),
                terminated,
                truncated,
                float(world_trajectory.angular_velocities[index]),
                capture_traffic_frame(self),
            )
            if terminated or truncated:
                break
        result_info = step_record.update_info(info, world_trajectory, total_reward)
        return observation, total_reward, terminated, truncated, result_info

    def _step_once(
        self, action: WorldTrajectory | None
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        actions = {DEFAULT_AGENT: action}
        engine_info = self._step_planner_simulator(actions)
        agent_id, agent = next(iter(self.agents.items()))
        self.episode_lengths[agent_id] += 1
        reward, reward_info = self.reward_function(agent_id)
        self.episode_rewards[agent_id] += reward
        done, done_info = self.done_function(agent_id)
        _, cost_info = self.cost_function(agent_id)
        self.dones[agent_id] = done or self.dones[agent_id]
        info = concat_step_infos(
            [engine_info, {agent_id: done_info}, {agent_id: reward_info}, {agent_id: cost_info}]
        )
        agent_info = info.pop(agent_id)
        info.update(agent_info)
        truncated = bool(info.get(TerminationState.MAX_STEP, False))
        terminated = bool(self.dones[agent_id])
        if self.config["horizon"] and self.episode_step > 5 * self.config["horizon"]:
            truncated = True
            if self.config["truncate_as_terminate"]:
                self.dones[agent_id] = terminated = True
        info["episode_reward"] = self.episode_rewards[agent_id]
        info["episode_length"] = self.episode_lengths[agent_id]
        if agent is not self.agent:
            raise RuntimeError("single-agent environment changed its active agent")
        return _PLANNER_ONLY_OBSERVATION.copy(), float(reward), terminated, truncated, info

    def _step_planner_simulator(self, actions: dict[str, WorldTrajectory | None]) -> dict:
        before_info = self.engine.before_step(actions)
        if self.config["_render_mode"] != "none" or self.config["record_episode"]:
            self.engine.step(self.config["decision_repeat"])
        else:
            repeats = self.config["decision_repeat"]
            for _ in range(repeats):
                for name, manager in self.engine.managers.items():
                    if name != "record_manager":
                        manager.step()
            physics_step = float(self.config["physics_world_step_size"])
            self.engine.physics_world.dynamic_world.doPhysics(
                physics_step * repeats,
                repeats,
                physics_step,
            )
        after_info = self.engine.after_step()
        return merge_dicts(after_info, before_info, allow_new_keys=True, without_copy=True)
