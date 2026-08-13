"""Trajectory-level MetaDrive environment and kinematic execution policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from metadrive.constants import DEFAULT_AGENT, TerminationState
from metadrive.engine.engine_utils import get_global_config
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.obs.observation_base import BaseObservation
from metadrive.policy.base_policy import BasePolicy
from metadrive.policy.replay_policy import ReplayTrafficParticipantPolicy
from metadrive.utils import Config, concat_step_infos, merge_dicts

from eco_planner.envs.geometry import (
    local_points_to_world,
    rear_axle_position,
    shortest_angle_delta,
)
from eco_planner.envs.lane_speed import (
    MAX_LANE_SPEED_LIMIT_KMH,
    PROGRAMMATIC_SPEED_LIMIT_SENTINEL_KMH,
    is_real_scalar,
    validated_programmatic_speed_limit_kmh,
)
from eco_planner.envs.traffic_state import TrafficFrame, capture_traffic_frame

TRAJECTORY_HORIZON = 80
TRAJECTORY_EXECUTION_STEPS = 5
TRAJECTORY_TIMESTEP_S = 0.1
_MIN_HEADING_NORM = 1e-6
_PLANNER_ONLY_OBSERVATION = np.zeros(1, dtype=np.float32)
_ZERO_VELOCITY = np.zeros(2, dtype=np.float64)


class PlannerOnlyObservation(BaseObservation):
    """Minimal Gym observation for an environment observed through project adapters."""

    @property
    def observation_space(self) -> gym.spaces.Box:
        return gym.spaces.Box(0.0, 0.0, shape=(1,), dtype=np.float32)

    def observe(self, *args: Any, **kwargs: Any) -> np.ndarray:
        return _PLANNER_ONLY_OBSERVATION.copy()


@dataclass(frozen=True)
class _WorldTrajectory:
    centers: np.ndarray
    headings: np.ndarray
    velocities: np.ndarray
    angular_velocities: np.ndarray


@dataclass(frozen=True)
class _PreparedTrajectory:
    local: np.ndarray
    world: _WorldTrajectory


@dataclass
class _TrajectoryStepRecord:
    states: np.ndarray
    rewards: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    traffic_frames: list[TrafficFrame]
    count: int

    @classmethod
    def empty(cls) -> _TrajectoryStepRecord:
        return cls(
            states=np.empty((TRAJECTORY_EXECUTION_STEPS, 7), dtype=np.float64),
            rewards=np.empty(TRAJECTORY_EXECUTION_STEPS, dtype=np.float64),
            terminated=np.empty(TRAJECTORY_EXECUTION_STEPS, dtype=np.bool_),
            truncated=np.empty(TRAJECTORY_EXECUTION_STEPS, dtype=np.bool_),
            traffic_frames=[],
            count=0,
        )

    def append(
        self,
        agent: Any,
        reward: float,
        terminated: bool,
        truncated: bool,
        angular_velocity: float,
        traffic_frame: TrafficFrame,
    ) -> None:
        velocity = np.asarray(agent.velocity, dtype=np.float64)
        if velocity.shape != (2,) or not np.isfinite(velocity).all():
            raise RuntimeError("controlled vehicle returned an invalid velocity")
        index = self.count
        self.states[index, :2] = np.asarray(agent.position, dtype=np.float64)
        self.states[index, 2] = float(agent.heading_theta)
        self.states[index, 3:5] = velocity
        self.states[index, 5] = float(agent.speed)
        self.states[index, 6] = angular_velocity
        self.rewards[index] = reward
        self.terminated[index] = terminated
        self.truncated[index] = truncated
        self.traffic_frames.append(traffic_frame)
        self.count += 1

    def update_info(
        self,
        final_info: dict[str, Any],
        world_trajectory: _WorldTrajectory,
        total_reward: float,
    ) -> dict[str, Any]:
        executed_steps = self.count
        if executed_steps == 0:
            raise RuntimeError("trajectory execution completed without a simulator step")
        state_array = self.states[:executed_steps].copy()
        target_centers = world_trajectory.centers[1 : executed_steps + 1]
        target_headings = world_trajectory.headings[1 : executed_steps + 1]
        result = dict(final_info)
        result["trajectory_execution_steps"] = executed_steps
        result["trajectory_reward_sum"] = total_reward
        result["trajectory_world_centers"] = world_trajectory.centers[1:].copy()
        result["trajectory_world_headings"] = world_trajectory.headings[1:].copy()
        result["trajectory_substep_states"] = state_array
        result["trajectory_target_centers"] = target_centers.copy()
        result["trajectory_target_headings"] = target_headings.copy()
        result["trajectory_position_errors_m"] = np.linalg.norm(
            state_array[:, :2] - target_centers, axis=1
        )
        result["trajectory_heading_errors_rad"] = np.abs(
            shortest_angle_delta(state_array[:, 2] - target_headings)
        )
        result["trajectory_substep_rewards"] = self.rewards[:executed_steps].copy()
        result["trajectory_substep_terminated"] = self.terminated[:executed_steps].copy()
        result["trajectory_substep_truncated"] = self.truncated[:executed_steps].copy()
        result["traffic_substep_frames"] = tuple(self.traffic_frames)
        return result


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
    return _WorldTrajectory(centers, headings, velocities, angular_velocities)


class KinematicTrajectoryPolicy(ReplayTrafficParticipantPolicy):
    """Execute an ego-local rear-axle trajectory by directly updating vehicle state."""

    def __init__(self, obj: Any, seed: int) -> None:
        # ReplayTrafficParticipantPolicy marks this policy for MetaDrive's after_step phase.  That
        # phase runs after physics integration but before BaseEnv samples observation, reward, and
        # termination, so the externally recorded state is the requested trajectory waypoint.
        BasePolicy.__init__(self, control_object=obj, random_seed=seed)
        config = self.engine.global_config
        self._horizon = _require_positive_int(config, "trajectory_horizon")
        self._execution_steps = _require_positive_int(config, "trajectory_execution_steps")
        _validated_timestep(config)
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
            if not isinstance(external_action, _PreparedTrajectory):
                raise TypeError("trajectory policy requires a prepared trajectory")
            if external_action.local.shape != (self._horizon, 4):
                raise ValueError("prepared trajectory has an invalid horizon")
            self._trajectory = external_action.world
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


class TrajectoryMetaDriveEnv(MetaDriveEnv):
    """Single-agent MetaDrive environment with a 0.5 s trajectory action."""

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
        horizon = _require_positive_int(self.config, "trajectory_horizon")
        execution_steps = _require_positive_int(self.config, "trajectory_execution_steps")
        if horizon != TRAJECTORY_HORIZON:
            raise ValueError(f"trajectory_horizon must be {TRAJECTORY_HORIZON}")
        if execution_steps != TRAJECTORY_EXECUTION_STEPS:
            raise ValueError(f"trajectory_execution_steps must be {TRAJECTORY_EXECUTION_STEPS}")
        _validated_timestep(self.config)
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
        prepared = _PreparedTrajectory(local=validated, world=world_trajectory)
        total_reward = 0.0
        final_result: tuple[Any, float, bool, bool, dict[str, Any]] | None = None
        step_record = _TrajectoryStepRecord.empty()
        for index in range(TRAJECTORY_EXECUTION_STEPS):
            action = prepared if index == 0 else None
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
                terminated,
                truncated,
                float(world_trajectory.angular_velocities[index]),
                capture_traffic_frame(self),
            )
            final_result = (observation, total_reward, terminated, truncated, info)
            if terminated or truncated:
                break
        if final_result is None:
            raise RuntimeError("trajectory execution completed without a simulator step")
        observation, _, terminated, truncated, final_info = final_result
        result_info = step_record.update_info(final_info, world_trajectory, total_reward)
        return observation, total_reward, terminated, truncated, result_info

    def _step_once(
        self, action: _PreparedTrajectory | None
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

    def _step_planner_simulator(self, actions: dict[str, _PreparedTrajectory | None]) -> dict:
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
