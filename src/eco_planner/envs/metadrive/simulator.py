"""Trajectory-level MetaDrive environment lifecycle and step orchestration."""

from __future__ import annotations

from typing import Any, cast

import gymnasium as gym
import numpy as np
from metadrive.constants import DEFAULT_AGENT, TerminationState
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.obs.observation_base import BaseObservation
from metadrive.utils import Config, concat_step_infos, merge_dicts

from eco_planner.envs.array_types import (
    PlannerOnlyObservationArray,
    TrajectoryArray,
    WorldVectorArray,
)
from eco_planner.envs.contracts import (
    METADRIVE_DECISION_REPEAT,
    METADRIVE_PHYSICS_STEP_S,
    ExecutionMode,
)
from eco_planner.envs.domain.traffic import TrafficFrame
from eco_planner.envs.domain.trajectory import WorldTrajectory, to_world_trajectory
from eco_planner.envs.metadrive.execution import (
    TRAJECTORY_TIMESTEP_S,
    TrajectoryExecutionRecorder,
    execution_steps_from_config,
    finite_info_scalar,
)
from eco_planner.envs.metadrive.lane_speed import (
    ProgrammaticLaneSpeedAdapter,
    model_lane_speed_limit_mps,
)
from eco_planner.envs.metadrive.map import navigation_route_roads
from eco_planner.envs.metadrive.policy import KinematicTrajectoryPolicy
from eco_planner.envs.metadrive.reward import (
    MetaDriveBuiltinRewardAudit,
    MetaDriveBuiltinRewardConfig,
    PlannerRFTEnergyRewardConfig,
    RewardProfileConfig,
    RewardStepInput,
    executed_fuel_proxy_step_energy_ml,
    score_plannerrft_energy_step,
)
from eco_planner.envs.metadrive.snapshot import capture_traffic_frame

_PLANNER_ONLY_OBSERVATION: PlannerOnlyObservationArray = np.zeros(1, dtype=np.float32)
_ZERO_VELOCITY: WorldVectorArray = np.zeros(2, dtype=np.float64)


class _PlannerOnlyObservation(BaseObservation):
    """Minimal Gym observation for an environment observed through project adapters."""

    @property
    def observation_space(self) -> gym.spaces.Box:
        return gym.spaces.Box(0.0, 0.0, shape=(1,), dtype=np.float32)

    def observe(self, *args: Any, **kwargs: Any) -> PlannerOnlyObservationArray:
        return _PLANNER_ONLY_OBSERVATION.copy()


class TrajectoryMetaDriveEnv(MetaDriveEnv):
    """Single-agent MetaDrive environment with an explicit 0.1 s or 0.5 s trajectory prefix."""

    @classmethod
    def default_config(cls) -> Config:
        config = super().default_config()
        config.update(
            {
                # Compatibility input keys for resolved experiments created before execution_mode.
                # They are normalized at construction and are not used by the execution path.
                "trajectory_horizon": None,
                "trajectory_execution_steps": None,
                "execution_mode": None,
                "programmatic_lane_speed_limit_kmh": None,
                "programmatic_lane_speed_limit_profile_kmh": None,
            },
            allow_add_new_key=True,
        )
        return config

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        *,
        reward_profile: RewardProfileConfig | None = None,
    ) -> None:
        if config is None:
            raise ValueError("TrajectoryMetaDriveEnv requires an explicit configuration")
        required = {
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
        execution_steps = execution_steps_from_config(config)
        configured = dict(config)
        if isinstance(reward_profile, MetaDriveBuiltinRewardConfig):
            configured.update(reward_profile.model_dump(exclude={"name"}))
        configured["physics_world_step_size"] = METADRIVE_PHYSICS_STEP_S
        configured["decision_repeat"] = METADRIVE_DECISION_REPEAT
        configured["execution_mode"] = (
            ExecutionMode.ROLLOUT.value
            if execution_steps == ExecutionMode.ROLLOUT.steps
            else ExecutionMode.EVALUATION.value
        )
        configured["agent_policy"] = KinematicTrajectoryPolicy
        configured["agent_observation"] = _PlannerOnlyObservation
        self._reward_profile = reward_profile
        self._previous_reward_position: np.ndarray | None = None
        self._previous_reward_velocity: np.ndarray | None = None
        self._previous_reward_acceleration: np.ndarray | None = None
        self._terminal_out_of_road = False
        super().__init__(configured)
        if self.config["is_multi_agent"]:
            raise ValueError("TrajectoryMetaDriveEnv supports only single-agent operation")
        if self.config["manual_control"]:
            raise ValueError("TrajectoryMetaDriveEnv does not support manual control")
        self._execution_steps = execution_steps
        self._programmatic_lane_speed_adapter = ProgrammaticLaneSpeedAdapter(
            self.config["programmatic_lane_speed_limit_kmh"],
            self.config["programmatic_lane_speed_limit_profile_kmh"],
        )
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

        return self._programmatic_lane_speed_adapter.audit

    @property
    def initial_traffic_frame(self) -> TrafficFrame:
        """Return the traffic snapshot captured immediately after the latest reset."""

        return cast(TrafficFrame, self._initial_traffic_frame)

    @property
    def route_completion(self) -> float:
        """Return the current finite route-completion fraction."""

        value = float(self.agent.navigation.route_completion)
        if not np.isfinite(value):
            raise RuntimeError("MetaDrive returned non-finite route completion")
        return value

    @property
    def route_length_m(self) -> float:
        """Return the total length of the current navigation route."""

        checkpoints = list(self.agent.navigation.checkpoints)
        if len(checkpoints) < 2:
            raise RuntimeError("MetaDrive navigation did not expose a complete route")
        current_map = cast(Any, self.current_map)
        graph = current_map.road_network.graph
        lengths: list[float] = []
        for start, end in zip(checkpoints[:-1], checkpoints[1:], strict=True):
            lanes = graph.get(start, {}).get(end, [])
            if not lanes:
                raise RuntimeError(f"route edge {(start, end)!r} has no lane")
            length = getattr(lanes[0], "length", None)
            if isinstance(length, (bool, np.bool_)) or not isinstance(
                length, (int, float, np.integer, np.floating)
            ):
                raise RuntimeError(f"route edge {(start, end)!r} has an invalid length")
            if not np.isfinite(length) or float(length) <= 0.0:
                raise RuntimeError(f"route edge {(start, end)!r} has an invalid length")
            lengths.append(float(length))
        return float(sum(lengths))

    @property
    def is_out_of_road_terminal(self) -> bool:
        """Whether the latest transition ended because the ego is out of road."""

        return self._terminal_out_of_road

    def reset(self, *args: Any, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        self._initial_traffic_frame = None
        self._terminal_out_of_road = False
        self._programmatic_lane_speed_adapter.clear_audit()
        if self.engine is not None:
            # BaseEnv.reset advances taskMgr once; do not replay the prior episode's trajectory.
            self.engine.external_actions = None
        result = super().reset(*args, **kwargs)
        current_map = self.current_map
        if current_map is None:
            raise RuntimeError("MetaDrive did not create a current map during reset")
        self._programmatic_lane_speed_adapter.apply(current_map)
        self._initial_traffic_frame = capture_traffic_frame(self)
        self._previous_reward_position = np.asarray(self.agent.position, dtype=np.float64).copy()
        self._previous_reward_velocity = np.asarray(self.agent.velocity, dtype=np.float64).copy()
        self._previous_reward_acceleration = np.zeros(2, dtype=np.float64)
        return result

    def step(self, trajectory: TrajectoryArray) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        rear_wheelbase = self.agent.REAR_WHEELBASE
        world_trajectory = to_world_trajectory(
            trajectory,
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
            observation, reward, terminated, truncated, info = self._step_once(
                action, float(world_trajectory.angular_velocities[index])
            )
            total_reward += float(reward)
            step_record.append(
                self.agent,
                float(reward),
                finite_info_scalar(info, "step_reward"),
                finite_info_scalar(info, "step_energy"),
                finite_info_scalar(info, "episode_energy"),
                terminated,
                truncated,
                float(world_trajectory.angular_velocities[index]),
                info.pop("_project_traffic_frame"),
                info.pop("_project_reward_audit"),
            )
            if terminated or truncated:
                break
        result_info = step_record.update_info(info, world_trajectory, total_reward)
        return observation, total_reward, terminated, truncated, result_info

    def _step_once(
        self, action: WorldTrajectory | None, yaw_rate_radps: float
    ) -> tuple[PlannerOnlyObservationArray, float, bool, bool, dict[str, Any]]:
        actions = {DEFAULT_AGENT: action}
        engine_info = self._step_planner_simulator(actions)
        agent_id, agent = next(iter(self.agents.items()))
        self.episode_lengths[agent_id] += 1
        builtin_reward, reward_info = super().reward_function(agent_id)
        done, done_info = self.done_function(agent_id)
        _, cost_info = self.cost_function(agent_id)
        self.dones[agent_id] = done or self.dones[agent_id]
        info = concat_step_infos(
            [engine_info, {agent_id: done_info}, {agent_id: reward_info}, {agent_id: cost_info}]
        )
        agent_info = info.pop(agent_id)
        info.update(agent_info)
        traffic_frame = capture_traffic_frame(self)
        reward_input = self._reward_step_input(info, traffic_frame, yaw_rate_radps)
        if isinstance(self._reward_profile, PlannerRFTEnergyRewardConfig):
            reward_audit = score_plannerrft_energy_step(self._reward_profile, reward_input)
            reward = reward_audit.reward_total
            info["step_reward"] = reward_audit.reward_ungated
            info["route_completion"] = self.route_completion
        else:
            reward = float(builtin_reward)
            distance_m = float(
                np.linalg.norm(
                    np.asarray(reward_input.position_xy_m)
                    - np.asarray(reward_input.previous_position_xy_m)
                )
            )
            proxy_ml = executed_fuel_proxy_step_energy_ml(
                reward_input.previous_position_xy_m,
                reward_input.position_xy_m,
                float(np.linalg.norm(reward_input.velocity_xy_mps)),
            )
            distance_valid = distance_m > 0.0
            reward_audit = MetaDriveBuiltinRewardAudit(
                profile_name="metadrive_builtin_v1",
                reward_total=reward,
                dense_reward=finite_info_scalar(info, "step_reward"),
                terminal_override=reward - finite_info_scalar(info, "step_reward"),
                step_distance_m=distance_m,
                native_step_energy_ml=reward_input.native_step_energy_ml,
                native_episode_energy_ml=reward_input.native_episode_energy_ml,
                executed_fuel_proxy_step_energy_ml=proxy_ml,
                executed_fuel_proxy_ml_per_km=(
                    proxy_ml * 1000.0 / distance_m if distance_valid else 0.0
                ),
                energy_distance_valid=distance_valid,
            )
        self.episode_rewards[agent_id] += reward
        self._advance_reward_state(reward_input)
        truncated = bool(info.get(TerminationState.MAX_STEP, False))
        terminated = bool(self.dones[agent_id])
        self._terminal_out_of_road = terminated and bool(info[TerminationState.OUT_OF_ROAD])
        if self.config["horizon"] and self.episode_step > 5 * self.config["horizon"]:
            truncated = True
            if self.config["truncate_as_terminate"]:
                self.dones[agent_id] = terminated = True
        info["episode_reward"] = self.episode_rewards[agent_id]
        info["episode_length"] = self.episode_lengths[agent_id]
        if agent is not self.agent:
            raise RuntimeError("single-agent environment changed its active agent")
        info["_project_traffic_frame"] = traffic_frame
        info["_project_reward_audit"] = reward_audit
        return _PLANNER_ONLY_OBSERVATION.copy(), float(reward), terminated, truncated, info

    def _is_out_of_road(self, vehicle: Any) -> bool:
        """Treat a drivable lane outside the complete navigation route as off-road."""

        if super()._is_out_of_road(vehicle):
            return True
        lane = vehicle.lane
        lane_index = getattr(lane, "index", None)
        if not isinstance(lane_index, tuple) or len(lane_index) < 2:
            raise RuntimeError(f"vehicle lane has invalid index: {lane_index!r}")
        lane_road = (lane_index[0], lane_index[1])
        return lane_road not in navigation_route_roads(vehicle.navigation)

    def _reward_step_input(
        self, info: dict[str, Any], traffic_frame: TrafficFrame, yaw_rate_radps: float
    ) -> RewardStepInput:
        if (
            self._previous_reward_position is None
            or self._previous_reward_velocity is None
            or self._previous_reward_acceleration is None
        ):
            raise RuntimeError("reward state is unavailable before environment reset")
        vehicle = self.agent
        reference_lanes = vehicle.navigation.current_ref_lanes
        if not reference_lanes:
            raise RuntimeError("navigation did not expose current reference lanes")
        lane = vehicle.lane if vehicle.lane in reference_lanes else reference_lanes[0]
        previous_longitudinal, _ = lane.local_coordinates(self._previous_reward_position)
        current_longitudinal, _ = lane.local_coordinates(vehicle.position)
        lane_length = float(lane.length)
        route_heading = float(
            lane.heading_theta_at(float(np.clip(current_longitudinal, 0.0, lane_length)))
        )
        speed_limit_mps, has_speed_limit = model_lane_speed_limit_mps(lane)
        if not has_speed_limit:
            raise RuntimeError("current route lane does not expose a configured speed limit")
        return RewardStepInput(
            previous_position_xy_m=_finite_pair(
                self._previous_reward_position, "previous reward position"
            ),
            position_xy_m=_finite_pair(vehicle.position, "vehicle position"),
            previous_velocity_xy_mps=_finite_pair(
                self._previous_reward_velocity, "previous reward velocity"
            ),
            velocity_xy_mps=_finite_pair(vehicle.velocity, "vehicle velocity"),
            previous_acceleration_xy_mps2=_finite_pair(
                self._previous_reward_acceleration, "previous reward acceleration"
            ),
            heading_rad=float(vehicle.heading_theta),
            yaw_rate_radps=yaw_rate_radps,
            route_progress_delta_m=float(current_longitudinal - previous_longitudinal),
            route_heading_rad=route_heading,
            speed_limit_mps=speed_limit_mps,
            ego_width_m=float(vehicle.WIDTH),
            ego_length_m=float(vehicle.LENGTH),
            traffic_frame=traffic_frame,
            crash_vehicle=bool(info["crash_vehicle"]),
            crash_object=bool(info["crash_object"]),
            crash_building=bool(info["crash_building"]),
            crash_human=bool(info["crash_human"]),
            crash_sidewalk=bool(info["crash_sidewalk"]),
            out_of_road=bool(info["out_of_road"]),
            native_step_energy_ml=finite_info_scalar(info, "step_energy"),
            native_episode_energy_ml=finite_info_scalar(info, "episode_energy"),
            timestep_s=TRAJECTORY_TIMESTEP_S,
        )

    def _advance_reward_state(self, reward_input: RewardStepInput) -> None:
        previous_velocity = np.asarray(reward_input.previous_velocity_xy_mps, dtype=np.float64)
        velocity = np.asarray(reward_input.velocity_xy_mps, dtype=np.float64)
        self._previous_reward_position = np.asarray(reward_input.position_xy_m, dtype=np.float64)
        self._previous_reward_velocity = velocity
        self._previous_reward_acceleration = (
            velocity - previous_velocity
        ) / reward_input.timestep_s

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
            dynamic_world = cast(Any, self.engine.physics_world.dynamic_world)
            dynamic_world.doPhysics(
                physics_step * repeats,
                repeats,
                physics_step,
            )
        after_info = self.engine.after_step()
        return merge_dicts(after_info, before_info, allow_new_keys=True, without_copy=True)


def _finite_pair(value: object, name: str) -> tuple[float, float]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (2,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite 2D vector")
    return float(array[0]), float(array[1])
