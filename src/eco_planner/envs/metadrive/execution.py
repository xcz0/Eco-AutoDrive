"""Kinematic trajectory execution and typed MetaDrive results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from eco_planner.contracts import (
    PLANNER_HORIZON,
    ExecutionMode,
    validate_metadrive_timestep,
)
from eco_planner.envs.array_types import (
    ExecutionBooleanArray,
    ExecutionPointArray,
    ExecutionScalarArray,
    ExecutionStateArray,
    WorldHeadingArray,
    WorldPointArray,
    WorldVectorArray,
)
from eco_planner.envs.domain.metrics import (
    TransitionMetrics,
    executed_fuel_proxy_step_energy_ml,
)
from eco_planner.envs.domain.traffic import TrafficFrame
from eco_planner.envs.domain.trajectory import WorldTrajectory
from eco_planner.envs.geometry import shortest_angle_delta


@dataclass(frozen=True, slots=True)
class TrajectoryExecutionRecord:
    start_center: WorldVectorArray
    start_heading: float
    world_centers: WorldPointArray
    world_headings: WorldHeadingArray
    substep_states: ExecutionStateArray
    target_centers: ExecutionPointArray
    target_headings: ExecutionScalarArray
    position_errors_m: ExecutionScalarArray
    heading_errors_rad: ExecutionScalarArray
    substep_rewards: ExecutionScalarArray
    substep_dense_rewards: ExecutionScalarArray
    substep_native_energy_ml: ExecutionScalarArray
    substep_native_episode_energy_ml: ExecutionScalarArray
    substep_terminated: ExecutionBooleanArray
    substep_truncated: ExecutionBooleanArray
    traffic_frames: tuple[TrafficFrame, ...]
    route_completion: float
    arrive_dest: bool
    out_of_road: bool
    crash_vehicle: bool
    crash_object: bool
    crash_building: bool
    crash_human: bool
    max_step: bool
    crash_sidewalk: bool = False
    substep_executed_fuel_proxy_energy_ml: ExecutionScalarArray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64)
    )
    substep_distance_m: ExecutionScalarArray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64)
    )
    substep_metrics: tuple[TransitionMetrics, ...] = ()


@dataclass(slots=True)
class TrajectoryExecutionRecorder:
    states: ExecutionStateArray
    rewards: ExecutionScalarArray
    dense_rewards: ExecutionScalarArray
    native_energy_ml: ExecutionScalarArray
    native_episode_energy_ml: ExecutionScalarArray
    executed_fuel_proxy_energy_ml: ExecutionScalarArray
    distance_m: ExecutionScalarArray
    metrics: list[TransitionMetrics]
    terminated: ExecutionBooleanArray
    truncated: ExecutionBooleanArray
    traffic_frames: list[TrafficFrame]
    count: int

    @classmethod
    def empty(cls, execution_steps: int) -> TrajectoryExecutionRecorder:
        return cls(
            states=np.empty((execution_steps, 7), dtype=np.float64),
            rewards=np.empty(execution_steps, dtype=np.float64),
            dense_rewards=np.empty(execution_steps, dtype=np.float64),
            native_energy_ml=np.empty(execution_steps, dtype=np.float64),
            native_episode_energy_ml=np.empty(execution_steps, dtype=np.float64),
            executed_fuel_proxy_energy_ml=np.empty(execution_steps, dtype=np.float64),
            distance_m=np.empty(execution_steps, dtype=np.float64),
            metrics=[],
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
        native_energy_ml: float,
        native_episode_energy_ml: float,
        terminated: bool,
        truncated: bool,
        angular_velocity: float,
        traffic_frame: TrafficFrame,
        metrics: TransitionMetrics,
    ) -> None:
        index = self.count
        self.states[index, :2] = np.asarray(agent.position, dtype=np.float64)
        self.states[index, 2] = float(agent.heading_theta)
        self.states[index, 3:5] = agent.velocity
        self.states[index, 5] = float(agent.speed)
        self.states[index, 6] = angular_velocity
        self.rewards[index] = reward
        self.dense_rewards[index] = dense_reward
        self.native_energy_ml[index] = native_energy_ml
        self.native_episode_energy_ml[index] = native_episode_energy_ml
        self.executed_fuel_proxy_energy_ml[index] = metrics.executed_fuel_proxy_step_energy_ml
        self.distance_m[index] = metrics.step_distance_m
        self.metrics.append(metrics)
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
            substep_native_energy_ml=self.native_energy_ml[:executed_steps].copy(),
            substep_native_episode_energy_ml=(
                self.native_episode_energy_ml[:executed_steps].copy()
            ),
            substep_executed_fuel_proxy_energy_ml=(
                self.executed_fuel_proxy_energy_ml[:executed_steps].copy()
            ),
            substep_distance_m=self.distance_m[:executed_steps].copy(),
            substep_metrics=tuple(self.metrics),
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
            crash_sidewalk=bool(final_info["crash_sidewalk"]),
        )
        result = dict(final_info)
        result["trajectory_execution_steps"] = executed_steps
        result["trajectory_reward_sum"] = total_reward
        result["trajectory_execution"] = execution
        return result


def metadrive_fuel_proxy_step_energy_ml(
    start_position: WorldVectorArray,
    end_position: WorldVectorArray,
    speed_mps: float,
) -> float:
    """Evaluate MetaDrive's fuel proxy for one executed 0.1 s substep."""

    return executed_fuel_proxy_step_energy_ml(start_position, end_position, speed_mps)


def execution_steps_from_config(config: Any) -> int:
    """Normalize the external execution-mode boundary to a fixed project contract."""

    mode_value = config.get("execution_mode")
    if mode_value is not None:
        try:
            return ExecutionMode(mode_value).steps
        except ValueError as error:
            raise ValueError("execution_mode must be 'rollout' or 'evaluation'") from error
    # Existing serialized experiment configs retain these keys. They are a compatibility input
    # boundary only; all downstream code receives the fixed mode-derived step count.
    horizon = _require_positive_int(config, "trajectory_horizon")
    execution_steps = _require_positive_int(config, "trajectory_execution_steps")
    if horizon != PLANNER_HORIZON or execution_steps not in {
        mode.steps for mode in ExecutionMode
    }:
        raise ValueError("legacy trajectory configuration does not match the fixed project ABI")
    _validated_timestep(config)
    return execution_steps


def _require_positive_int(config: Any, name: str) -> int:
    value = config[name]
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validated_timestep(config: Any) -> float:
    return validate_metadrive_timestep(
        config["physics_world_step_size"], config["decision_repeat"]
    )


def finite_info_scalar(info: dict[str, Any], name: str) -> float:
    value: float = info[name]
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)
