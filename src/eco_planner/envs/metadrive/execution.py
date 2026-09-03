"""Kinematic trajectory-prefix execution over the MetaDrive backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from eco_planner.contracts import (
    PLANNER_HORIZON,
    SIMULATOR_STEP_S,
    ExecutionMode,
    validate_metadrive_timestep,
)
from eco_planner.energy import EnergyMetricProvider
from eco_planner.envs.array_types import (
    ExecutionBooleanArray,
    ExecutionScalarArray,
    ExecutionStateArray,
    TrajectoryArray,
    WorldVectorArray,
)
from eco_planner.envs.domain.execution import TrajectoryExecutionRecord
from eco_planner.envs.domain.metrics import TransitionMetrics
from eco_planner.envs.domain.traffic import TrafficFrame
from eco_planner.envs.domain.trajectory import WorldTrajectory, to_world_trajectory
from eco_planner.envs.geometry import shortest_angle_delta
from eco_planner.envs.metadrive.simulator import MetaDriveBackend, MetaDriveStepResult
from eco_planner.envs.metadrive.transition import TransitionExtractor

_ZERO_VELOCITY: WorldVectorArray = np.zeros(2, dtype=np.float64)


class TrajectoryExecutor:
    """Execute one canonical local trajectory prefix and assemble its immutable facts."""

    def __init__(
        self,
        backend: MetaDriveBackend,
        execution_steps: int,
        energy_provider: EnergyMetricProvider,
    ) -> None:
        if execution_steps not in {mode.steps for mode in ExecutionMode}:
            raise ValueError("execution_steps must match a fixed execution mode")
        self._backend = backend
        self._execution_steps = execution_steps
        self._transition_extractor = TransitionExtractor(energy_provider)

    def reset(self) -> TrafficFrame:
        """Reset transition history after the backend has reset."""

        return self._transition_extractor.reset(self._backend)

    def execute(
        self, trajectory: TrajectoryArray
    ) -> tuple[float, bool, bool, TrajectoryExecutionRecord]:
        """Execute the configured prefix, stopping at the first episode boundary."""

        rear_wheelbase = self._backend.agent.REAR_WHEELBASE
        if rear_wheelbase is None:
            raise RuntimeError("controlled vehicle does not define REAR_WHEELBASE")
        world_trajectory = to_world_trajectory(
            trajectory,
            center_position=np.asarray(self._backend.agent.position, dtype=np.float64),
            center_heading=float(self._backend.agent.heading_theta),
            rear_wheelbase=float(rear_wheelbase),
            timestep_s=SIMULATOR_STEP_S,
        )
        total_reward = 0.0
        recorder = TrajectoryExecutionRecorder.empty(self._execution_steps)
        final_step: MetaDriveStepResult | None = None
        for index in range(self._execution_steps):
            action = world_trajectory if index == 0 else None
            # The policy applies the waypoint in after_step. Suppress movement from the
            # previous waypoint during the intervening physics phase.
            self._backend.agent.set_velocity(_ZERO_VELOCITY)
            self._backend.agent.set_angular_velocity(0.0)
            step = self._backend.step_world_trajectory(action)
            metrics = self._transition_extractor.extract(
                self._backend,
                step,
                float(world_trajectory.angular_velocities[index]),
            )
            total_reward += step.builtin_reward
            recorder.append(
                self._backend.agent,
                step.builtin_reward,
                step.builtin_dense_reward,
                step,
                float(world_trajectory.angular_velocities[index]),
                metrics,
            )
            final_step = step
            if step.terminated or step.truncated:
                break
        if final_step is None:
            raise RuntimeError("trajectory executor did not advance the simulator")
        return (
            total_reward,
            final_step.terminated,
            final_step.truncated,
            recorder.build(world_trajectory, final_step),
        )


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
            count=0,
        )

    def append(
        self,
        agent: Any,
        reward: float,
        dense_reward: float,
        step: MetaDriveStepResult,
        angular_velocity: float,
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
        self.native_energy_ml[index] = step.native_step_energy_ml
        self.native_episode_energy_ml[index] = step.native_episode_energy_ml
        fuel_ml = metrics.energy.fuel_ml
        if fuel_ml is None:
            raise RuntimeError("trajectory execution requires a fuel-volume energy metric")
        self.executed_fuel_proxy_energy_ml[index] = fuel_ml
        self.distance_m[index] = metrics.step_distance_m
        self.metrics.append(metrics)
        self.terminated[index] = step.terminated
        self.truncated[index] = step.truncated
        self.count += 1

    def build(
        self,
        world_trajectory: WorldTrajectory,
        final_step: MetaDriveStepResult,
    ) -> TrajectoryExecutionRecord:
        executed_steps = self.count
        state_array = self.states[:executed_steps].copy()
        target_centers = world_trajectory.centers[1 : executed_steps + 1]
        target_headings = world_trajectory.headings[1 : executed_steps + 1]
        return TrajectoryExecutionRecord(
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
            traffic_frames=tuple(metric.input.traffic_frame for metric in self.metrics),
            route_completion=final_step.route_completion,
            arrive_dest=final_step.arrive_dest,
            out_of_road=final_step.out_of_road,
            crash_vehicle=final_step.crash_vehicle,
            crash_object=final_step.crash_object,
            crash_building=final_step.crash_building,
            crash_human=final_step.crash_human,
            crash_sidewalk=final_step.crash_sidewalk,
            max_step=final_step.max_step,
        )


def execution_steps_from_config(config: Any) -> int:
    """Normalize the external execution-mode boundary to a fixed project contract."""

    mode_value = config.get("execution_mode")
    if mode_value is not None:
        try:
            return ExecutionMode(mode_value).steps
        except ValueError as error:
            raise ValueError("execution_mode must be 'rollout' or 'evaluation'") from error
    horizon = _require_positive_int(config, "trajectory_horizon")
    execution_steps = _require_positive_int(config, "trajectory_execution_steps")
    if horizon != PLANNER_HORIZON or execution_steps not in {mode.steps for mode in ExecutionMode}:
        raise ValueError("legacy trajectory configuration does not match the fixed project ABI")
    validate_metadrive_timestep(config["physics_world_step_size"], config["decision_repeat"])
    return execution_steps


def _require_positive_int(config: Any, name: str) -> int:
    value = config[name]
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value
