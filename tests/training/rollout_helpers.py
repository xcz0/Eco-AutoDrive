"""Scripted rollout decisions and execution records shared by collector tests."""

from __future__ import annotations

import numpy as np
import torch

from eco_planner.envs import TrajectoryExecutionRecord, TrajectoryExecutionResult
from eco_planner.envs.domain import (
    EnergyMetrics,
    TrafficFrame,
    TransitionMetricInput,
    TransitionMetrics,
)
from eco_planner.planning.policy import ExplorationPolicyContext
from eco_planner.rl.rollout import (
    ExecutionTransitionAudit,
    RolloutProvenance,
    build_training_decision,
)

from .helpers import _decision_audit, _reward_result


def _context() -> ExplorationPolicyContext:
    return ExplorationPolicyContext(
        scene_tokens=torch.zeros((1, 2, 4)),
        scene_padding_mask=torch.zeros((1, 2), dtype=torch.bool),
        navigation_tokens=torch.zeros((1, 1, 4)),
        navigation_padding_mask=torch.zeros((1, 1), dtype=torch.bool),
        reference_trajectory=torch.zeros((1, 80, 4)),
    )


def _transition():
    context = _context()
    training_decision = build_training_decision(
        context,
        torch.tensor([[-0.5, 0.5]]),
        torch.tensor([0.5]),
        torch.tensor([1.0]),
    )
    decision_audit = _decision_audit(context)
    execution_audit = ExecutionTransitionAudit(
        reward_result=_reward_result(0.25),
        substep_results=(_reward_result(0.25),),
        route_completion_delta=0.1,
        distance_m=1.0,
        speed_mps=2.0,
        stopped=False,
        collision=False,
        wrong_direction=False,
        position_error_m=0.0,
        heading_error_rad=0.0,
        arrive_dest=False,
        out_of_road=False,
        crash_vehicle=False,
        crash_object=False,
        crash_building=False,
        crash_human=False,
        crash_sidewalk=False,
        terminated=False,
        truncated=False,
    )
    provenance = RolloutProvenance(0, 1, 2, 0)
    return training_decision, decision_audit, execution_audit, provenance


def _substep_metrics(
    index: int,
    *,
    stopped: bool = False,
    collision: bool = False,
    wrong_direction: bool = False,
) -> TransitionMetrics:
    position = (float(index), 0.0)
    return TransitionMetrics(
        input=TransitionMetricInput(
            previous_position_xy_m=(0.0, 0.0),
            position_xy_m=position,
            previous_velocity_xy_mps=(0.0, 0.0),
            velocity_xy_mps=(0.0, 0.0),
            previous_acceleration_xy_mps2=(0.0, 0.0),
            heading_rad=0.0,
            yaw_rate_radps=0.0,
            route_progress_delta_m=0.0,
            route_heading_rad=0.0,
            speed_limit_mps=10.0,
            ego_width_m=2.0,
            ego_length_m=4.0,
            traffic_frame=TrafficFrame(index, position, 0.0, 1.0, (), ()),
            target_position_xy_m=position,
            target_heading_rad=0.0,
            crash_vehicle=False,
            crash_object=False,
            crash_building=False,
            crash_human=False,
            crash_sidewalk=False,
            out_of_road=False,
            native_step_energy_ml=0.0,
            native_episode_energy_ml=0.0,
            timestep_s=0.1,
        ),
        speed_mps=2.0 + index,
        longitudinal_acceleration_mps2=0.0,
        lateral_acceleration_mps2=0.0,
        jerk_mps3=0.0,
        step_distance_m=1.0,
        position_error_m=float(index),
        heading_error_rad=float(index) / 10.0,
        route_heading_error_rad=0.0,
        wrong_direction=wrong_direction,
        stopped=stopped,
        collision=collision,
        energy=EnergyMetrics("metadrive_fuel_proxy", 1.0, None, 0.1),
    )


def _execution_result(
    metrics: tuple[TransitionMetrics, ...], *, route_completion: float = 0.5
) -> TrajectoryExecutionResult:
    count = len(metrics)
    return TrajectoryExecutionResult(
        execution=TrajectoryExecutionRecord(
            start_center=np.zeros(2),
            start_heading=0.0,
            world_centers=np.zeros((80, 2)),
            world_headings=np.zeros(80),
            substep_states=np.zeros((count, 7)),
            target_centers=np.zeros((count, 2)),
            target_headings=np.zeros(count),
            substep_terminated=np.zeros(count, dtype=np.bool_),
            substep_truncated=np.zeros(count, dtype=np.bool_),
            traffic_frames=(),
            route_completion=route_completion,
            arrive_dest=False,
            out_of_road=False,
            crash_vehicle=False,
            crash_object=False,
            crash_building=False,
            crash_human=False,
            max_step=False,
        ),
        metrics=metrics,
        terminated=False,
        truncated=False,
    )
