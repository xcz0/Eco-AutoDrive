"""Reward configurations and physical inputs shared by training tests."""

from __future__ import annotations

import math

import torch
from omegaconf import OmegaConf

from eco_planner._repository import CONFIG_ROOT
from eco_planner.envs.domain import (
    MetaDriveFuelProxyProvider,
    StaticTrafficObjectState,
    TrafficFrame,
    TrafficParticipantState,
    TransitionMetricInput,
    derive_transition_metrics,
)
from eco_planner.reward import (
    PlannerRFTEnergyRewardConfig,
    PlannerRFTNoEnergyRewardConfig,
    aggregate_transition_reward,
    evaluate_plannerrft_no_energy_step,
)
from eco_planner.rl.rollout.assembly import (
    RolloutEpisodeBuilder,
    build_training_decision,
    build_training_transition,
)
from eco_planner.rl.rollout.contracts import ExecutionTransitionAudit, RolloutProvenance

from .helpers import _context, _decision_audit


def _config() -> PlannerRFTEnergyRewardConfig:
    return PlannerRFTEnergyRewardConfig.model_validate(
        {
            "name": "plannerrft_energy_v1",
            "weights": {
                "ttc": 5.0,
                "progress": 5.0,
                "comfort": 2.0,
                "speed": 4.0,
                "energy": 1.0,
            },
            "gates": {
                "collision_vehicle": True,
                "collision_object": True,
                "collision_building": True,
                "collision_human": True,
                "collision_sidewalk": True,
            },
            "ttc": {
                "critical_ttc_s": 1.0,
                "safe_ttc_s": 4.0,
                "maximum_ttc_s": 10.0,
                "minimum_closing_speed_mps": 0.1,
                "lateral_margin_m": 0.5,
                "longitudinal_margin_m": 0.5,
            },
            "progress": {"full_score_delta_m": 1.0},
            "comfort": {
                "longitudinal_acceleration_limit_mps2": 3.0,
                "lateral_acceleration_limit_mps2": 3.0,
                "jerk_limit_mps3": 5.0,
                "yaw_rate_limit_radps": 0.5,
            },
            "speed": {"overspeed_margin_mps": 0.0, "zero_score_overspeed_mps": 5.0},
            "energy": {"reference_ml_per_km": 50.0, "minimum_step_distance_m": 0.01},
        }
    )


def _no_energy_config() -> PlannerRFTNoEnergyRewardConfig:
    payload = _config().model_dump(mode="python")
    payload["name"] = "plannerrft_no_energy_v1"
    payload["weights"] = {
        key: value for key, value in payload["weights"].items() if key != "energy"
    }
    return PlannerRFTNoEnergyRewardConfig.model_validate(payload)


def _frame(
    *,
    participants: tuple[TrafficParticipantState, ...] = (),
    static_objects: tuple[StaticTrafficObjectState, ...] = (),
) -> TrafficFrame:
    return TrafficFrame(
        simulator_step=1,
        ego_center_xy_m=(1.0, 0.0),
        ego_heading_rad=0.0,
        ego_rear_wheelbase_m=1.0,
        participants=participants,
        static_objects=static_objects,
    )


def _input(**updates: object) -> TransitionMetricInput:
    values: dict[str, object] = {
        "previous_position_xy_m": (0.0, 0.0),
        "position_xy_m": (1.0, 0.0),
        "previous_velocity_xy_mps": (10.0, 0.0),
        "velocity_xy_mps": (10.0, 0.0),
        "previous_acceleration_xy_mps2": (0.0, 0.0),
        "heading_rad": 0.0,
        "yaw_rate_radps": 0.0,
        "route_progress_delta_m": 1.0,
        "route_heading_rad": 0.0,
        "speed_limit_mps": 10.0,
        "ego_width_m": 2.0,
        "ego_length_m": 4.0,
        "traffic_frame": _frame(),
        "target_position_xy_m": (1.0, 0.0),
        "target_heading_rad": 0.0,
        "crash_vehicle": False,
        "crash_object": False,
        "crash_building": False,
        "crash_human": False,
        "crash_sidewalk": False,
        "out_of_road": False,
        "native_step_energy_ml": 0.0,
        "native_episode_energy_ml": 0.0,
        "timestep_s": 0.1,
    }
    values.update(updates)
    return TransitionMetricInput(**values)  # type: ignore[arg-type]


def _metrics(**updates: object):
    return derive_transition_metrics(_input(**updates), MetaDriveFuelProxyProvider())


def _reward_profile_payload(name: str) -> dict:
    config_root = CONFIG_ROOT / "components" / "reward"
    payload = OmegaConf.to_container(OmegaConf.load(config_root / f"{name}.yaml"), resolve=True)
    assert isinstance(payload, dict)
    return payload


def _five_substep_parity_case(
    base: PlannerRFTNoEnergyRewardConfig | PlannerRFTEnergyRewardConfig,
    evaluate=evaluate_plannerrft_no_energy_step,
    count=5,
):
    metrics = (
        _metrics(),
        _metrics(route_progress_delta_m=0.4, velocity_xy_mps=(12.0, 0.0)),
        _metrics(position_xy_m=(0.0, 0.0), velocity_xy_mps=(0.0, 0.0), route_progress_delta_m=0.0),
        _metrics(crash_vehicle=True),
        _metrics(route_heading_rad=math.pi),
    )
    metrics = metrics[:count]
    results = tuple(evaluate(base, metric) for metric in metrics)
    execution = ExecutionTransitionAudit(
        reward_result=aggregate_transition_reward(results),
        substep_results=results,
        route_completion_delta=0.1,
        distance_m=5.0,
        speed_mps=10.0,
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
        terminated=True,
        truncated=False,
    )
    builder = RolloutEpisodeBuilder()
    builder.append(
        build_training_transition(
            build_training_decision(
                _context(),
                torch.tensor([[-0.5, 0.5]]),
                torch.tensor([0.5]),
                torch.tensor([1.0]),
            ),
            execution.reward_result,
            terminated=execution.terminated,
            truncated=execution.truncated,
        ),
        _decision_audit(),
        execution,
        RolloutProvenance(0, 1, 2, 0),
    )
    return builder.finish("terminated", torch.zeros(1)), metrics, results
