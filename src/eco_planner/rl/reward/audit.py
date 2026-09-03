"""Profile-specific reward audits derived from execution metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

from eco_planner.envs.domain.metrics import TransitionMetrics


@dataclass(frozen=True, slots=True)
class PlannerRFTEnergyRewardAudit:
    profile_name: Literal["plannerrft_energy_v1"]
    reward_total: float
    reward_ungated: float
    reward_gate: float
    collision_score: float
    drivable_score: float
    wrong_direction_score: float
    ttc_score: float
    progress_score: float
    comfort_score: float
    speed_score: float
    energy_score: float
    has_ttc_candidate: bool
    min_ttc_s: float
    route_progress_delta_m: float
    speed_mps: float
    speed_limit_mps: float
    overspeed_mps: float
    longitudinal_acceleration_mps2: float
    lateral_acceleration_mps2: float
    jerk_mps3: float
    yaw_rate_radps: float
    step_distance_m: float
    native_step_energy_ml: float
    native_episode_energy_ml: float
    executed_fuel_proxy_step_energy_ml: float
    executed_fuel_proxy_ml_per_km: float
    energy_distance_valid: bool


@dataclass(frozen=True, slots=True)
class MetaDriveBuiltinRewardAudit:
    profile_name: Literal["metadrive_builtin_v1"]
    reward_total: float
    dense_reward: float
    terminal_override: float
    step_distance_m: float
    native_step_energy_ml: float
    native_episode_energy_ml: float
    executed_fuel_proxy_step_energy_ml: float
    executed_fuel_proxy_ml_per_km: float
    energy_distance_valid: bool


RewardAudit: TypeAlias = MetaDriveBuiltinRewardAudit | PlannerRFTEnergyRewardAudit


def build_metadrive_builtin_reward_audit(
    metrics: TransitionMetrics, *, reward_total: float, dense_reward: float
) -> MetaDriveBuiltinRewardAudit:
    """Attach native MetaDrive reward fields to already-derived transition metrics."""

    fuel_ml = metrics.energy.fuel_ml
    if fuel_ml is None:
        raise ValueError("MetaDrive reward audit requires a fuel-volume metric")
    return MetaDriveBuiltinRewardAudit(
        profile_name="metadrive_builtin_v1",
        reward_total=reward_total,
        dense_reward=dense_reward,
        terminal_override=reward_total - dense_reward,
        step_distance_m=metrics.step_distance_m,
        native_step_energy_ml=metrics.input.native_step_energy_ml,
        native_episode_energy_ml=metrics.input.native_episode_energy_ml,
        executed_fuel_proxy_step_energy_ml=fuel_ml,
        executed_fuel_proxy_ml_per_km=metrics.energy.fuel_ml_per_km or 0.0,
        energy_distance_valid=metrics.step_distance_m > 0.0,
    )


__all__ = [
    "MetaDriveBuiltinRewardAudit",
    "PlannerRFTEnergyRewardAudit",
    "RewardAudit",
    "build_metadrive_builtin_reward_audit",
]
