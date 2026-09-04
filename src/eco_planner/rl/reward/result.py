"""Typed results emitted by RL reward evaluators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class RewardComponents:
    """Normalized component scores combined by one reward objective."""

    ttc: float
    progress: float
    comfort: float
    speed: float
    energy: float


@dataclass(frozen=True, slots=True)
class RewardDiagnostics:
    """Non-objective values retained to explain one reward evaluation."""

    collision_score: float
    drivable_score: float
    wrong_direction_score: float
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
class RewardResult:
    """Final reward, objective components, and diagnostics from one evaluation."""

    profile_name: Literal["plannerrft_energy_v1"]
    total: float
    base_total: float
    safety_gate: float
    components: RewardComponents
    diagnostics: RewardDiagnostics


__all__ = ["RewardComponents", "RewardDiagnostics", "RewardResult"]
