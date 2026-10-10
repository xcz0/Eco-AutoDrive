"""PlannerRFT-style gated objective over reusable reward components."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, fields
from typing import Any

from eco_planner.envs.domain import TransitionMetrics

from ..components import (
    comfort_score,
    energy_score,
    progress_score,
    safety_gate,
    speed_score,
    ttc_score,
)
from ..config import (
    PlannerRFTEnergyRewardConfig,
    PlannerRFTNoEnergyRewardConfig,
    RewardProfileConfig,
)
from ..result import (
    PlannerRFTObjectiveResult,
    PlannerRFTRewardResult,
    RewardComponents,
    RewardDiagnostics,
    RewardProfileName,
)


def combine_component_scores(
    weights: Mapping[str, float],
    component_scores: Mapping[str, Any],
) -> Any:
    """Normalized weighted sum shared by online and offline objective evaluation.

    Accepts scalar or tensor component scores; only the component names present in
    ``weights`` participate, so a no-energy profile keeps its denominator at four.
    """

    total: Any = None
    for name, weight in weights.items():
        term = component_scores[name] * weight
        total = term if total is None else total + term
    return total / sum(weights.values())


def apply_safety_gate(base_total: Any, safety_gate: Any) -> Any:
    """Apply the PlannerRFT safety gate to a base total."""

    return safety_gate * base_total


def compose_plannerrft_objective(
    weights: Mapping[str, float], components: RewardComponents, gate: float
) -> PlannerRFTObjectiveResult:
    """Recompose stored scores without claiming to recompute any component.

    Profile validation owns the declared weight constraints. The existing
    energy-only diagnostic uses this same formulation with ``{"energy": 1.0}``;
    it does not create a new profile or change the stored component mappings.
    """

    base = combine_component_scores(weights, asdict(components))
    return PlannerRFTObjectiveResult(
        total=apply_safety_gate(base, gate),
        base_total=base,
        safety_gate=gate,
        components=components,
    )


def _evaluate_shared(
    config: RewardProfileConfig, metrics: TransitionMetrics
) -> tuple[float, RewardComponents, RewardDiagnostics]:
    """Gates, components, and diagnostics shared by every PlannerRFT profile."""

    gate, collision, drivable, wrong_direction = safety_gate(config.gates, metrics)
    ttc, min_ttc_s, has_ttc_candidate = ttc_score(config.ttc, metrics)
    progress = progress_score(config.progress, metrics)
    comfort = comfort_score(config.comfort, metrics)
    speed, overspeed_mps = speed_score(config.speed, metrics)
    energy, fuel_ml_per_km, energy_distance_valid = energy_score(config.energy, metrics)
    components = RewardComponents(ttc, progress, comfort, speed, energy)
    step = metrics.input
    fuel_ml = metrics.energy.fuel_ml
    if fuel_ml is None:
        raise RuntimeError("energy component accepted a missing fuel-volume metric")
    diagnostics = RewardDiagnostics(
        collision_score=collision,
        drivable_score=drivable,
        wrong_direction_score=wrong_direction,
        has_ttc_candidate=has_ttc_candidate,
        min_ttc_s=min_ttc_s,
        route_progress_delta_m=step.route_progress_delta_m,
        speed_mps=metrics.speed_mps,
        speed_limit_mps=step.speed_limit_mps,
        overspeed_mps=overspeed_mps,
        longitudinal_acceleration_mps2=metrics.longitudinal_acceleration_mps2,
        lateral_acceleration_mps2=metrics.lateral_acceleration_mps2,
        jerk_mps3=metrics.jerk_mps3,
        yaw_rate_radps=step.yaw_rate_radps,
        step_distance_m=metrics.step_distance_m,
        native_step_energy_ml=step.native_step_energy_ml,
        native_episode_energy_ml=step.native_episode_energy_ml,
        executed_fuel_proxy_step_energy_ml=fuel_ml,
        executed_fuel_proxy_ml_per_km=fuel_ml_per_km,
        energy_distance_valid=energy_distance_valid,
    )
    return gate, components, diagnostics


def _finalize(
    profile_name: RewardProfileName,
    objective: PlannerRFTObjectiveResult,
    diagnostics: RewardDiagnostics,
) -> PlannerRFTRewardResult:
    result = PlannerRFTRewardResult(
        profile_name=profile_name,
        total=objective.total,
        base_total=objective.base_total,
        safety_gate=objective.safety_gate,
        components=objective.components,
        diagnostics=diagnostics,
    )
    values = [result.total, result.base_total, result.safety_gate]
    values.extend(
        float(getattr(result.components, item.name)) for item in fields(result.components)
    )
    values.extend(
        float(value)
        for item in fields(result.diagnostics)
        if not isinstance((value := getattr(result.diagnostics, item.name)), bool)
    )
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError("PlannerRFT reward produced a non-finite result")
    return result


def evaluate_plannerrft_energy_step(
    config: PlannerRFTEnergyRewardConfig,
    metrics: TransitionMetrics,
) -> PlannerRFTRewardResult:
    """Evaluate one transition without accessing simulator or runtime objects."""

    gate, components, diagnostics = _evaluate_shared(config, metrics)
    objective = compose_plannerrft_objective(config.weights.model_dump(), components, gate)
    return _finalize(config.name, objective, diagnostics)


def evaluate_plannerrft_no_energy_step(
    config: PlannerRFTNoEnergyRewardConfig,
    metrics: TransitionMetrics,
) -> PlannerRFTRewardResult:
    """Evaluate the no-energy R0 objective; energy stays an unweighted diagnostic."""

    gate, components, diagnostics = _evaluate_shared(config, metrics)
    objective = compose_plannerrft_objective(config.weights.model_dump(), components, gate)
    return _finalize(config.name, objective, diagnostics)


__all__ = [
    "apply_safety_gate",
    "combine_component_scores",
    "compose_plannerrft_objective",
    "evaluate_plannerrft_energy_step",
    "evaluate_plannerrft_no_energy_step",
]
