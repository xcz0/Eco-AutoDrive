"""Pure aggregation of substep reward results into one transition reward.

The canonical closed-loop cadence gives one PPO transition its own execution
prefix of consecutive simulator substeps. Reward is evaluated per substep and
reduced into a single transition result. `aggregate_substep_rewards` owns the
optimization-relevant reduction and is reused by the online transition boundary
and by offline reweighting/rescoring, so the two paths cannot drift. This module
is pure: it applies no truncation, repair, or fallback and does not depend on
rollout or training state.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from .objectives.plannerrft import compose_plannerrft_objective
from .result import (
    PlannerRFTObjectiveResult,
    PlannerRFTRewardResult,
    RewardComponents,
    RewardDiagnostics,
    RewardResult,
)


def aggregate_scalar_rewards(substeps: Sequence[RewardResult]) -> RewardResult:
    """Sum executed scalar rewards without requiring objective explanation data."""

    if not substeps:
        raise ValueError("aggregate_scalar_rewards requires at least one result")
    return RewardResult(total=sum(substep.total for substep in substeps))


def aggregate_substep_rewards(
    substeps: Sequence[PlannerRFTObjectiveResult],
) -> PlannerRFTObjectiveResult:
    """Reduce consecutive substep rewards into one transition reward.

    The rule is fixed per objective field:

    * sum: `total`, `base_total`, and every `RewardComponents` entry.
    * min: `safety_gate`.

    `total` is the authoritative PPO scalar. With more than one substep it is
    not equal to `base_total * safety_gate`, because the base total is summed
    over substeps while the gate is their minimum.
    """

    if not substeps:
        raise ValueError("aggregate_substep_rewards requires at least one objective result")
    return PlannerRFTObjectiveResult(
        total=aggregate_scalar_rewards(substeps).total,
        base_total=sum(substep.base_total for substep in substeps),
        safety_gate=min(substep.safety_gate for substep in substeps),
        components=RewardComponents(
            ttc=sum(substep.components.ttc for substep in substeps),
            progress=sum(substep.components.progress for substep in substeps),
            comfort=sum(substep.components.comfort for substep in substeps),
            speed=sum(substep.components.speed for substep in substeps),
            energy=sum(substep.components.energy for substep in substeps),
        ),
    )


def recompose_reward_prefix(
    weights: Mapping[str, float],
    components: Sequence[RewardComponents],
    gates: Sequence[float],
    count: int,
) -> PlannerRFTObjectiveResult:
    """Recompose only the executed prefix of padded, stored component scores.

    This is score reuse, not component recalculation. No measurements or missing
    scores are inferred from profile identity or transition-level aggregates.
    Scalar math uses Python floats (double), matching online evaluation and the
    offline adapter's float32-storage -> double-math -> float32-storage boundary.
    """

    if len(components) != len(gates) or not 1 <= count <= len(components):
        raise ValueError("reward prefix requires aligned scores/gates and a valid substep count")
    return aggregate_substep_rewards(
        [compose_plannerrft_objective(weights, components[j], gates[j]) for j in range(count)]
    )


def aggregate_transition_reward(
    results: Sequence[PlannerRFTRewardResult],
) -> PlannerRFTRewardResult:
    """Reduce consecutive substep results into one transition reward.

    Objective scalars come from `aggregate_substep_rewards` (sum/sum/min/sum).
    Diagnostics use explicit rules:

    * sum: additive diagnostics (`route_progress_delta_m`, `step_distance_m`,
      `native_step_energy_ml`, `executed_fuel_proxy_step_energy_ml`).
    * last: `native_episode_energy_ml`, which MetaDrive exposes as the running
      episode cumulative value and therefore must not be summed over substeps.
    * distance-weighted ratio: `executed_fuel_proxy_ml_per_km` is an intensive
      `ml/km` ratio, so it is `sum(intensity_i * distance_i) / sum(distance_i)`
      (equivalently `sum(fuel) / sum(distance) * 1000`), never a plain mean.
    * mean: other intensive diagnostics (`speed_mps`, `speed_limit_mps`,
      `overspeed_mps`, `longitudinal_acceleration_mps2`,
      `lateral_acceleration_mps2`, `jerk_mps3`, `yaw_rate_radps`, `min_ttc_s`).
    * any: `has_ttc_candidate`; all: `energy_distance_valid`.
    * min: gate-like scores (`collision_score`, `drivable_score`,
      `wrong_direction_score`).

    All results must share one `profile_name`; mixing profiles is an error.
    """

    if not results:
        raise ValueError("aggregate_transition_reward requires at least one RewardResult")
    profile_name = results[0].profile_name
    if any(result.profile_name != profile_name for result in results):
        raise ValueError("cannot aggregate results from different reward profiles")
    scalar = aggregate_substep_rewards(results)
    diagnostics = RewardDiagnostics(
        collision_score=min(result.diagnostics.collision_score for result in results),
        drivable_score=min(result.diagnostics.drivable_score for result in results),
        wrong_direction_score=min(result.diagnostics.wrong_direction_score for result in results),
        has_ttc_candidate=any(result.diagnostics.has_ttc_candidate for result in results),
        min_ttc_s=_mean(result.diagnostics.min_ttc_s for result in results),
        route_progress_delta_m=sum(result.diagnostics.route_progress_delta_m for result in results),
        speed_mps=_mean(result.diagnostics.speed_mps for result in results),
        speed_limit_mps=_mean(result.diagnostics.speed_limit_mps for result in results),
        overspeed_mps=_mean(result.diagnostics.overspeed_mps for result in results),
        longitudinal_acceleration_mps2=_mean(
            result.diagnostics.longitudinal_acceleration_mps2 for result in results
        ),
        lateral_acceleration_mps2=_mean(
            result.diagnostics.lateral_acceleration_mps2 for result in results
        ),
        jerk_mps3=_mean(result.diagnostics.jerk_mps3 for result in results),
        yaw_rate_radps=_mean(result.diagnostics.yaw_rate_radps for result in results),
        step_distance_m=sum(result.diagnostics.step_distance_m for result in results),
        native_step_energy_ml=sum(result.diagnostics.native_step_energy_ml for result in results),
        native_episode_energy_ml=results[-1].diagnostics.native_episode_energy_ml,
        executed_fuel_proxy_step_energy_ml=sum(
            result.diagnostics.executed_fuel_proxy_step_energy_ml for result in results
        ),
        executed_fuel_proxy_ml_per_km=_distance_weighted_intensity(results),
        energy_distance_valid=all(result.diagnostics.energy_distance_valid for result in results),
    )
    return PlannerRFTRewardResult(
        profile_name=profile_name,
        total=scalar.total,
        base_total=scalar.base_total,
        safety_gate=scalar.safety_gate,
        components=scalar.components,
        diagnostics=diagnostics,
    )


def energy_only_prefix(
    components: Sequence[RewardComponents], gates: Sequence[float], count: int
) -> PlannerRFTObjectiveResult:
    """Existing diagnostic endpoint: sum of gate times energy score per substep."""

    return recompose_reward_prefix({"energy": 1.0}, components, gates, count)


def _distance_weighted_intensity(results: Sequence[PlannerRFTRewardResult]) -> float:
    distance_m = sum(result.diagnostics.step_distance_m for result in results)
    if distance_m <= 0.0:
        return 0.0
    weighted = sum(
        result.diagnostics.executed_fuel_proxy_ml_per_km * result.diagnostics.step_distance_m
        for result in results
    )
    return weighted / distance_m


def _mean(values: Iterable[float]) -> float:
    materialized = tuple(values)
    return sum(materialized) / len(materialized)


__all__ = [
    "aggregate_scalar_rewards",
    "aggregate_substep_rewards",
    "aggregate_transition_reward",
    "energy_only_prefix",
    "recompose_reward_prefix",
]
