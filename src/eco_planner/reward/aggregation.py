"""Pure aggregation of substep reward results into one transition reward.

The canonical closed-loop cadence gives one PPO transition its own execution
prefix of consecutive simulator substeps. Reward is evaluated per substep and
reduced into a single transition result with the explicit per-field rules in
`aggregate_transition_reward`. This module is pure: it applies no truncation,
repair, or fallback and does not depend on rollout or training state.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from .result import RewardComponents, RewardDiagnostics, RewardResult


def aggregate_transition_reward(results: Sequence[RewardResult]) -> RewardResult:
    """Reduce consecutive substep results into one transition reward.

    Aggregation is fixed per field:

    * sum: `total`, `base_total`, `RewardComponents.*`, and additive
      diagnostics (`route_progress_delta_m`, `step_distance_m`,
      `native_step_energy_ml`, `native_episode_energy_ml`,
      `executed_fuel_proxy_step_energy_ml`).
    * mean: intensive diagnostics (`speed_mps`, `speed_limit_mps`,
      `overspeed_mps`, `longitudinal_acceleration_mps2`,
      `lateral_acceleration_mps2`, `jerk_mps3`, `yaw_rate_radps`, `min_ttc_s`,
      `executed_fuel_proxy_ml_per_km`).
    * any: `has_ttc_candidate`; all: `energy_distance_valid`.
    * min: gate-like scores (`safety_gate`, `collision_score`,
      `drivable_score`, `wrong_direction_score`).

    All results must share one `profile_name`; mixing profiles is an error.
    `total` is the authoritative PPO scalar. With more than one substep it is
    not equal to `base_total * safety_gate`, because the base total is summed
    over substeps while the gate is their minimum.
    """
    if not results:
        raise ValueError("aggregate_transition_reward requires at least one RewardResult")
    profile_name = results[0].profile_name
    if any(result.profile_name != profile_name for result in results):
        raise ValueError("cannot aggregate RewardResults from different reward profiles")

    total = sum(result.total for result in results)
    base_total = sum(result.base_total for result in results)
    safety_gate = min(result.safety_gate for result in results)
    components = RewardComponents(
        ttc=sum(result.components.ttc for result in results),
        progress=sum(result.components.progress for result in results),
        comfort=sum(result.components.comfort for result in results),
        speed=sum(result.components.speed for result in results),
        energy=sum(result.components.energy for result in results),
    )
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
        native_episode_energy_ml=sum(
            result.diagnostics.native_episode_energy_ml for result in results
        ),
        executed_fuel_proxy_step_energy_ml=sum(
            result.diagnostics.executed_fuel_proxy_step_energy_ml for result in results
        ),
        executed_fuel_proxy_ml_per_km=_mean(
            result.diagnostics.executed_fuel_proxy_ml_per_km for result in results
        ),
        energy_distance_valid=all(result.diagnostics.energy_distance_valid for result in results),
    )
    return RewardResult(
        profile_name=profile_name,
        total=total,
        base_total=base_total,
        safety_gate=safety_gate,
        components=components,
        diagnostics=diagnostics,
    )


def _mean(values: Iterable[float]) -> float:
    materialized = tuple(values)
    return sum(materialized) / len(materialized)


__all__ = ["aggregate_transition_reward"]
