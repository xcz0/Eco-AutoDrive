from __future__ import annotations

from statistics import median
from typing import Any

from .config import GateConfig


def evaluate_update_gate(metrics: dict[str, Any] | None, gate: GateConfig) -> dict[str, Any]:
    """Evaluate update gate conditions 1-6 plus the under-update classification."""

    if metrics is None:
        return {
            "under_update": None,
            "conditions": {f"c{index}": False for index in range(1, 7)},
            "passed_1_6": False,
            "failure_reasons": ["training_run_failed"],
        }
    kl = metrics["post_update_kl"]
    kl_median = metrics["post_update_kl_median"]
    tail = kl[-gate.runaway_tail_updates :]
    tail_median = median(tail)
    runaway = tail_median > gate.runaway_tail_factor * max(kl_median, gate.kl_floor)
    within_fraction = sum(1 for value in kl if value <= gate.target_kl) / len(kl)
    conditions = {
        "c1_median_kl_above_floor": kl_median >= gate.kl_floor,
        "c2_kl_stable_within_target": (
            within_fraction >= gate.within_target_kl_fraction and not runaway
        ),
        "c3_policy_ratio_moves": metrics["policy_ratio_change"] >= gate.ratio_change_floor,
        "c4_guidance_rms_shift": (
            metrics["probe_guidance_rms_shift"] >= gate.guidance_rms_shift_floor
        ),
        "c5_no_beta_boundary_collapse": (
            metrics["min_beta_alpha"] >= gate.beta_parameter_floor
            and metrics["min_beta_beta"] >= gate.beta_parameter_floor
            and metrics["probe_boundary_mass_max_after"] <= gate.boundary_mass_ceiling
        ),
        "c6_no_behavioral_collapse": (
            metrics["behavior"]["collision_count"] + metrics["behavior"]["out_of_road_count"]
            <= gate.collision_budget
            and metrics["behavior"]["episode_length_last_median"]
            >= gate.episode_length_retention_floor
            * metrics["behavior"]["episode_length_first_median"]
        ),
    }
    under_update = (
        kl_median < gate.kl_floor
        and metrics["policy_ratio_change"] < gate.ratio_change_floor
        and metrics["probe_guidance_rms_shift"] < gate.guidance_rms_shift_floor
    )
    return {
        "under_update": under_update,
        "conditions": conditions,
        "passed_1_6": all(conditions.values()),
        "post_update_kl_within_target_fraction": within_fraction,
        "post_update_kl_tail_median": tail_median,
        "kl_runaway": runaway,
        "failure_reasons": [name for name, passed in conditions.items() if not passed],
    }


def evaluate_heldout_change(
    initial: dict[str, Any], final: dict[str, Any], gate: GateConfig
) -> dict[str, Any]:
    """Evaluate update gate condition 7: held-out change beyond matched evaluation noise."""

    relative: dict[str, Any] = {}
    exceeds: dict[str, Any] = {}
    for name in gate.heldout_metrics:
        initial_value = initial[name]
        final_value = final[name]
        if initial_value is None or final_value is None or initial_value == 0.0:
            relative[name] = None
            exceeds[name] = False
            continue
        relative[name] = abs(final_value - initial_value) / abs(initial_value)
        exceeds[name] = relative[name] >= gate.heldout_relative_change_floor
    return {
        "relative_change": relative,
        "exceeds_noise": any(exceeds.values()),
        "exceeds_noise_metrics": [name for name, value in exceeds.items() if value],
    }
