from __future__ import annotations

import numpy as np

from eco_planner.analysis.statistics import statistics
from eco_planner.rl.reward.config import PlannerRFTNoEnergyRewardConfig


def dynamic_range_audit(
    raw: dict[str, np.ndarray],
    base: PlannerRFTNoEnergyRewardConfig,
    calibrated: PlannerRFTNoEnergyRewardConfig,
    quantiles: list[float],
    original_scores: dict[str, np.ndarray],
    calibrated_scores: dict[str, np.ndarray],
    motion_limits: dict[str, str],
    scenario_ids: np.ndarray,
    cycle_ids: np.ndarray,
) -> tuple[dict, dict[str, np.ndarray]]:
    before, after = original_scores, calibrated_scores
    arrays = {**raw, "scenario_index": scenario_ids, "planning_cycle_index": cycle_ids}
    arrays.update(
        {
            f"{label}_{k}": v
            for label, s in (("original", before), ("calibrated", after))
            for k, v in s.items()
        }
    )

    def describe(mask: np.ndarray) -> dict:
        result: dict = {"sample_count": int(mask.sum()), "raw": {}, "scores": {}}
        for key, value in raw.items():
            measured = np.abs(value) if key in motion_limits else value
            result["raw"][key] = statistics(measured[mask], quantiles)
            if key in motion_limits:
                limit = getattr(base.comfort, motion_limits[key])
                result["raw"][key]["original_limit_exceeded_fraction"] = float(
                    np.mean(measured[mask] > limit)
                )
        for label, scores in (("original", before), ("calibrated", after)):
            result["scores"][label] = {
                key: {
                    **statistics(value[mask], quantiles),
                    "zero_fraction": float(np.mean(value[mask] == 0)),
                    "one_fraction": float(np.mean(value[mask] == 1)),
                    **(
                        {
                            "minimum_fraction_including_ties": float(
                                np.mean(value[mask] == scores["comfort"][mask])
                            )
                        }
                        if key in motion_limits
                        else {}
                    ),
                }
                for key, value in scores.items()
            }
        return result

    return {
        "all": describe(np.ones(len(scenario_ids), dtype=bool)),
        "per_scenario": {str(i): describe(scenario_ids == i) for i in np.unique(scenario_ids)},
        "per_planning_cycle": {str(i): describe(cycle_ids == i) for i in np.unique(cycle_ids)},
        "interpretation": (
            "Distribution-relative smoothness; original limits remain audit references, "
            "not redefined physical comfort standards. All transitions retained."
        ),
    }, arrays
