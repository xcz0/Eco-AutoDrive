from __future__ import annotations

import numpy as np

from eco_planner.analysis.statistics import statistics
from eco_planner.experiments.fixed_batch.calibration import MOTION_LIMITS, scored_arrays
from eco_planner.experiments.reward_calibration.config import CalibrationConfig
from eco_planner.rl.reward.config import PlannerRFTNoEnergyRewardConfig


def dynamic_range_audit(
    raw: dict[str, np.ndarray],
    base: PlannerRFTNoEnergyRewardConfig,
    calibrated: PlannerRFTNoEnergyRewardConfig,
    study: CalibrationConfig,
    scenario_ids: np.ndarray,
    cycle_ids: np.ndarray,
) -> tuple[dict, dict[str, np.ndarray]]:
    before, after = scored_arrays(raw, base), scored_arrays(raw, calibrated)
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
            measured = np.abs(value) if key in MOTION_LIMITS else value
            result["raw"][key] = statistics(measured[mask], study.quantiles)
            if key in MOTION_LIMITS:
                limit = getattr(base.comfort, MOTION_LIMITS[key])
                result["raw"][key]["original_limit_exceeded_fraction"] = float(
                    np.mean(measured[mask] > limit)
                )
        for label, scores in (("original", before), ("calibrated", after)):
            result["scores"][label] = {
                key: {
                    **statistics(value[mask], study.quantiles),
                    "zero_fraction": float(np.mean(value[mask] == 0)),
                    "one_fraction": float(np.mean(value[mask] == 1)),
                    **(
                        {
                            "minimum_fraction_including_ties": float(
                                np.mean(value[mask] == scores["comfort"][mask])
                            )
                        }
                        if key in MOTION_LIMITS
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
