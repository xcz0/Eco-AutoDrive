"""Fixed-batch distribution calibration and physical-threshold audit for Issue 94."""

from __future__ import annotations

from dataclasses import replace
from typing import Protocol

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field, StrictFloat, model_validator

from eco_planner.experiments.lambda_identifiability.diagnostics import reweight, statistics
from eco_planner.rl.reward.components.comfort import component_score
from eco_planner.rl.reward.components.progress import score_delta
from eco_planner.rl.reward.config import PlannerRFTNoEnergyRewardConfig
from eco_planner.rl.rollout.contracts import RolloutEpisode, concatenate_tensordicts

MOTION_LIMITS = {
    "longitudinal_acceleration_mps2": "longitudinal_acceleration_limit_mps2",
    "lateral_acceleration_mps2": "lateral_acceleration_limit_mps2",
    "jerk_mps3": "jerk_limit_mps3",
    "yaw_rate_radps": "yaw_rate_limit_radps",
}


class CalibrationTargets(Protocol):
    """Distribution target scores consumed by the fixed calibration rule."""

    progress_target_score: float
    comfort_target_score: float


class CalibrationConfig(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid", allow_inf_nan=False)
    progress_target_score: StrictFloat = Field(gt=0.0, lt=1.0)
    comfort_target_score: StrictFloat = Field(gt=0.0, lt=1.0)
    lambdas: list[StrictFloat] = Field(min_length=2)
    quantiles: list[StrictFloat] = Field(min_length=2)

    @model_validator(mode="after")
    def validate_axes(self) -> CalibrationConfig:
        if self.lambdas[0] != 0 or sorted(set(self.lambdas)) != self.lambdas:
            raise ValueError("lambdas must start at zero and strictly increase")
        if (
            self.quantiles[0] != 0
            or self.quantiles[-1] != 1
            or sorted(set(self.quantiles)) != self.quantiles
        ):
            raise ValueError("quantiles must increase from zero to one")
        return self


def raw_arrays(episodes: list[RolloutEpisode]) -> dict[str, np.ndarray]:
    audit = concatenate_tensordicts([e.audit for e in episodes])
    return {
        key: audit[key].numpy().astype(np.float64).reshape(-1)
        for key in ("route_progress_delta_m", *MOTION_LIMITS)
    }


def scored_arrays(
    raw: dict[str, np.ndarray], profile: PlannerRFTNoEnergyRewardConfig
) -> dict[str, np.ndarray]:
    result = {
        "progress": np.asarray(
            [
                score_delta(x, profile.progress.full_score_delta_m)
                for x in raw["route_progress_delta_m"]
            ]
        )
    }
    for key, field in MOTION_LIMITS.items():
        result[key] = np.asarray(
            [component_score(abs(x), getattr(profile.comfort, field)) for x in raw[key]]
        )
    result["comfort"] = np.min([result[key] for key in MOTION_LIMITS], axis=0)
    return result


def calibrate(
    raw: dict[str, np.ndarray],
    base: PlannerRFTNoEnergyRewardConfig,
    study: CalibrationTargets,
) -> PlannerRFTNoEnergyRewardConfig:
    for value in raw.values():
        if value.size == 0 or not np.isfinite(value).all():
            raise ValueError("calibration requires nonempty finite motion arrays")
    positive = raw["route_progress_delta_m"][raw["route_progress_delta_m"] > 0]
    if not positive.size:
        raise ValueError("progress calibration requires positive route progress")
    payload = base.model_dump()
    payload["progress"]["full_score_delta_m"] = float(
        np.median(positive) / study.progress_target_score
    )
    old_scores = scored_arrays(raw, base)
    for key, field in MOTION_LIMITS.items():
        if np.any(old_scores[key] == 0):
            payload["comfort"][field] = max(
                getattr(base.comfort, field),
                float(np.median(np.abs(raw[key])) / (2 - study.comfort_target_score)),
            )
    return PlannerRFTNoEnergyRewardConfig.model_validate(payload)


def rescore(episode: RolloutEpisode, profile: PlannerRFTNoEnergyRewardConfig) -> RolloutEpisode:
    scores = scored_arrays(raw_arrays([episode]), profile)
    audit = episode.audit.clone()
    for name in ("progress", "comfort"):
        key = f"reward_component_{name}"
        audit[key] = torch.from_numpy(scores[name]).reshape_as(audit[key]).to(audit[key])
    return reweight(replace(episode, audit=audit), profile)


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
