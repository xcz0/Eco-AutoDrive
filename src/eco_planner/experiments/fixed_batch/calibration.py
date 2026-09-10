from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import numpy as np
import torch

from eco_planner.experiments.fixed_batch.config import CalibrationGuardSource, CalibrationTargets
from eco_planner.experiments.fixed_batch.rewards import reweight
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


def verify_original_components(
    episodes: list[RolloutEpisode], base: PlannerRFTNoEnergyRewardConfig
) -> None:
    for episode in episodes:
        rebuilt = rescore(episode, base)
        for key in (
            "reward_component_progress",
            "reward_component_comfort",
            "reward_base_total",
            "reward_total",
        ):
            torch.testing.assert_close(rebuilt.audit[key], episode.audit[key], rtol=1e-6, atol=1e-7)


_EXPECTED_CALIBRATION_FIELDS = (
    ("progress.full_score_delta_m", "full_score_delta_m"),
    (
        "comfort.longitudinal_acceleration_limit_mps2",
        "longitudinal_acceleration_limit_mps2",
    ),
    ("comfort.lateral_acceleration_limit_mps2", "lateral_acceleration_limit_mps2"),
    ("comfort.jerk_limit_mps3", "jerk_limit_mps3"),
    ("comfort.yaw_rate_limit_radps", "yaw_rate_limit_radps"),
)


def verify_expected_calibration(
    calibrated: PlannerRFTNoEnergyRewardConfig, study: CalibrationGuardSource
) -> dict[str, Any]:
    """Guard the source-batch provenance against the E-034 frozen calibration values.

    The re-collected rollout matches the E-033 protocol and hashes but not bitwise, so
    raw kinematic medians drift by up to ~0.5%; the tolerance catches a wrong source
    batch or protocol drift, not float noise.
    """
    checks: dict[str, dict[str, float]] = {}
    for field, attribute in _EXPECTED_CALIBRATION_FIELDS:
        section, _, name = field.partition(".")
        actual = float(getattr(getattr(calibrated, section), name))
        expected = float(getattr(study.expected_calibration, attribute))
        if not math.isclose(
            actual,
            expected,
            rel_tol=study.calibration_match_tolerance.rtol,
            abs_tol=study.calibration_match_tolerance.atol,
        ):
            raise ValueError(
                f"source batch calibration is inconsistent with the E-034 frozen value for "
                f"{field}: {actual!r} vs {expected!r}"
            )
        checks[field] = {"actual": actual, "expected": expected}
    return dict(checks)
