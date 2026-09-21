from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import numpy as np
import torch
from tensordict import cat

from eco_planner.reward import PlannerRFTNoEnergyRewardConfig, component_score, score_delta
from eco_planner.reward.components.energy import calibrated_band_score
from eco_planner.reward.config import EnergyRewardConfig
from eco_planner.rl import RolloutEpisode
from eco_planner.rl.reward.calibration_config import CalibrationTargets, EnergyBandConfig
from eco_planner.rl.reward.reweighting import reweight

MOTION_LIMITS = {
    "longitudinal_acceleration_mps2": "longitudinal_acceleration_limit_mps2",
    "lateral_acceleration_mps2": "lateral_acceleration_limit_mps2",
    "jerk_mps3": "jerk_limit_mps3",
    "yaw_rate_radps": "yaw_rate_limit_radps",
}


def raw_arrays(episodes: list[RolloutEpisode]) -> dict[str, np.ndarray]:
    audit = cat([e.audit for e in episodes])
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


def rescore_energy(episode: RolloutEpisode, energy: EnergyRewardConfig) -> RolloutEpisode:
    """Replace the audited energy score with the calibrated-band rescore."""
    if energy.mode != "calibrated_band":
        return episode
    full = energy.band_full_score_ml_per_km
    zero = energy.band_zero_score_ml_per_km
    assert full is not None and zero is not None
    audit = episode.audit.clone()
    key = audit["reward_component_energy"]
    intensity = audit["executed_fuel_proxy_ml_per_km"].numpy().astype(np.float64).reshape(-1)
    valid = audit["energy_distance_valid"].numpy().astype(bool).reshape(-1)
    scores = np.asarray(
        [
            calibrated_band_score(x, full, zero) if ok else 0.0
            for x, ok in zip(intensity, valid, strict=True)
        ]
    )
    audit["reward_component_energy"] = torch.from_numpy(scores).reshape_as(key).to(key)
    return replace(episode, audit=audit)


def rescore(episode: RolloutEpisode, profile: PlannerRFTNoEnergyRewardConfig) -> RolloutEpisode:
    scores = scored_arrays(raw_arrays([episode]), profile)
    audit = episode.audit.clone()
    for name in ("progress", "comfort"):
        key = f"reward_component_{name}"
        audit[key] = torch.from_numpy(scores[name]).reshape_as(audit[key]).to(audit[key])
    episode = rescore_energy(replace(episode, audit=audit), profile.energy)
    return reweight(episode, profile)


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


def apply_energy_band(
    calibrated: PlannerRFTNoEnergyRewardConfig,
    episodes: Sequence[RolloutEpisode],
    band: EnergyBandConfig,
) -> PlannerRFTNoEnergyRewardConfig:
    """Derive the energy representation from this batch's intensity distribution."""
    audit = cat([episode.audit for episode in episodes])
    intensity = audit["executed_fuel_proxy_ml_per_km"].numpy().astype(np.float64).reshape(-1)
    full = float(np.quantile(intensity, band.full_score_intensity_quantile))
    zero = float(np.quantile(intensity, band.zero_score_intensity_quantile))
    payload = calibrated.model_dump()
    payload["energy"]["mode"] = "calibrated_band"
    payload["energy"]["band_full_score_ml_per_km"] = full
    payload["energy"]["band_zero_score_ml_per_km"] = zero
    return PlannerRFTNoEnergyRewardConfig.model_validate(payload)
