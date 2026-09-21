"""RL rollout adapter for reward calibration, reweighting, and offline rescoring.

Pure reward math, configuration, calibration, and objective scalarization are
owned by ``eco_planner.reward``. This adapter reads and writes ``RolloutEpisode``
audit / training TensorDicts: it extracts measurement arrays, delegates the
numeric work to ``eco_planner.reward``, and writes component scores and scalar
reward back into the episode.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import cast

import numpy as np
import torch
from tensordict import cat

from eco_planner.reward import (
    MOTION_LIMITS,
    EnergyBandConfig,
    PlannerRFTEnergyRewardConfig,
    PlannerRFTNoEnergyRewardConfig,
    RewardProfileConfig,
    apply_safety_gate,
    combine_component_scores,
    energy_band_thresholds,
    scored_arrays,
)
from eco_planner.reward.components.energy import calibrated_band_score
from eco_planner.reward.config import EnergyRewardConfig
from eco_planner.rl import RolloutEpisode

COMPONENTS = ("ttc", "progress", "comfort", "speed", "energy")


def reward_profile(base: PlannerRFTNoEnergyRewardConfig, weight: float) -> RewardProfileConfig:
    if weight == 0:
        return base
    payload = base.model_dump()
    payload["name"] = "plannerrft_energy_v1"
    payload["weights"]["energy"] = weight
    return PlannerRFTEnergyRewardConfig.model_validate(payload)


def reweight(episode: RolloutEpisode, profile: RewardProfileConfig) -> RolloutEpisode:
    # Recompose fixed audited scores; no environment or component calibration is rerun.
    audit = episode.audit.clone()
    weights = profile.weights.model_dump()
    components = {
        name: cast(torch.Tensor, audit[f"reward_component_{name}"].double()) for name in weights
    }
    base = cast(torch.Tensor, combine_component_scores(weights, components))
    guarded = apply_safety_gate(base, audit["reward_safety_gate"].double())
    total = cast(torch.Tensor, guarded).float()
    audit["reward_base_total"] = base.float()
    audit["reward_total"] = total
    training = episode.training.clone()
    training["next", "reward"] = total.to(training["next", "reward"])
    return replace(episode, training=training, audit=audit, reward_profile=profile.name)


def energy_only_reward(episode: RolloutEpisode) -> RolloutEpisode:
    """Objective endpoint: safety gate times the audited energy score; diagnostic only."""
    audit = episode.audit.clone()
    energy = audit["reward_component_energy"].double()
    guarded = apply_safety_gate(energy, audit["reward_safety_gate"].double())
    total = cast(torch.Tensor, guarded).float()
    audit["reward_base_total"] = energy.float()
    audit["reward_total"] = total
    training = episode.training.clone()
    training["next", "reward"] = total.to(training["next", "reward"])
    return replace(episode, training=training, audit=audit)


def raw_arrays(episodes: list[RolloutEpisode]) -> dict[str, np.ndarray]:
    audit = cat([e.audit for e in episodes])
    return {
        key: audit[key].numpy().astype(np.float64).reshape(-1)
        for key in ("route_progress_delta_m", *MOTION_LIMITS)
    }


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
    full, zero = energy_band_thresholds(intensity, band)
    payload = calibrated.model_dump()
    payload["energy"]["mode"] = "calibrated_band"
    payload["energy"]["band_full_score_ml_per_km"] = full
    payload["energy"]["band_zero_score_ml_per_km"] = zero
    return PlannerRFTNoEnergyRewardConfig.model_validate(payload)


__all__ = [
    "COMPONENTS",
    "apply_energy_band",
    "energy_only_reward",
    "raw_arrays",
    "rescore",
    "rescore_energy",
    "reward_profile",
    "reweight",
    "verify_original_components",
]
