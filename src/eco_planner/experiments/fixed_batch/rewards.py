from __future__ import annotations

from dataclasses import replace
from typing import cast

import torch

from eco_planner.rl.reward.config import (
    PlannerRFTEnergyRewardConfig,
    PlannerRFTNoEnergyRewardConfig,
    RewardProfileConfig,
)
from eco_planner.rl.rollout.contracts import RolloutEpisode

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
    base = cast(
        torch.Tensor,
        (
            sum(
                audit[f"reward_component_{name}"].double() * weight
                for name, weight in weights.items()
            )
            / profile.weights.total
        ),
    )
    total = (base * audit["reward_safety_gate"].double()).float()
    audit["reward_base_total"] = base.float()
    audit["reward_total"] = total
    training = episode.training.clone()
    training["next", "reward"] = total.to(training["next", "reward"])
    return replace(episode, training=training, audit=audit, reward_profile=profile.name)


def energy_only_reward(episode: RolloutEpisode) -> RolloutEpisode:
    """Objective endpoint: safety gate times the audited energy score; diagnostic only."""
    audit = episode.audit.clone()
    energy = audit["reward_component_energy"].double()
    total = (energy * audit["reward_safety_gate"].double()).float()
    audit["reward_base_total"] = energy.float()
    audit["reward_total"] = total
    training = episode.training.clone()
    training["next", "reward"] = total.to(training["next", "reward"])
    return replace(episode, training=training, audit=audit)
