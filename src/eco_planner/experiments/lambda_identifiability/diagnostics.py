"""Matched reward, advantage and actor-gradient measurements without policy updates."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations
from typing import Any

import numpy as np
import torch

from eco_planner.analysis.statistics import (
    advantage_comparison,
    cosine,
    paired_difference,
    statistics,
)
from eco_planner.experiments.fixed_batch.gradients import (
    actor_backward,
)
from eco_planner.experiments.fixed_batch.rewards import COMPONENTS, reward_profile, reweight
from eco_planner.rl.optimization import (
    PPOUpdater,
    build_ppo_batch,
    normalize_full_batch_advantage,
)
from eco_planner.rl.reward.config import (
    PlannerRFTNoEnergyRewardConfig,
)
from eco_planner.rl.rollout.contracts import RolloutEpisode, concatenate_tensordicts


def analyze(
    updater: PPOUpdater,
    episodes: Sequence[RolloutEpisode],
    base: PlannerRFTNoEnergyRewardConfig,
    lambdas: Sequence[float],
    quantiles: Sequence[float],
    scenario_ids: np.ndarray,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    policy = updater.policy
    original = {name: p.detach().clone() for name, p in policy.state_dict().items()}
    audit = concatenate_tensordicts([episode.audit for episode in episodes])
    arrays = {"scenario_index": scenario_ids}
    summary: dict[str, Any] = {
        "sample_count": len(scenario_ids),
        "optimizer_steps": 0,
        "decision": "continuous_diagnostics_only",
        "undefined_reason": "Zero-norm vectors have undefined cosine; zero denominators have "
        "undefined norm ratios. A zero-initialized actor head blocks trunk actor gradients.",
        "components": {},
        "arms": [],
        "pairs": [],
    }
    for key in (*[f"reward_component_{name}" for name in COMPONENTS], "reward_safety_gate"):
        arrays[key] = audit[key].numpy().reshape(-1)
        summary["components"][key] = statistics(arrays[key], quantiles)
    gradients = []
    for index, weight in enumerate(lambdas):
        profile = reward_profile(base, weight)
        matched = tuple(reweight(episode, profile) for episode in episodes)
        batch = build_ppo_batch(matched, updater.config)
        if (
            batch.batch_size[0] != updater.config.batch_size
            or len(scenario_ids) != batch.batch_size[0]
        ):
            raise ValueError("diagnostic batch must match the complete configured PPO batch")
        raw = batch["advantage"].detach().cpu().numpy().reshape(-1).copy()
        normalize_full_batch_advantage(batch)
        norm = batch["advantage"].detach().cpu().numpy().reshape(-1).copy()
        reward = torch.cat([e.training["next", "reward"] for e in matched])
        values = {
            "reward": reward.cpu().numpy().reshape(-1),
            "raw_advantage": raw,
            "normalized_advantage": norm,
            "value_target": batch["value_target"].cpu().numpy().reshape(-1),
        }
        loss, gradient, layout = actor_backward(updater, batch, batch["advantage"])
        gradients.append(gradient)
        summary["actor_parameter_layout"] = layout
        arm = {
            "lambda": weight,
            "reward_profile": profile.model_dump(mode="json"),
            "actor_loss": loss,
            "gradient_norms": {
                k: float(np.linalg.norm(v.astype(np.float64))) for k, v in gradient.items()
            },
        }
        for key, value in values.items():
            arrays[f"arm_{index}_{key}"] = value
            arm[key] = statistics(value, quantiles, ddof=1 if "advantage" in key else 0)
        for key, value in gradient.items():
            arrays[f"arm_{index}_gradient_{key}"] = value
        summary["arms"].append(arm)
        if any(not torch.equal(p, original[name]) for name, p in policy.state_dict().items()):
            raise RuntimeError("backward-only diagnostic changed policy state")
    policy.zero_grad(set_to_none=True)
    for i, j in combinations(range(len(lambdas)), 2):
        pair = {
            "lambda_i": lambdas[i],
            "lambda_j": lambdas[j],
            **advantage_comparison(
                arrays[f"arm_{i}_normalized_advantage"], arrays[f"arm_{j}_normalized_advantage"]
            ),
            "gradients": {},
            "matched_differences": {},
        }
        for group in gradients[i]:
            x, y = gradients[i][group], gradients[j][group]
            nx = np.linalg.norm(x.astype(np.float64))
            pair["gradients"][group] = {
                "cosine": cosine(x, y),
                "norm_ratio_j_over_i": None
                if nx == 0
                else float(np.linalg.norm(y.astype(np.float64)) / nx),
            }
        for key in ("reward", "raw_advantage", "normalized_advantage"):
            difference, delta = paired_difference(
                arrays[f"arm_{i}_{key}"], arrays[f"arm_{j}_{key}"], scenario_ids, quantiles
            )
            arrays[f"pair_{i}_{j}_{key}_delta"] = delta
            pair["matched_differences"][key] = difference
        summary["pairs"].append(pair)
    if updater.completed_optimizer_steps != 0:
        raise RuntimeError("diagnostic performed an optimizer step")
    return summary, arrays
