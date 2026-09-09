"""Matched reward, advantage and actor-gradient measurements without policy updates."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from itertools import combinations
from typing import Any, cast

import numpy as np
import torch

from eco_planner.rl.optimization.ppo import (
    PPOUpdater,
    _batch_trajectories,
    _normalize_full_batch_advantage,
)
from eco_planner.rl.policy import ExplorationPolicy
from eco_planner.rl.reward.config import (
    PlannerRFTEnergyRewardConfig,
    PlannerRFTNoEnergyRewardConfig,
    RewardProfileConfig,
)
from eco_planner.rl.rollout.contracts import RolloutEpisode, concatenate_tensordicts

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


def statistics(value: np.ndarray, quantiles: Sequence[float], *, ddof: int = 0) -> dict:
    x = np.asarray(value, dtype=np.float64).reshape(-1)
    if x.size <= ddof or not np.isfinite(x).all():
        raise ValueError("statistics require enough finite samples")
    return {
        "mean": float(x.mean()),
        "std": float(x.std(ddof=ddof)),
        "quantiles": {
            str(q): float(v) for q, v in zip(quantiles, np.quantile(x, quantiles), strict=True)
        },
    }


def _ranks(x: np.ndarray) -> np.ndarray:
    _, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    return (np.cumsum(counts) - (counts - 1) / 2)[inverse]


def cosine(x: np.ndarray, y: np.ndarray) -> float | None:
    x, y = x.astype(np.float64), y.astype(np.float64)
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    return None if denominator == 0 else float(np.dot(x, y) / denominator)


def advantage_comparison(x: np.ndarray, y: np.ndarray) -> dict:
    x, y = x.astype(np.float64).reshape(-1), y.astype(np.float64).reshape(-1)
    rx, ry = _ranks(x), _ranks(y)
    return {
        "pearson": cosine(x - x.mean(), y - y.mean()),
        "spearman": cosine(rx - rx.mean(), ry - ry.mean()),
        "sign_flip_fraction": float(np.mean(x * y < 0)),
        "zero_fraction_i": float(np.mean(x == 0)),
        "zero_fraction_j": float(np.mean(y == 0)),
    }


def actor_gradients(policy: ExplorationPolicy) -> tuple[dict[str, np.ndarray], list[dict]]:
    groups: dict[str, list[np.ndarray]] = {key: [] for key in ("actor_head", "shared_trunk")}
    layout = []
    offset = 0
    for name, parameter in policy.named_parameters():
        if name.startswith("value_head."):
            if parameter.grad is not None:
                raise RuntimeError("actor backward reached value head")
            continue
        if parameter.grad is None:
            raise RuntimeError(f"actor parameter has no gradient: {name}")
        value = parameter.grad.detach().cpu().float().numpy().reshape(-1).copy()
        if not np.isfinite(value).all():
            raise FloatingPointError(f"nonfinite actor gradient: {name}")
        group = "actor_head" if name.startswith("actor_head.") else "shared_trunk"
        groups[group].append(value)
        layout.append(
            {"name": name, "shape": list(parameter.shape), "offset": offset, "size": value.size}
        )
        offset += value.size
    result = {key: np.concatenate(values) for key, values in groups.items()}
    result["actor"] = np.concatenate(
        [
            cast(torch.Tensor, parameter.grad).detach().cpu().float().numpy().reshape(-1)
            for name, parameter in policy.named_parameters()
            if not name.startswith("value_head.")
        ]
    )
    # forward_tensors views the four rows as [lateral/longitudinal, alpha/beta].
    for name, rows in (("lateral", slice(0, 2)), ("longitudinal", slice(2, 4))):
        result[name] = np.concatenate(
            [
                cast(torch.Tensor, parameter.grad)[rows].detach().cpu().float().numpy().reshape(-1)
                for parameter in policy.actor_head.parameters()
            ]
        )
    return result, layout


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
        batch = _batch_trajectories(matched, updater.config)
        if (
            batch.batch_size[0] != updater.config.batch_size
            or len(scenario_ids) != batch.batch_size[0]
        ):
            raise ValueError("diagnostic batch must match the complete configured PPO batch")
        raw = batch["advantage"].detach().cpu().numpy().reshape(-1).copy()
        _normalize_full_batch_advantage(batch)
        norm = batch["advantage"].detach().cpu().numpy().reshape(-1).copy()
        reward = torch.cat([e.training["next", "reward"] for e in matched])
        values = {
            "reward": reward.cpu().numpy().reshape(-1),
            "raw_advantage": raw,
            "normalized_advantage": norm,
            "value_target": batch["value_target"].cpu().numpy().reshape(-1),
        }
        policy.zero_grad(set_to_none=True)
        losses = updater.loss_module(batch.to(updater.device))
        loss = losses["loss_objective"]
        if not torch.isfinite(loss).all():
            raise FloatingPointError("actor loss must be finite")
        loss.backward()
        gradient, layout = actor_gradients(policy)
        gradients.append(gradient)
        summary["actor_parameter_layout"] = layout
        arm = {
            "lambda": weight,
            "reward_profile": profile.model_dump(mode="json"),
            "actor_loss": float(loss.detach()),
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
            delta = arrays[f"arm_{j}_{key}"].astype(np.float64) - arrays[f"arm_{i}_{key}"]
            arrays[f"pair_{i}_{j}_{key}_delta"] = delta
            pair["matched_differences"][key] = {
                "all": statistics(delta, quantiles),
                "per_scenario": {
                    str(slot): statistics(delta[scenario_ids == slot], quantiles)
                    for slot in np.unique(scenario_ids)
                },
            }
        summary["pairs"].append(pair)
    if updater.completed_optimizer_steps != 0:
        raise RuntimeError("diagnostic performed an optimizer step")
    return summary, arrays
