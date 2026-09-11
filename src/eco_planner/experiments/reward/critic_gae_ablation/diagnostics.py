"""Configuration and pure diagnostics for critic gae ablation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field, StrictFloat, model_validator
from tensordict import TensorDictBase

from eco_planner.analysis import advantage_comparison, gradient_comparison, rmse, statistics
from eco_planner.experiments.reward.fixed_batch.config import (
    CalibrationMatchTolerance,
    ExpectedCalibration,
)
from eco_planner.experiments.reward.fixed_batch.gradients import (
    ADVANTAGE_FORMS,
    GRADIENT_GROUPS,
    actor_backward,
)
from eco_planner.experiments.reward.fixed_batch.rewards import energy_only_reward, reweight
from eco_planner.rl import (
    PPO_BATCH_KEYS,
    PlannerRFTNoEnergyRewardConfig,
    PPOConfig,
    PPOUpdater,
    RolloutEpisode,
    build_ppo_batch,
    concatenate_tensordicts,
    normalize_full_batch_advantage,
)


class AttributionThresholds(BaseModel):
    """Issue #94 Gate C endpoint thresholds reused by the C4 attribution rules."""

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid", allow_inf_nan=False)
    endpoint_max_actor_head_cosine: StrictFloat = Field(gt=-1.0, lt=1.0)
    min_normalized_advantage_rmse: StrictFloat = Field(ge=0.0)
    min_sign_flip_fraction: StrictFloat = Field(ge=0.0, le=1.0)


class AblationConfig(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid", allow_inf_nan=False)
    quantiles: list[StrictFloat] = Field(min_length=2)
    progress_target_score: StrictFloat = Field(gt=0.0, lt=1.0)
    comfort_target_score: StrictFloat = Field(gt=0.0, lt=1.0)
    calibration_match_tolerance: CalibrationMatchTolerance
    expected_calibration: ExpectedCalibration
    reference_match_tolerance: CalibrationMatchTolerance
    gate: AttributionThresholds

    @model_validator(mode="after")
    def validate_axes(self) -> AblationConfig:
        if (
            self.quantiles[0] != 0
            or self.quantiles[-1] != 1
            or sorted(set(self.quantiles)) != self.quantiles
        ):
            raise ValueError("quantiles must increase from zero to one")
        return self


CREDIT_FORMS = ("standard_gae", "reward_only_gae", "discounted_return")


ARM_LABELS = ("r0", "energy_only")


def zero_critic_values(episode: RolloutEpisode) -> RolloutEpisode:
    """Reward-only GAE input: V(s)=V(s')=0, which also removes the tail bootstrap."""
    training = episode.training.clone()
    training["state_value"] = torch.zeros_like(training["state_value"])
    training["next", "state_value"] = torch.zeros_like(training["next", "state_value"])
    return replace(episode, training=training)


def discounted_return_batch(episodes: Sequence[RolloutEpisode], gamma: float) -> TensorDictBase:
    """Critic-free diagnostic: per-episode R_t = r_t + gamma * R_{t+1}, no bootstrap."""
    trajectories = []
    for episode in episodes:
        reward = episode.training["next", "reward"]
        returns = torch.zeros_like(reward)
        running = torch.zeros_like(reward[0])
        for step in reversed(range(reward.shape[0])):
            running = reward[step] + gamma * running
            returns[step] = running
        trajectory = episode.training.clone()
        trajectory["advantage"] = returns.clone()
        trajectory["value_target"] = returns.clone()
        trajectories.append(trajectory)
    return concatenate_tensordicts(trajectories).select(*PPO_BATCH_KEYS)


def credit_batch(
    episodes: Sequence[RolloutEpisode], config: PPOConfig, form: str
) -> TensorDictBase:
    """Build the full PPO batch for one temporal-credit form; episode boundaries shared."""
    if form == "standard_gae":
        return build_ppo_batch(episodes, config)
    if form == "reward_only_gae":
        return build_ppo_batch([zero_critic_values(episode) for episode in episodes], config)
    if form == "discounted_return":
        return discounted_return_batch(episodes, config.gamma)
    raise ValueError(f"unknown temporal-credit form: {form}")


def analyze_critic_gae_ablation(
    updater: PPOUpdater,
    episodes: Sequence[RolloutEpisode],
    base: PlannerRFTNoEnergyRewardConfig,
    quantiles: Sequence[float],
    scenario_ids: np.ndarray,
    gate: AttributionThresholds,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Backward-only R0 vs Energy-only comparison under three temporal-credit forms."""
    policy = updater.policy
    original = {name: p.detach().clone() for name, p in policy.state_dict().items()}
    matched = {
        "r0": tuple(reweight(episode, base) for episode in episodes),
        "energy_only": tuple(energy_only_reward(episode) for episode in episodes),
    }
    arrays: dict[str, np.ndarray] = {"scenario_index": scenario_ids}
    summary: dict[str, Any] = {
        "sample_count": len(scenario_ids),
        "optimizer_steps": 0,
        "advantage_forms": list(ADVANTAGE_FORMS),
        "credit_forms": list(CREDIT_FORMS),
        "undefined_reason": "Zero-norm vectors have undefined cosine; zero denominators have "
        "undefined norm ratios. A zero-initialized actor head blocks trunk actor gradients.",
        "arms": [],
        "pairs": [],
    }
    gradients: dict[tuple[str, str], dict[str, dict[str, np.ndarray]]] = {}
    layout: list[dict] | None = None
    for label in ARM_LABELS:
        arm_episodes = matched[label]
        reward = torch.cat([episode.training["next", "reward"] for episode in arm_episodes])
        reward_array = reward.cpu().numpy().reshape(-1)
        arrays[f"arm_{label}_reward"] = reward_array
        arm: dict[str, Any] = {
            "label": label,
            "reward": statistics(reward_array, quantiles),
            "credit_forms": {},
        }
        if label == "r0":
            arm["reward_profile"] = base.model_dump(mode="json")
        else:
            arm["objective"] = "safety_gate * reward_component_energy"
        for form in CREDIT_FORMS:
            batch = credit_batch(arm_episodes, updater.config, form)
            if (
                batch.batch_size[0] != updater.config.batch_size
                or len(scenario_ids) != batch.batch_size[0]
            ):
                raise ValueError("diagnostic batch must match the complete configured PPO batch")
            raw = batch["advantage"].detach().clone()
            centered = raw - raw.mean()
            normalize_full_batch_advantage(batch)
            normalized = batch["advantage"].detach().clone()
            forms = {"raw": raw, "center": centered, "z": normalized}
            arm_gradients: dict[str, dict[str, np.ndarray]] = {}
            actor_loss: dict[str, float] = {}
            for advantage_form, advantage in forms.items():
                batch["advantage"] = advantage
                loss, gradient, layout = actor_backward(updater, batch, batch["advantage"])
                arm_gradients[advantage_form] = gradient
                actor_loss[advantage_form] = loss
            gradients[(label, form)] = arm_gradients
            summary["actor_parameter_layout"] = layout
            if any(not torch.equal(p, original[name]) for name, p in policy.state_dict().items()):
                raise RuntimeError("backward-only diagnostic changed policy state")
            form_entry: dict[str, Any] = {
                "actor_loss": actor_loss,
                "gradient_norms": {
                    advantage_form: {
                        group: float(np.linalg.norm(value.astype(np.float64)))
                        for group, value in gradient.items()
                    }
                    for advantage_form, gradient in arm_gradients.items()
                },
            }
            values = {
                "raw_advantage": raw.numpy().reshape(-1).copy(),
                "center_advantage": centered.numpy().reshape(-1).copy(),
                "normalized_advantage": normalized.numpy().reshape(-1).copy(),
                "value_target": batch["value_target"].cpu().numpy().reshape(-1).copy(),
            }
            for key, value in values.items():
                arrays[f"arm_{label}__{form}__{key}"] = value
                form_entry[key] = statistics(value, quantiles, ddof=1)
            for advantage_form, gradient in arm_gradients.items():
                for group, value in gradient.items():
                    arrays[f"arm_{label}__{form}__gradient_{advantage_form}_{group}"] = value
            arm["credit_forms"][form] = form_entry
        summary["arms"].append(arm)
    policy.zero_grad(set_to_none=True)
    for form in CREDIT_FORMS:
        pair: dict[str, Any] = {
            "arm_i": "r0",
            "arm_j": "energy_only",
            "credit_form": form,
            "form": "z",
            **advantage_comparison(
                arrays[f"arm_r0__{form}__normalized_advantage"],
                arrays[f"arm_energy_only__{form}__normalized_advantage"],
            ),
            "normalized_advantage_rmse": rmse(
                arrays[f"arm_r0__{form}__normalized_advantage"],
                arrays[f"arm_energy_only__{form}__normalized_advantage"],
            ),
            "gradients": {
                group: gradient_comparison(
                    gradients[("r0", form)]["z"][group],
                    gradients[("energy_only", form)]["z"][group],
                )
                for group in GRADIENT_GROUPS
            },
            "advantage_forms": {},
        }
        for advantage_form, key in (("raw", "raw_advantage"), ("center", "center_advantage")):
            pair["advantage_forms"][advantage_form] = {
                **advantage_comparison(
                    arrays[f"arm_r0__{form}__{key}"],
                    arrays[f"arm_energy_only__{form}__{key}"],
                ),
                "advantage_rmse": rmse(
                    arrays[f"arm_r0__{form}__{key}"],
                    arrays[f"arm_energy_only__{form}__{key}"],
                ),
                "gradients": {
                    group: gradient_comparison(
                        gradients[("r0", form)][advantage_form][group],
                        gradients[("energy_only", form)][advantage_form][group],
                    )
                    for group in GRADIENT_GROUPS
                },
            }
        summary["pairs"].append(pair)
    if updater.completed_optimizer_steps != 0:
        raise RuntimeError("diagnostic performed an optimizer step")
    summary["attribution"] = evaluate_attribution(summary["pairs"], gate)
    return summary, arrays


def _endpoint_identifiable(pair: dict[str, Any], thresholds: AttributionThresholds) -> bool:
    value = pair["gradients"]["actor_head"]["cosine"]
    if value is None:
        raise RuntimeError("actor-head gradient cosine is undefined for the attribution gate")
    return float(value) <= thresholds.endpoint_max_actor_head_cosine and (
        float(pair["normalized_advantage_rmse"]) >= thresholds.min_normalized_advantage_rmse
        or float(pair["sign_flip_fraction"]) >= thresholds.min_sign_flip_fraction
    )


def evaluate_attribution(
    pairs: Sequence[dict[str, Any]], thresholds: AttributionThresholds
) -> dict[str, Any]:
    """Apply the Issue #94 C4 attribution rules to the measured endpoint pairs."""
    by_form = {pair["credit_form"]: pair for pair in pairs}
    if set(by_form) != set(CREDIT_FORMS):
        raise ValueError("attribution requires exactly one endpoint pair per credit form")
    endpoint = {
        form: {
            "actor_head_cosine": float(by_form[form]["gradients"]["actor_head"]["cosine"]),
            "normalized_advantage_rmse": float(by_form[form]["normalized_advantage_rmse"]),
            "sign_flip_fraction": float(by_form[form]["sign_flip_fraction"]),
            "identifiable": _endpoint_identifiable(by_form[form], thresholds),
        }
        for form in CREDIT_FORMS
    }
    standard_ok = endpoint["standard_gae"]["identifiable"]
    reward_only_ok = endpoint["reward_only_gae"]["identifiable"]
    return_only_ok = endpoint["discounted_return"]["identifiable"]
    if standard_ok:
        attribution = None
    elif reward_only_ok:
        attribution = "critic_gae_common_term_dominated"
    elif return_only_ok:
        attribution = "temporal_credit_structure_sensitivity"
    else:
        attribution = "reward_batch_collinearity"
    return {
        "thresholds": thresholds.model_dump(),
        "endpoint": endpoint,
        "gate_c_endpoint_identifiable_under_standard_gae": standard_ok,
        "attribution": attribution,
        "decision": "attribution only; Task D (guidance control authority) follows regardless "
        "of the branch, and this ablation never modifies the PPO training definition",
    }
