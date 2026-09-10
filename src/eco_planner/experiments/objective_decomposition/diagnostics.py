"""Issue 94 Task C: objective decomposition, normalization attribution, stress gate."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations, pairwise
from typing import Any, cast

import numpy as np
import torch

from eco_planner.analysis.statistics import (
    advantage_comparison,
    gradient_comparison,
    paired_difference,
    statistics,
)
from eco_planner.analysis.statistics import rmse as rmse
from eco_planner.experiments.fixed_batch.gradients import (
    ADVANTAGE_FORMS,
    GRADIENT_GROUPS,
    actor_backward,
)
from eco_planner.experiments.fixed_batch.rewards import (
    COMPONENTS,
    energy_only_reward,
    reward_profile,
    reweight,
)
from eco_planner.experiments.objective_decomposition.config import (
    GateThresholds,
)
from eco_planner.rl.optimization import (
    PPOUpdater,
    build_ppo_batch,
    normalize_full_batch_advantage,
)
from eco_planner.rl.reward.config import PlannerRFTNoEnergyRewardConfig
from eco_planner.rl.rollout.contracts import RolloutEpisode, concatenate_tensordicts


def _arm_label(kind: str, weight: float | None) -> str:
    if kind == "r0":
        return "r0"
    if kind == "energy_only":
        return "energy_only"
    return f"lambda_{cast(float, weight):g}"


def _angular_separation(cosine_value: float) -> float:
    return float(np.arccos(np.clip(cosine_value, -1.0, 1.0)))


def _arm_episodes(
    episodes: Sequence[RolloutEpisode],
    base: PlannerRFTNoEnergyRewardConfig,
    kind: str,
    weight: float | None,
) -> tuple[RolloutEpisode, ...]:
    if kind == "r0":
        return tuple(reweight(episode, base) for episode in episodes)
    if kind == "energy_only":
        return tuple(energy_only_reward(episode) for episode in episodes)
    profile = reward_profile(base, cast(float, weight))
    return tuple(reweight(episode, profile) for episode in episodes)


def _pair_difference(
    arrays: dict[str, np.ndarray],
    scenario_ids: np.ndarray,
    quantiles: Sequence[float],
    i: int,
    j: int,
    key: str,
) -> tuple[dict, np.ndarray]:
    return paired_difference(
        arrays[f"arm_{i}_{key}"], arrays[f"arm_{j}_{key}"], scenario_ids, quantiles
    )


def analyze_decomposition(
    updater: PPOUpdater,
    episodes: Sequence[RolloutEpisode],
    base: PlannerRFTNoEnergyRewardConfig,
    lambdas: Sequence[float],
    quantiles: Sequence[float],
    scenario_ids: np.ndarray,
    gate: GateThresholds,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Backward-only R0 / stress-lambda / Energy-only decomposition on one fixed batch."""
    policy = updater.policy
    original = {name: p.detach().clone() for name, p in policy.state_dict().items()}
    audit = concatenate_tensordicts([episode.audit for episode in episodes])
    arms_spec: list[tuple[str, float | None]] = [
        ("r0", None),
        *[("lambda", float(weight)) for weight in lambdas],
        ("energy_only", None),
    ]
    arrays: dict[str, np.ndarray] = {"scenario_index": scenario_ids}
    summary: dict[str, Any] = {
        "sample_count": len(scenario_ids),
        "optimizer_steps": 0,
        "advantage_forms": list(ADVANTAGE_FORMS),
        "undefined_reason": "Zero-norm vectors have undefined cosine; zero denominators have "
        "undefined norm ratios. A zero-initialized actor head blocks trunk actor gradients.",
        "components": {},
        "arms": [],
        "pairs": [],
    }
    for key in (*[f"reward_component_{name}" for name in COMPONENTS], "reward_safety_gate"):
        arrays[key] = audit[key].numpy().reshape(-1)
        summary["components"][key] = statistics(arrays[key], quantiles)
    gradients: list[dict[str, dict[str, np.ndarray]]] = []
    layout: list[dict] | None = None
    for index, (kind, weight) in enumerate(arms_spec):
        matched = _arm_episodes(episodes, base, kind, weight)
        batch = build_ppo_batch(matched, updater.config)
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
        for form, advantage in forms.items():
            batch["advantage"] = advantage
            loss, gradient, layout = actor_backward(updater, batch, batch["advantage"])
            arm_gradients[form] = gradient
            actor_loss[form] = loss
        gradients.append(arm_gradients)
        summary["actor_parameter_layout"] = layout
        if any(not torch.equal(p, original[name]) for name, p in policy.state_dict().items()):
            raise RuntimeError("backward-only diagnostic changed policy state")
        reward = torch.cat([episode.training["next", "reward"] for episode in matched])
        profile = (
            base
            if kind == "r0"
            else reward_profile(base, cast(float, weight))
            if kind == "lambda"
            else None
        )
        arm: dict[str, Any] = {
            "index": index,
            "kind": kind,
            "label": _arm_label(kind, weight),
            "actor_loss": actor_loss,
            "gradient_norms": {
                form: {
                    group: float(np.linalg.norm(value.astype(np.float64)))
                    for group, value in gradient.items()
                }
                for form, gradient in arm_gradients.items()
            },
        }
        if profile is None:
            arm["objective"] = "safety_gate * reward_component_energy"
        else:
            arm["reward_profile"] = profile.model_dump(mode="json")
        if kind == "lambda":
            arm["lambda"] = weight
        values = {
            "reward": reward.cpu().numpy().reshape(-1),
            "raw_advantage": raw.numpy().reshape(-1).copy(),
            "center_advantage": centered.numpy().reshape(-1).copy(),
            "normalized_advantage": normalized.numpy().reshape(-1).copy(),
            "value_target": batch["value_target"].cpu().numpy().reshape(-1).copy(),
        }
        for key, value in values.items():
            arrays[f"arm_{index}_{key}"] = value
            arm[key] = statistics(value, quantiles, ddof=1 if "advantage" in key else 0)
        for form, gradient in arm_gradients.items():
            for group, value in gradient.items():
                arrays[f"arm_{index}_gradient_{form}_{group}"] = value
        summary["arms"].append(arm)
    policy.zero_grad(set_to_none=True)
    endpoint_index = len(arms_spec) - 1
    for i, j in combinations(range(len(arms_spec)), 2):
        pair: dict[str, Any] = {
            "arm_i": _arm_label(*arms_spec[i]),
            "arm_j": _arm_label(*arms_spec[j]),
            "lambda_j": arms_spec[j][1] if arms_spec[j][0] == "lambda" else None,
            "form": "z",
            **advantage_comparison(
                arrays[f"arm_{i}_normalized_advantage"], arrays[f"arm_{j}_normalized_advantage"]
            ),
            "normalized_advantage_rmse": rmse(
                arrays[f"arm_{i}_normalized_advantage"], arrays[f"arm_{j}_normalized_advantage"]
            ),
            "gradients": {
                group: gradient_comparison(gradients[i]["z"][group], gradients[j]["z"][group])
                for group in GRADIENT_GROUPS
            },
            "matched_differences": {},
        }
        for key in ("reward", "raw_advantage", "normalized_advantage"):
            difference, delta = _pair_difference(arrays, scenario_ids, quantiles, i, j, key)
            arrays[f"pair_{i}_{j}_{key}_delta"] = delta
            pair["matched_differences"][key] = difference
        summary["pairs"].append(pair)
    endpoint_forms: dict[str, Any] = {}
    for form, key in (("raw", "raw_advantage"), ("center", "center_advantage")):
        comparison = advantage_comparison(
            arrays[f"arm_0_{key}"], arrays[f"arm_{endpoint_index}_{key}"]
        )
        _, delta = _pair_difference(arrays, scenario_ids, quantiles, 0, endpoint_index, key)
        if form == "center":
            arrays["endpoint_center_advantage_delta"] = delta
        endpoint_forms[form] = {
            **comparison,
            "advantage_rmse": rmse(arrays[f"arm_0_{key}"], arrays[f"arm_{endpoint_index}_{key}"]),
            "gradients": {
                group: gradient_comparison(
                    gradients[0][form][group], gradients[endpoint_index][form][group]
                )
                for group in GRADIENT_GROUPS
            },
        }
    summary["endpoint_forms"] = endpoint_forms
    z_pairs = {pair["arm_j"]: pair for pair in summary["pairs"] if pair["arm_i"] == "r0"}
    summary["gate"] = evaluate_gate(
        endpoint_pair=z_pairs[_arm_label(*arms_spec[endpoint_index])],
        endpoint_forms=endpoint_forms,
        stress_pairs=[
            z_pairs[_arm_label(*arms_spec[index])] for index in range(1, len(lambdas) + 1)
        ],
        thresholds=gate,
    )
    if updater.completed_optimizer_steps != 0:
        raise RuntimeError("diagnostic performed an optimizer step")
    return summary, arrays


def evaluate_gate(
    endpoint_pair: dict[str, Any],
    endpoint_forms: dict[str, Any],
    stress_pairs: list[dict[str, Any]],
    thresholds: GateThresholds,
) -> dict[str, Any]:
    """Apply Issue #94 Gate C thresholds and attribution rules to measured pair metrics."""

    def head_cosine(pair: dict[str, Any]) -> float:
        value = pair["gradients"]["actor_head"]["cosine"]
        if value is None:
            raise RuntimeError("actor-head gradient cosine is undefined for the gate")
        return float(value)

    endpoint_cosine = head_cosine(endpoint_pair)
    endpoint_rmse = float(endpoint_pair["normalized_advantage_rmse"])
    endpoint_sign_flip = float(endpoint_pair["sign_flip_fraction"])
    endpoint_identifiable = endpoint_cosine <= thresholds.endpoint_max_actor_head_cosine and (
        endpoint_rmse >= thresholds.min_normalized_advantage_rmse
        or endpoint_sign_flip >= thresholds.min_sign_flip_fraction
    )
    endpoint_separation = _angular_separation(endpoint_cosine)
    stress: list[dict[str, Any]] = []
    for pair in stress_pairs:
        value = head_cosine(pair)
        separation = _angular_separation(value)
        stress.append(
            {
                "lambda": pair["lambda_j"],
                "actor_head_cosine": value,
                "angular_separation_rad": separation,
                "fraction_of_endpoint_separation": separation / endpoint_separation,
                "norm_ratio_j_over_i": pair["gradients"]["actor_head"]["norm_ratio_j_over_i"],
            }
        )
    separations = [entry["angular_separation_rad"] for entry in stress]
    nondecreasing = all(b >= a for a, b in pairwise(separations))
    increases_overall = separations[-1] > separations[0]
    reaches_fraction = any(
        entry["fraction_of_endpoint_separation"]
        >= thresholds.min_stress_fraction_of_endpoint_separation
        for entry in stress
    )
    failure_reasons: list[str] = []
    if endpoint_cosine > thresholds.endpoint_max_actor_head_cosine:
        failure_reasons.append("endpoint actor-head cosine above threshold")
    if (
        endpoint_rmse < thresholds.min_normalized_advantage_rmse
        and endpoint_sign_flip < thresholds.min_sign_flip_fraction
    ):
        failure_reasons.append(
            "endpoint normalized-advantage RMSE and sign-flip fraction below thresholds"
        )
    if not nondecreasing:
        failure_reasons.append("stress angular distance is not nondecreasing in lambda")
    if not increases_overall:
        failure_reasons.append("stress angular distance does not increase overall")
    if not reaches_fraction:
        failure_reasons.append("no stress arm reaches the required fraction of endpoint separation")

    def separable(form: str) -> bool:
        value = endpoint_forms[form]["gradients"]["actor_head"]["cosine"]
        return value is not None and float(value) <= thresholds.endpoint_max_actor_head_cosine

    if endpoint_identifiable:
        attribution = (
            "relative_scale_lambda_parameterization_too_weak" if not reaches_fraction else None
        )
    elif separable("raw") or separable("center"):
        attribution = "normalization_suppressed_identifiability"
    else:
        attribution = "objective_batch_collinearity"
    return {
        "thresholds": thresholds.model_dump(),
        "endpoint": {
            "actor_head_cosine": endpoint_cosine,
            "normalized_advantage_rmse": endpoint_rmse,
            "sign_flip_fraction": endpoint_sign_flip,
            "angular_separation_rad": endpoint_separation,
            "identifiable": endpoint_identifiable,
        },
        "stress": stress,
        "stress_angular_distance_nondecreasing": nondecreasing,
        "stress_angular_distance_increases_overall": increases_overall,
        "stress_reaches_fraction_of_endpoint_separation": reaches_fraction,
        "failure_reasons": failure_reasons,
        "gate_c_passed": not failure_reasons,
        "attribution": attribution,
    }
