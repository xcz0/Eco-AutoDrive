"""Issue 94 Task C: objective decomposition, normalization attribution, stress gate."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from itertools import combinations, pairwise
from typing import Any, cast

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field, StrictFloat, model_validator

from eco_planner.experiments.lambda_identifiability.diagnostics import (
    COMPONENTS,
    actor_gradients,
    advantage_comparison,
    cosine,
    reward_profile,
    reweight,
    statistics,
)
from eco_planner.rl.optimization.ppo import (
    PPOUpdater,
    _batch_trajectories,
    _normalize_full_batch_advantage,
)
from eco_planner.rl.reward.config import PlannerRFTNoEnergyRewardConfig
from eco_planner.rl.rollout.contracts import RolloutEpisode, concatenate_tensordicts

ADVANTAGE_FORMS = ("raw", "center", "z")
_GRADIENT_GROUPS = ("actor_head", "shared_trunk", "actor", "lateral", "longitudinal")


class ExpectedCalibration(BaseModel):
    """E-034 frozen calibration values used to verify the restored source batch."""

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid", allow_inf_nan=False)
    full_score_delta_m: StrictFloat = Field(gt=0.0)
    longitudinal_acceleration_limit_mps2: StrictFloat = Field(gt=0.0)
    lateral_acceleration_limit_mps2: StrictFloat = Field(gt=0.0)
    jerk_limit_mps3: StrictFloat = Field(gt=0.0)
    yaw_rate_limit_radps: StrictFloat = Field(gt=0.0)


class CalibrationMatchTolerance(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid", allow_inf_nan=False)
    rtol: StrictFloat = Field(ge=0.0)
    atol: StrictFloat = Field(ge=0.0)


class GateThresholds(BaseModel):
    """Issue #94 Gate C engineering thresholds; not claimed as general theory."""

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid", allow_inf_nan=False)
    endpoint_max_actor_head_cosine: StrictFloat = Field(gt=-1.0, lt=1.0)
    min_normalized_advantage_rmse: StrictFloat = Field(ge=0.0)
    min_sign_flip_fraction: StrictFloat = Field(ge=0.0, le=1.0)
    min_stress_fraction_of_endpoint_separation: StrictFloat = Field(gt=0.0, le=1.0)


class DecompositionConfig(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid", allow_inf_nan=False)
    lambdas: list[StrictFloat] = Field(min_length=1)
    quantiles: list[StrictFloat] = Field(min_length=2)
    progress_target_score: StrictFloat = Field(gt=0.0, lt=1.0)
    comfort_target_score: StrictFloat = Field(gt=0.0, lt=1.0)
    calibration_match_tolerance: CalibrationMatchTolerance
    expected_calibration: ExpectedCalibration
    gate: GateThresholds

    @model_validator(mode="after")
    def validate_axes(self) -> DecompositionConfig:
        if any(weight <= 0 for weight in self.lambdas) or sorted(set(self.lambdas)) != self.lambdas:
            raise ValueError("stress lambdas must be positive and strictly increasing")
        if (
            self.quantiles[0] != 0
            or self.quantiles[-1] != 1
            or sorted(set(self.quantiles)) != self.quantiles
        ):
            raise ValueError("quantiles must increase from zero to one")
        return self


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


def _arm_label(kind: str, weight: float | None) -> str:
    if kind == "r0":
        return "r0"
    if kind == "energy_only":
        return "energy_only"
    return f"lambda_{cast(float, weight):g}"


def _rmse(x: np.ndarray, y: np.ndarray) -> float:
    delta = x.astype(np.float64) - y.astype(np.float64)
    return float(np.sqrt(np.mean(np.square(delta))))


def _angular_separation(cosine_value: float) -> float:
    return float(np.arccos(np.clip(cosine_value, -1.0, 1.0)))


def _gradient_comparison(x: np.ndarray, y: np.ndarray) -> dict[str, float | None]:
    norm_x = np.linalg.norm(x.astype(np.float64))
    norm_y = np.linalg.norm(y.astype(np.float64))
    return {
        "cosine": cosine(x, y),
        "norm_ratio_j_over_i": None if norm_x == 0 else float(norm_y / norm_x),
    }


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
    delta = arrays[f"arm_{j}_{key}"].astype(np.float64) - arrays[f"arm_{i}_{key}"]
    return {
        "all": statistics(delta, quantiles),
        "per_scenario": {
            str(slot): statistics(delta[scenario_ids == slot], quantiles)
            for slot in np.unique(scenario_ids)
        },
    }, delta


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
        batch = _batch_trajectories(matched, updater.config)
        if (
            batch.batch_size[0] != updater.config.batch_size
            or len(scenario_ids) != batch.batch_size[0]
        ):
            raise ValueError("diagnostic batch must match the complete configured PPO batch")
        raw = batch["advantage"].detach().clone()
        centered = raw - raw.mean()
        _normalize_full_batch_advantage(batch)
        normalized = batch["advantage"].detach().clone()
        forms = {"raw": raw, "center": centered, "z": normalized}
        arm_gradients: dict[str, dict[str, np.ndarray]] = {}
        actor_loss: dict[str, float] = {}
        for form, advantage in forms.items():
            batch["advantage"] = advantage
            policy.zero_grad(set_to_none=True)
            losses = updater.loss_module(batch.to(updater.device))
            loss = losses["loss_objective"]
            if not torch.isfinite(loss).all():
                raise FloatingPointError("actor loss must be finite")
            loss.backward()
            gradient, layout = actor_gradients(policy)
            arm_gradients[form] = gradient
            actor_loss[form] = float(loss.detach())
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
            "normalized_advantage_rmse": _rmse(
                arrays[f"arm_{i}_normalized_advantage"], arrays[f"arm_{j}_normalized_advantage"]
            ),
            "gradients": {
                group: _gradient_comparison(gradients[i]["z"][group], gradients[j]["z"][group])
                for group in _GRADIENT_GROUPS
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
            "advantage_rmse": _rmse(arrays[f"arm_0_{key}"], arrays[f"arm_{endpoint_index}_{key}"]),
            "gradients": {
                group: _gradient_comparison(
                    gradients[0][form][group], gradients[endpoint_index][form][group]
                )
                for group in _GRADIENT_GROUPS
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


def render_decomposition_report(summary: dict[str, Any]) -> str:
    gate = summary["gate"]
    lines = [
        "# Task C: objective decomposition + normalization attribution",
        "",
        "Reused fixed source batch; actor objective only; no optimizer steps. "
        "Endpoint and stress comparisons use the actor-head gradient.",
        "",
        summary["undefined_reason"],
        "",
        "Component std uses population variance; advantage std uses sample variance.",
        "",
        "| Arm | Reward mean | Reward std | Raw A mean | Raw A std | Z A std | "
        "Head grad norm raw / center / z |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for arm in summary["arms"]:
        norms = arm["gradient_norms"]
        lines.append(
            f"| {arm['label']} | {arm['reward']['mean']:.9g} | {arm['reward']['std']:.9g} | "
            f"{arm['raw_advantage']['mean']:.9g} | {arm['raw_advantage']['std']:.9g} | "
            f"{arm['normalized_advantage']['std']:.9g} | "
            f"{norms['raw']['actor_head']:.9g} / {norms['center']['actor_head']:.9g} / "
            f"{norms['z']['actor_head']:.9g} |"
        )
    endpoint = next(
        pair
        for pair in summary["pairs"]
        if pair["arm_i"] == "r0" and pair["arm_j"] == "energy_only"
    )
    lines += [
        "",
        "## Endpoint attribution: R0 vs Energy-only",
        "",
        "| Advantage form | Pearson | Spearman | Sign flip | Advantage RMSE | "
        "Head cosine | Head norm ratio |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for form, entry in (
        ("z (current preprocessing)", endpoint),
        ("raw", summary["endpoint_forms"]["raw"]),
        ("center-only", summary["endpoint_forms"]["center"]),
    ):
        head = entry["gradients"]["actor_head"]
        rmse = entry.get("normalized_advantage_rmse", entry.get("advantage_rmse"))
        lines.append(
            f"| {form} | {entry['pearson']} | {entry['spearman']} | "
            f"{entry['sign_flip_fraction']} | {rmse:.9g} | {head['cosine']} | "
            f"{head['norm_ratio_j_over_i']} |"
        )
    lines += [
        "",
        "## Stress trajectory: R0 -> finite lambda -> Energy-only",
        "",
        "| Lambda | Head cosine vs R0 | Angular separation (rad) | "
        "Fraction of endpoint separation | Head norm ratio |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for entry in gate["stress"]:
        lines.append(
            f"| {entry['lambda']:g} | {entry['actor_head_cosine']:.9g} | "
            f"{entry['angular_separation_rad']:.9g} | "
            f"{entry['fraction_of_endpoint_separation']:.9g} | "
            f"{entry['norm_ratio_j_over_i']:.9g} |"
        )
    verdict = "PASSED" if gate["gate_c_passed"] else "FAILED"
    attribution = gate["attribution"] or "none"
    lines += [
        "",
        "## Gate C",
        "",
        f"Verdict: **{verdict}**; attribution: `{attribution}`.",
        "",
        f"Endpoint actor-head cosine: {gate['endpoint']['actor_head_cosine']:.9g} "
        f"(threshold {gate['thresholds']['endpoint_max_actor_head_cosine']}); "
        f"normalized-advantage RMSE: {gate['endpoint']['normalized_advantage_rmse']:.9g} "
        f"(threshold {gate['thresholds']['min_normalized_advantage_rmse']}); "
        f"sign-flip fraction: {gate['endpoint']['sign_flip_fraction']:.9g} "
        f"(threshold {gate['thresholds']['min_sign_flip_fraction']}).",
        "",
    ]
    if gate["failure_reasons"]:
        lines.append("Failure reasons:")
        lines.extend(f"- {reason}" for reason in gate["failure_reasons"])
        lines.append("")
    lines += [
        "Full arm and pair statistics, per-form gradients and per-scenario matched "
        "differences: [summary.json](summary.json). Per-transition values and all gradient "
        "vectors: [diagnostics.npz](diagnostics.npz), indexed by "
        "[sample_index.json](sample_index.json).",
        "",
        "These measurements concern this batch and initial policy only. They do not "
        "establish learned behavioral separation, do not select a training reward, and do "
        "not run any optimizer step.",
        "",
    ]
    return "\n".join(lines)


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
