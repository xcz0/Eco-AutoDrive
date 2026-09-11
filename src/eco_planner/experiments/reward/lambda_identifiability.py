"""Configuration, diagnostics and execution for lambda identifiability."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field, StrictFloat, model_validator

from eco_planner.analysis import (
    advantage_comparison,
    cosine,
    paired_difference,
    publish,
    statistics,
)
from eco_planner.artifacts import write_json, write_npz
from eco_planner.configuration import load_resolved_yaml_mapping
from eco_planner.experiments.reward.fixed_batch import (
    COMPONENTS,
    actor_backward,
    load_fixed_batch,
    restore_runtime,
    reward_profile,
    reweight,
    verify_original_components,
    write_runtime_metadata,
)
from eco_planner.rl import (
    PlannerRFTNoEnergyRewardConfig,
    PPOUpdater,
    RolloutEpisode,
    build_ppo_batch,
    concatenate_tensordicts,
    normalize_full_batch_advantage,
)


class IdentifiabilityConfig(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid", allow_inf_nan=False)
    lambdas: list[StrictFloat] = Field(min_length=2)
    quantiles: list[StrictFloat] = Field(min_length=2)

    @model_validator(mode="after")
    def validate_axes(self) -> IdentifiabilityConfig:
        if self.lambdas[0] != 0 or sorted(set(self.lambdas)) != self.lambdas:
            raise ValueError("lambdas must start at zero and be strictly increasing")
        if (
            self.quantiles[0] != 0
            or self.quantiles[-1] != 1
            or sorted(set(self.quantiles)) != self.quantiles
        ):
            raise ValueError("quantiles must increase from zero to one")
        return self


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


def run(source: Path, config_path: Path, output: Path, *, figures: bool = True) -> dict[str, Any]:
    study = IdentifiabilityConfig.model_validate(load_resolved_yaml_mapping(config_path))
    batch = load_fixed_batch(source)
    base = PlannerRFTNoEnergyRewardConfig.model_validate(batch.resolved_config["reward"])
    verify_original_components(batch.episodes, base)
    runtime = restore_runtime(source, batch)
    output.mkdir(parents=True, exist_ok=False)
    OmegaConf.save(OmegaConf.create(study.model_dump()), output / "diagnostic_config.yaml")
    OmegaConf.save(OmegaConf.create(batch.resolved_config), output / "resolved_config.yaml")
    write_json(output / "sample_index.json", {"samples": batch.samples})
    write_runtime_metadata(output, source, batch, runtime)
    summary, arrays = analyze(
        runtime.updater, batch.episodes, base, study.lambdas, study.quantiles, batch.scenario_ids
    )
    runtime.verify_unchanged()
    summary.update(
        {
            "status": "completed",
            "source_batch": str(source.resolve()),
            "initial_policy_hash": runtime.initial_policy_hash,
            "policy_unchanged": True,
            "episode_count": len(batch.episodes),
            "batch_source": "reused fixed source batch",
        }
    )
    write_npz(output / "diagnostics.npz", arrays)
    write_json(output / "summary.json", summary)
    publish("lambda-identifiability", output, output, figures=figures)
    return {
        "status": "completed",
        "output_dir": str(output),
        "sample_count": len(batch.samples),
        "optimizer_steps": 0,
    }
