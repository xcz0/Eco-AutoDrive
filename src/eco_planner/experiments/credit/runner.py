"""Compose reward arms and credit ablations over a single immutable rollout batch."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf

from eco_planner.analysis import publish
from eco_planner.analysis.statistics import advantage_comparison, gradient_comparison, rmse
from eco_planner.artifacts import write_json, write_npz
from eco_planner.configuration import load_resolved_yaml_mapping
from eco_planner.reward import calibrate
from eco_planner.reward.config import PlannerRFTNoEnergyRewardConfig
from eco_planner.rl.optimization import PPOUpdater
from eco_planner.rl.optimization.credit import credit_batch
from eco_planner.rl.optimization.diagnostic_runtime import restore_runtime, write_runtime_metadata
from eco_planner.rl.optimization.gradients import GRADIENT_GROUPS, diagnostic_variants
from eco_planner.rl.reward import (
    apply_energy_band,
    energy_only_reward,
    raw_arrays,
    rescore,
    reward_profile,
    reweight,
    verify_original_components,
)
from eco_planner.rl.rollout.contracts import RolloutEpisode
from eco_planner.rl.rollout.fixed_batch import load_fixed_batch

from .config import CreditStudyConfig
from .decisions import evaluate_attribution, evaluate_gate


def measure(
    updater: PPOUpdater,
    episodes: Sequence[RolloutEpisode],
    base: PlannerRFTNoEnergyRewardConfig,
    study: CreditStudyConfig,
    scenario_ids: np.ndarray,
) -> tuple[dict, dict]:
    arrays: dict[str, np.ndarray] = {"scenario_index": scenario_ids}
    measurements: dict[tuple[str, str], Any] = {}
    arms, pairs = [], []
    for arm in study.arms:
        matched = (
            [energy_only_reward(e) for e in episodes]
            if arm.weight == "energy_only"
            else [reweight(e, reward_profile(base, arm.weight)) for e in episodes]
        )
        arrays[f"{arm.label}__reward"] = np.concatenate(
            [e.training["next", "reward"].cpu().numpy().reshape(-1) for e in matched]
        )
        entry = {"label": arm.label, "weight": arm.weight, "actor_losses": {}}
        for credit in study.credit_forms:
            values, gradients, losses, layout = diagnostic_variants(
                updater, credit_batch(matched, updater.config, credit), tuple(study.advantage_forms)
            )
            if len(values["raw_advantage"]) != len(scenario_ids):
                raise ValueError("sample index differs from diagnostic batch")
            measurements[(arm.label, credit)] = (values, gradients)
            for key, value in values.items():
                arrays[f"{arm.label}__{credit}__{key}"] = value
            for form, groups in gradients.items():
                for group, value in groups.items():
                    arrays[f"{arm.label}__{credit}__gradient_{form}_{group}"] = value
            entry["actor_losses"][credit] = losses
        arms.append(entry)
    ordered = sorted(study.arms, key=lambda arm: (arm.label != "r0", arm.label == "energy_only"))
    for first, second in combinations(ordered, 2):
        for credit in study.credit_forms:
            a, ga = measurements[(first.label, credit)]
            b, gb = measurements[(second.label, credit)]
            pair = {
                "arm_i": first.label,
                "arm_j": second.label,
                "credit_form": credit,
                "lambda_j": second.weight,
                "advantage_forms": {},
            }
            for form in study.advantage_forms:
                key = {
                    "raw": "raw_advantage",
                    "center": "center_advantage",
                    "z": "normalized_advantage",
                }[form]
                result = {
                    **advantage_comparison(a[key], b[key]),
                    "advantage_rmse": rmse(a[key], b[key]),
                    "gradients": {
                        group: gradient_comparison(ga[form][group], gb[form][group])
                        for group in GRADIENT_GROUPS
                    },
                }
                pair["advantage_forms"][form] = result
                if form == "z":
                    pair.update(result)
                    pair["normalized_advantage_rmse"] = result["advantage_rmse"]
            pairs.append(pair)
    decisions = {}
    endpoints = [p for p in pairs if p["arm_i"] == "r0" and p["arm_j"] == "energy_only"]
    if study.objective_gate:
        endpoint = next(p for p in endpoints if p["credit_form"] == "standard_gae")
        stress = sorted(
            [
                p
                for p in pairs
                if p["arm_i"] == "r0"
                and isinstance(p["lambda_j"], float)
                and p["lambda_j"] > 0
                and p["credit_form"] == "standard_gae"
            ],
            key=lambda p: p["lambda_j"],
        )
        decisions["objective"] = evaluate_gate(
            endpoint, endpoint["advantage_forms"], stress, study.objective_gate
        )
    if study.attribution_gate:
        decisions["attribution"] = evaluate_attribution(endpoints, study.attribution_gate)
    return {
        "arms": arms,
        "pairs": pairs,
        "decisions": decisions,
        "actor_parameter_layout": layout,
        "gradient_groups": list(GRADIENT_GROUPS),
        "advantage_forms": study.advantage_forms,
        "credit_forms": study.credit_forms,
        "value_target_ddof": study.value_target_ddof,
    }, arrays


def run(source: Path, config_path: Path, output: Path, *, figures: bool = True) -> dict:
    study = CreditStudyConfig.model_validate(load_resolved_yaml_mapping(config_path))
    batch = load_fixed_batch(source)
    base = PlannerRFTNoEnergyRewardConfig.model_validate(batch.resolved_config["reward"])
    verify_original_components(batch.episodes, base)
    if study.calibration:
        base = calibrate(raw_arrays(batch.episodes), base, study.calibration)
    if study.energy_band:
        base = apply_energy_band(base, batch.episodes, study.energy_band)
    episodes = [rescore(e, base) for e in batch.episodes]
    runtime = restore_runtime(source, batch)
    summary, arrays = measure(runtime.updater, episodes, base, study, batch.scenario_ids)
    runtime.verify_unchanged()
    summary.update(
        {
            "status": "completed",
            "kind": "credit",
            "source_batch": str(source.resolve()),
            "initial_policy_hash": runtime.initial_policy_hash,
            "policy_unchanged": True,
            "sample_count": len(batch.samples),
            "quantiles": study.quantiles,
            "optimizer_steps": 0,
            "reward_profile": base.model_dump(),
            "interpretation": "Fixed-batch actor diagnostics do not establish learned behavior.",
        }
    )
    output.mkdir(parents=True, exist_ok=False)
    OmegaConf.save(OmegaConf.create(study.model_dump()), output / "diagnostic_config.yaml")
    write_json(output / "sample_index.json", {"samples": batch.samples})
    write_runtime_metadata(output, source, batch, runtime)
    write_npz(output / "diagnostics.npz", arrays)
    write_json(output / "summary.json", summary)
    publish("credit", output, output, figures=figures)
    return {"status": "completed", "output_dir": str(output), "optimizer_steps": 0}
