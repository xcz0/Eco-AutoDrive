"""Offline critic / temporal-credit attribution over saved training runs.

Every measurement is backward-only: it reconstructs the on-policy rollout of one
saved update, restores that update's pre-update policy, and compares the actor
gradient produced by the current standard GAE against a critic-free credit form.
No optimizer step is performed and the source training evidence is never written.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

from eco_planner._repository import REPOSITORY_ROOT
from eco_planner.analysis import publish
from eco_planner.analysis.statistics import advantage_comparison, gradient_comparison
from eco_planner.artifacts import collect_repository_metadata, write_json, write_npz
from eco_planner.configuration import load_resolved_yaml_mapping
from eco_planner.rl import (
    ExplorationPolicy,
    PPOUpdater,
    load_exploration_policy_checkpoint,
    parse_training_config,
    read_rollout_episode,
)
from eco_planner.rl.artifacts import policy_state_hash
from eco_planner.rl.optimization.credit import credit_batch
from eco_planner.rl.optimization.gradients import GRADIENT_GROUPS, diagnostic_variants

from .diagnostics import AttributionRun, CriticAttributionConfig, evaluate_materiality

_CROSS_ARM_GROUPS = ("actor_head", "lateral", "longitudinal")


def _load_update_episodes(run_dir: Path, update_index: int):
    update_dir = run_dir / "updates" / f"update-{update_index:03d}"
    paths = sorted(update_dir.glob("slot-*-episode-*.npz"))
    if not paths:
        raise ValueError(f"update {update_index} has no persisted rollout episodes: {run_dir}")
    return [read_rollout_episode(path) for path in paths]


def _pre_update_checkpoint(run_dir: Path, update_index: int) -> Path:
    if update_index == 0:
        return run_dir / "policy-initial.pt"
    return run_dir / f"policy-update-{update_index - 1:03d}.pt"


def _close(value: float, reference: float, rtol: float) -> bool:
    return math.isclose(value, reference, rel_tol=rtol, abs_tol=1e-6)


def _measure_run(
    run: AttributionRun,
    source: Path,
    study: CriticAttributionConfig,
    arrays: dict[str, np.ndarray],
    samples: list[dict[str, Any]],
    gradient_store: dict[tuple[str, int, int], dict[str, np.ndarray]],
) -> dict:
    run_dir = (source / run.path).resolve()
    resolved = load_resolved_yaml_mapping(run_dir / "resolved_config.yaml")
    config = parse_training_config(OmegaConf.create(resolved))
    source_summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    if source_summary.get("status") != "completed":
        raise ValueError(f"source run is not completed: {run.label}")
    if source_summary.get("reward_profile") != run.reward_profile:
        raise ValueError(f"source reward profile differs from declared arm: {run.label}")
    if source_summary.get("training_seed") != run.training_seed:
        raise ValueError(f"source training seed differs from declared run: {run.label}")
    update_count = config.training.update_count
    if len(source_summary["updates"]) != update_count:
        raise ValueError(f"source update count differs from resolved config: {run.label}")
    if any(index >= update_count for index in study.update_indices):
        raise ValueError("update index exceeds the source training budget")
    credit_forms = (study.baseline_credit_form, *study.comparison_credit_forms)
    updates = []
    for index in study.update_indices:
        episodes = _load_update_episodes(run_dir, index)
        transition_count = sum(episode.transition_count for episode in episodes)
        if transition_count != config.ppo.batch_size:
            raise ValueError(
                f"reconstructed batch size {transition_count} differs from the configured "
                f"{config.ppo.batch_size} at {run.label} update {index}"
            )
        checkpoint = _pre_update_checkpoint(run_dir, index)
        policy = ExplorationPolicy(config.policy)
        load_exploration_policy_checkpoint(checkpoint, policy)
        updater = PPOUpdater(policy, config.ppo)
        before_hash = policy_state_hash(policy)
        measurements: dict[str, dict[str, Any]] = {}
        prefix = f"{run.label}__update{index:03d}"
        for credit in credit_forms:
            values, gradients, losses, _layout = diagnostic_variants(
                updater, credit_batch(episodes, config.ppo, credit), (study.advantage_form,)
            )
            measurements[credit] = {
                "raw_advantage": values["raw_advantage"],
                "gradients": {
                    group: gradients[study.advantage_form][group] for group in GRADIENT_GROUPS
                },
                "loss": losses[study.advantage_form],
            }
            arrays[f"{prefix}__{credit}__raw_advantage"] = values["raw_advantage"]
            arrays[f"{prefix}__{credit}__value_target"] = values["value_target"]
        if policy_state_hash(policy) != before_hash or updater.completed_optimizer_steps != 0:
            raise RuntimeError("critic attribution changed the policy or performed an update")
        baseline = measurements[study.baseline_credit_form]
        recorded = source_summary["updates"][index]
        provenance = {
            "advantage_mean": float(baseline["raw_advantage"].mean()),
            "advantage_std": float(baseline["raw_advantage"].std(ddof=1)),
            "recorded_advantage_mean": recorded["raw_advantage_mean"],
            "recorded_advantage_std": recorded["raw_advantage_std"],
        }
        if not _close(
            provenance["advantage_mean"], recorded["raw_advantage_mean"], study.provenance_rtol
        ) or not _close(
            provenance["advantage_std"], recorded["raw_advantage_std"], study.provenance_rtol
        ):
            raise ValueError(
                f"reconstructed advantage differs from recorded source at "
                f"{run.label} update {index}"
            )
        comparisons = {
            f"{study.baseline_credit_form}_vs_{credit}": {
                "advantage": advantage_comparison(
                    baseline["raw_advantage"], measurements[credit]["raw_advantage"]
                ),
                "gradients": {
                    group: gradient_comparison(
                        baseline["gradients"][group], measurements[credit]["gradients"][group]
                    )
                    for group in GRADIENT_GROUPS
                },
            }
            for credit in study.comparison_credit_forms
        }
        updates.append(
            {
                "update_index": index,
                "policy_checkpoint": checkpoint.name,
                "losses": {credit: measurements[credit]["loss"] for credit in credit_forms},
                "comparisons": comparisons,
                "provenance": provenance,
                "critic": {
                    "explained_variance": recorded["mean_explained_variance"],
                    "value_loss": recorded["mean_value_loss"],
                },
            }
        )
        samples.append(
            {
                "run": run.label,
                "arm": run.arm,
                "training_seed": run.training_seed,
                "update_index": index,
                "prefix": prefix,
                "transition_count": transition_count,
                "credit_forms": list(credit_forms),
            }
        )
        gradient_store[(run.arm, run.training_seed, index)] = {
            group: baseline["gradients"][group] for group in _CROSS_ARM_GROUPS
        }
    return {
        "label": run.label,
        "arm": run.arm,
        "training_seed": run.training_seed,
        "reward_profile": run.reward_profile,
        "initial_policy_hash": source_summary["initial_policy_hash"],
        "updates": updates,
    }


def _cross_arm(
    records: list[dict],
    gradient_store: dict[tuple[str, int, int], dict[str, np.ndarray]],
) -> list[dict]:
    """Descriptive, unmatched comparison of r0 and rstress actor gradients."""

    rows = []
    seeds = sorted({record["training_seed"] for record in records})
    for seed in seeds:
        for index in sorted(
            {key[2] for key in gradient_store if key[1] == seed and key[0] in ("r0", "rstress")}
        ):
            left = gradient_store.get(("r0", seed, index))
            right = gradient_store.get(("rstress", seed, index))
            if left is None or right is None:
                continue
            for group in _CROSS_ARM_GROUPS:
                rows.append(
                    {
                        "training_seed": seed,
                        "update_index": index,
                        "group": group,
                        "arm_i": "r0",
                        "arm_j": "rstress",
                        "matched": False,
                        **gradient_comparison(left[group], right[group]),
                    }
                )
    return rows


def run(source: Path, config_path: Path, output: Path, *, figures: bool = True) -> dict:
    study = CriticAttributionConfig.model_validate(load_resolved_yaml_mapping(config_path))
    arrays: dict[str, np.ndarray] = {}
    samples: list[dict[str, Any]] = []
    gradient_store: dict[tuple[str, int, int], dict[str, np.ndarray]] = {}
    records = [
        _measure_run(run_spec, source, study, arrays, samples, gradient_store)
        for run_spec in study.runs
    ]
    comparisons: dict[str, list[dict]] = {}
    for record in records:
        for update in record["updates"]:
            for label, comparison in update["comparisons"].items():
                comparisons.setdefault(label, []).append(
                    {"run": record["label"], "update_index": update["update_index"], **comparison}
                )
    gate = evaluate_materiality(comparisons, study.thresholds)
    summary = {
        "kind": "training-critic-attribution",
        "status": "completed",
        "source": str(source.resolve()),
        "credit_forms": [study.baseline_credit_form, *study.comparison_credit_forms],
        "advantage_form": study.advantage_form,
        "update_indices": list(study.update_indices),
        "optimizer_steps": 0,
        "policy_checkpoints": (
            "pre-update policy: policy-initial for update 0, else policy-update-(k-1)"
        ),
        "runs": records,
        "cross_arm": _cross_arm(records, gradient_store),
        "gate": gate,
        "interpretation": (
            "Offline backward-only actor-gradient attribution over saved training "
            "rollouts; it does not establish learned behavior."
        ),
    }
    output.mkdir(parents=True, exist_ok=False)
    OmegaConf.save(OmegaConf.create(study.model_dump()), output / "diagnostic_config.yaml")
    write_json(
        output / "runtime_metadata.json",
        {
            **collect_repository_metadata(REPOSITORY_ROOT),
            "source": str(source.resolve()),
            "device": "cpu",
            "torch_version": str(torch.__version__),
            "actor_backward_precision": "float32 (no rollout autocast)",
            "optimizer_steps": 0,
            "policy_checkpoints": summary["policy_checkpoints"],
            "runs": [
                {
                    "label": record["label"],
                    "arm": record["arm"],
                    "training_seed": record["training_seed"],
                    "initial_policy_hash": record["initial_policy_hash"],
                }
                for record in records
            ],
        },
    )
    write_json(output / "sample_index.json", {"samples": samples})
    write_npz(output / "diagnostics.npz", arrays)
    write_json(output / "summary.json", summary)
    publish("training-critic-attribution", output, output, figures=figures)
    return {
        "status": "completed",
        "output_dir": str(output),
        "optimizer_steps": 0,
        "verdict": gate["verdict"],
    }
