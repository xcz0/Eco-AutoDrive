"""Shared matched-training measurements for cadence/counterfactual attribution.

These helpers read one completed training run's persisted diagnostics; they are
shared by the Issue #105 Phase A cadence attribution and the Phase B causal
counterfactual study. They do not define study semantics: each runner owns its
own manifest, arm composition and verdict.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import median
from typing import Any, cast

import torch
from omegaconf import OmegaConf

from eco_planner.analysis.training import heldout_metric_values
from eco_planner.configuration import load_resolved_yaml_mapping
from eco_planner.experiments.protocol.composition import (
    CheckpointLabel,
    compose_policy_evaluation_config,
)
from eco_planner.jobs import run_evaluation_job
from eco_planner.rl import parse_training_config, read_rollout_episode
from eco_planner.rl.optimization.ppo import build_ppo_batch
from eco_planner.rl.optimization.update_diagnostics import (
    GRADIENT_GROUPS,
    extract_arm_metrics,
    post_update_kl_series,
)

REWARD_COMPONENTS = ("ttc", "progress", "comfort", "speed", "energy")


def load_training_config(run_dir: Path) -> Any:
    resolved = load_resolved_yaml_mapping(run_dir / "resolved_config.yaml")
    return parse_training_config(OmegaConf.create(resolved))


def arm_metrics(
    *,
    update_count: int,
    mc_draws: int,
    mc_seed: int,
    run_dir: Path,
    parsed: Any,
) -> dict[str, Any]:
    """Collect the matched per-update measurements persisted for one arm."""

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    updates = summary["updates"]
    if len(updates) != update_count:
        raise ValueError(
            f"training run {run_dir} has {len(updates)} updates, expected {update_count}"
        )
    metrics = extract_arm_metrics(
        run_dir,
        max_gradient_norm=parsed.ppo.max_gradient_norm,
        update_count=update_count,
    )
    metrics.update(
        post_update_kl_series(
            run_dir,
            update_count=update_count,
            mc_draws=mc_draws,
            mc_seed=mc_seed,
        )
    )
    metrics["raw_advantage_mean"] = [update["raw_advantage_mean"] for update in updates]
    metrics["raw_advantage_std"] = [update["raw_advantage_std"] for update in updates]
    metrics["normalized_advantage_mean"] = [
        update["normalized_advantage_mean"] for update in updates
    ]
    metrics["normalized_advantage_std"] = [update["normalized_advantage_std"] for update in updates]
    # Centering subtracts the full-batch mean; the centered std equals the raw std.
    metrics["center_advantage_mean"] = [0.0 for _ in updates]
    metrics["center_advantage_std"] = list(metrics["raw_advantage_std"])
    metrics["value_target_mean"] = [update["mean_value_target"] for update in updates]
    metrics["value_target_std"] = [update["std_value_target"] for update in updates]
    metrics["reward_total"] = [update["total_reward"] for update in updates]
    metrics["reward_component_means"] = {
        component: [update["reward_component_means"][component] for update in updates]
        for component in REWARD_COMPONENTS
    }
    metrics["effective_global_clip_coefficient"] = [
        min(1.0, parsed.ppo.max_gradient_norm / value) if value > 0.0 else 1.0
        for value in metrics["pre_clip_gradient_norm"]
    ]
    metrics["beta_concentration_mean"] = [
        [
            update["beta_alpha_mean"][dimension] + update["beta_beta_mean"][dimension]
            for dimension in range(2)
        ]
        for update in updates
    ]
    metrics["gradient_diagnostics"] = {
        group: [update["gradient_diagnostics"][group] for update in updates]
        for group in GRADIENT_GROUPS
    }
    return metrics


def offline_gae_stats(*, update_count: int, run_dir: Path, parsed: Any) -> list[dict[str, Any]]:
    """Rebuild GAE offline from persisted rollouts as an independent provenance check."""

    rows: list[dict[str, Any]] = []
    for index in range(update_count):
        update_dir = run_dir / "updates" / f"update-{index:03d}"
        paths = sorted(update_dir.glob("slot-*-episode-*.npz"))
        if not paths:
            raise ValueError(f"update {index} has no persisted rollout episodes: {run_dir}")
        episodes = [read_rollout_episode(path) for path in paths]
        batch = build_ppo_batch(episodes, parsed.ppo, device=torch.device("cpu"))
        advantage = cast(torch.Tensor, batch["advantage"]).to(dtype=torch.float64)
        value_target = cast(torch.Tensor, batch["value_target"]).to(dtype=torch.float64)
        bootstrap = torch.cat(
            [
                episode.tail_bootstrap_value.reshape(-1).to(dtype=torch.float64)
                for episode in episodes
            ]
        )
        rows.append(
            {
                "advantage_mean": float(advantage.mean()),
                "advantage_std": float(advantage.std(correction=1)),
                "advantage_abs_mean": float(advantage.abs().mean()),
                "value_target_mean": float(value_target.mean()),
                "value_target_std": float(value_target.std(correction=0)),
                "bootstrap_mean": float(bootstrap.mean()),
                "bootstrap_abs_mean": float(bootstrap.abs().mean()),
            }
        )
    return rows


def check_provenance(metrics: dict[str, Any], gae: list[dict[str, Any]]) -> None:
    for index, row in enumerate(gae):
        if not math.isclose(
            row["advantage_mean"], metrics["raw_advantage_mean"][index], rel_tol=1e-4, abs_tol=1e-6
        ) or not math.isclose(
            row["advantage_std"], metrics["raw_advantage_std"][index], rel_tol=1e-4, abs_tol=1e-6
        ):
            raise ValueError(f"offline GAE advantage differs from recorded update {index}")
        if not math.isclose(
            row["value_target_mean"],
            metrics["value_target_mean"][index],
            rel_tol=1e-4,
            abs_tol=1e-6,
        ):
            raise ValueError(f"offline GAE value target differs from recorded update {index}")


def heldout_values(
    protocol: Any,
    arm: str,
    eval_dir: Path,
    label: str,
    checkpoint_path: Path,
) -> dict[str, Any]:
    """Evaluate one checkpoint on the matched held-out pool (idempotent)."""

    if (eval_dir / "summary.json").exists():
        summary = json.loads((eval_dir / "summary.json").read_text(encoding="utf-8"))
    else:
        config, _ = compose_policy_evaluation_config(
            protocol, arm, cast(CheckpointLabel, label), checkpoint_path
        )
        run_evaluation_job(config, eval_dir)
        summary = json.loads((eval_dir / "summary.json").read_text(encoding="utf-8"))
    if summary["status"] != "completed":
        raise RuntimeError(f"held-out evaluation {eval_dir} did not complete")
    return heldout_metric_values(summary["episodes"])


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("summary statistics require at least one value")
    return {
        "median": median(values),
        "mean": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
    }


def compare(a: list[float], b: list[float]) -> dict[str, Any]:
    a_stats, b_stats = summarize(a), summarize(b)
    ratio = b_stats["median"] / a_stats["median"] if a_stats["median"] != 0.0 else None
    return {"diagnostic": a_stats, "canonical": b_stats, "median_ratio": ratio}
