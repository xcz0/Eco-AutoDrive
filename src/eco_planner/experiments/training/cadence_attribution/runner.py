"""Matched cadence-PPO attribution over two execution-prefix arms (Issue #105).

Both arms share the calibrated R0 arm, initial policy, training seed, replay,
scenario pool, batch size and update budget; they differ only in the diagnostic
``training.diagnostic_execution_steps`` override. The canonical k=5 arm is the
formal contract; the k=1 arm is a matched causal diagnostic. Every measurement
is persisted so the mismatch can be attributed before any counterfactual search.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import median
from typing import Any, cast

import torch
from omegaconf import OmegaConf

from eco_planner.analysis import publish
from eco_planner.analysis.training import heldout_metric_values
from eco_planner.artifacts import write_json
from eco_planner.configuration import load_resolved_yaml_mapping
from eco_planner.contracts import CLOSED_LOOP_EXECUTION_STEPS
from eco_planner.experiments.protocol.composition import (
    CheckpointLabel,
    compose_arm_training_config,
    compose_policy_evaluation_config,
)
from eco_planner.experiments.protocol.config import load_protocol
from eco_planner.jobs import run_evaluation_job, run_training_job
from eco_planner.rl import parse_training_config, read_rollout_episode
from eco_planner.rl.optimization.ppo import build_ppo_batch
from eco_planner.rl.optimization.update_diagnostics import (
    extract_arm_metrics,
    post_update_kl_series,
)

from ..decisions import evaluate_heldout_change, evaluate_update_gate
from .diagnostics import CadenceAttributionConfig, load_cadence_attribution

_GRADIENT_GROUPS = (
    "actor_head_policy",
    "shared_trunk_policy",
    "value_head_critic",
    "shared_trunk_critic",
    "actor_head_entropy",
    "shared_trunk_entropy",
)
_REWARD_COMPONENTS = ("ttc", "progress", "comfort", "speed", "energy")


def run(config_path: Path, output_dir: Path, *, figures: bool = True) -> dict[str, Any]:
    """Train the matched diagnostic and canonical arms and attribute the gap."""

    study = load_cadence_attribution(config_path)
    protocol = load_protocol(study.protocol_path())
    output_dir.mkdir(parents=True, exist_ok=False)
    OmegaConf.save(OmegaConf.load(config_path), output_dir / "study_manifest.yaml", resolve=True)

    arm_metrics: dict[int, dict[str, Any]] = {}
    arm_gate: dict[int, dict[str, Any]] = {}
    for execution_steps in study.execution_steps:
        label = study.arm_label(execution_steps)
        run_dir = output_dir / label
        _ensure_training_run(study, protocol, run_dir, execution_steps)
        parsed = _load_training_config(run_dir)
        metrics = _arm_metrics(study, run_dir, parsed)
        gae = _offline_gae_stats(study, run_dir, parsed)
        _check_provenance(metrics, gae)
        metrics["offline_gae"] = gae
        metrics["execution_steps"] = execution_steps
        arm_metrics[execution_steps] = metrics
        arm_gate[execution_steps] = evaluate_update_gate(metrics, study.gate)

    heldout = _heldout(study, protocol, output_dir, arm_metrics)
    for execution_steps, values in heldout["arms"].items():
        final_values = values["final_values"]
        change = evaluate_heldout_change(heldout["initial_values"], final_values, study.gate)
        arm_metrics[execution_steps]["heldout"] = {
            "initial_values": heldout["initial_values"],
            "final_values": final_values,
            **change,
        }
        gate = arm_gate[execution_steps]
        gate["conditions"]["c7_heldout_exceeds_noise"] = change["exceeds_noise"]
        gate["passed_all"] = gate["passed_1_6"] and change["exceeds_noise"]
        if not change["exceeds_noise"]:
            gate["failure_reasons"].append("c7_heldout_exceeds_noise")

    arms = [
        {
            "label": study.arm_label(steps),
            "execution_steps": steps,
            "run_dir": str(output_dir / study.arm_label(steps)),
            "metrics": arm_metrics[steps],
            "gate": arm_gate[steps],
        }
        for steps in study.execution_steps
    ]
    summary = {
        "status": "completed",
        "kind": "training-cadence-attribution",
        "study_name": study.study_name,
        "historical_comparator": study.historical_comparator,
        "diagnostic_execution_steps": study.diagnostic_steps(),
        "canonical_execution_steps": CLOSED_LOOP_EXECUTION_STEPS,
        "update_count": study.update_count,
        "training_seed": study.training_seed,
        "arm_name": study.arm,
        "arms": arms,
        "heldout": heldout,
        "attribution": _attribution(study, arm_metrics, heldout),
    }
    write_json(output_dir / "summary.json", summary)
    publish("training", output_dir, output_dir, figures=figures)
    return {
        "status": "completed",
        "output_dir": str(output_dir),
        "execution_steps": list(study.execution_steps),
        "analytic_kl_medians": {
            study.arm_label(steps): arm_metrics[steps]["post_update_kl_analytic_median"]
            for steps in study.execution_steps
        },
    }


def _compose_arm(
    study: CadenceAttributionConfig, protocol: Any, execution_steps: int
) -> tuple[Any, Any]:
    overrides = [
        *study.base_overrides,
        f"training.update_count={study.update_count}",
        "ppo.gradient_diagnostics=true",
        f"training.diagnostic_execution_steps={execution_steps}",
    ]
    _, probe = compose_arm_training_config(protocol, study.arm, study.training_seed, overrides)
    steps_per_update = probe.ppo.optimizer_steps_per_update
    overrides.append(f"ppo.scheduler_total_optimizer_steps={study.update_count * steps_per_update}")
    return compose_arm_training_config(protocol, study.arm, study.training_seed, overrides)


def _ensure_training_run(
    study: CadenceAttributionConfig,
    protocol: Any,
    run_dir: Path,
    execution_steps: int,
) -> None:
    if (run_dir / "summary.json").exists():
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        if summary.get("status") != "completed":
            raise RuntimeError(f"existing run is not completed: {run_dir}")
        return
    config, _ = _compose_arm(study, protocol, execution_steps)
    summary = run_training_job(config, run_dir)
    if summary.status != "completed":
        raise RuntimeError(f"training run did not complete: {run_dir}")


def _load_training_config(run_dir: Path) -> Any:
    resolved = load_resolved_yaml_mapping(run_dir / "resolved_config.yaml")
    return parse_training_config(OmegaConf.create(resolved))


def _arm_metrics(study: CadenceAttributionConfig, run_dir: Path, parsed: Any) -> dict[str, Any]:
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    updates = summary["updates"]
    if len(updates) != study.update_count:
        raise ValueError(
            f"training run {run_dir} has {len(updates)} updates, expected {study.update_count}"
        )
    metrics = extract_arm_metrics(
        run_dir,
        max_gradient_norm=parsed.ppo.max_gradient_norm,
        update_count=study.update_count,
    )
    metrics.update(
        post_update_kl_series(
            run_dir,
            update_count=study.update_count,
            mc_draws=study.mc_draws,
            mc_seed=study.mc_seed,
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
        for component in _REWARD_COMPONENTS
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
        for group in _GRADIENT_GROUPS
    }
    return metrics


def _offline_gae_stats(
    study: CadenceAttributionConfig, run_dir: Path, parsed: Any
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in range(study.update_count):
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


def _check_provenance(metrics: dict[str, Any], gae: list[dict[str, Any]]) -> None:
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


def _heldout(
    study: CadenceAttributionConfig,
    protocol: Any,
    output_dir: Path,
    arm_metrics: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    first = min(study.execution_steps)
    initial_path = output_dir / study.arm_label(first) / "policy-initial.pt"
    initial_values = _heldout_values(
        protocol, study, output_dir / "heldout" / "initial", "initial", initial_path
    )
    arms: dict[int, dict[str, Any]] = {}
    for steps in study.execution_steps:
        label = study.arm_label(steps)
        checkpoint = output_dir / label / "policy-final.pt"
        arms[steps] = {
            "label": label,
            "final_values": _heldout_values(
                protocol,
                study,
                output_dir / "heldout" / label,
                "final",
                checkpoint,
            ),
        }
    return {"initial_values": initial_values, "arms": arms}


def _heldout_values(
    protocol: Any,
    study: CadenceAttributionConfig,
    eval_dir: Path,
    label: str,
    checkpoint_path: Path,
) -> dict[str, Any]:
    if (eval_dir / "summary.json").exists():
        summary = json.loads((eval_dir / "summary.json").read_text(encoding="utf-8"))
    else:
        config, _ = compose_policy_evaluation_config(
            protocol, study.arm, cast(CheckpointLabel, label), checkpoint_path
        )
        run_evaluation_job(config, eval_dir)
        summary = json.loads((eval_dir / "summary.json").read_text(encoding="utf-8"))
    if summary["status"] != "completed":
        raise RuntimeError(f"held-out evaluation {eval_dir} did not complete")
    return heldout_metric_values(summary["episodes"])


def _summarize(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("summary statistics require at least one value")
    return {
        "median": median(values),
        "mean": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
    }


def _compare(a: list[float], b: list[float]) -> dict[str, Any]:
    a_stats, b_stats = _summarize(a), _summarize(b)
    ratio = b_stats["median"] / a_stats["median"] if a_stats["median"] != 0.0 else None
    return {"diagnostic": a_stats, "canonical": b_stats, "median_ratio": ratio}


def _attribution(
    study: CadenceAttributionConfig,
    arm_metrics: dict[int, dict[str, Any]],
    heldout: dict[str, Any],
) -> dict[str, Any]:
    diagnostic = study.diagnostic_steps()
    canonical = CLOSED_LOOP_EXECUTION_STEPS
    a, b = arm_metrics[diagnostic], arm_metrics[canonical]
    advantage = {
        form: {
            "mean": _compare(
                arm_metrics[diagnostic][f"{form}_advantage_mean"],
                arm_metrics[canonical][f"{form}_advantage_mean"],
            ),
            "std": _compare(
                arm_metrics[diagnostic][f"{form}_advantage_std"],
                arm_metrics[canonical][f"{form}_advantage_std"],
            ),
        }
        for form in ("raw", "center", "normalized")
    }
    gradients = {
        group: _compare(
            arm_metrics[diagnostic]["gradient_diagnostics"][group],
            arm_metrics[canonical]["gradient_diagnostics"][group],
        )
        for group in _GRADIENT_GROUPS
    }
    reward_components = {
        component: _compare(
            arm_metrics[diagnostic]["reward_component_means"][component],
            arm_metrics[canonical]["reward_component_means"][component],
        )
        for component in _REWARD_COMPONENTS
    }
    return {
        "execution_steps": {"diagnostic": diagnostic, "canonical": canonical},
        "post_update_kl": {
            "analytic": {
                "diagnostic": _summarize(a["post_update_kl_analytic"]),
                "canonical": _summarize(b["post_update_kl_analytic"]),
            },
            "seeded_mc": {
                "diagnostic": _summarize(a["post_update_kl"]),
                "canonical": _summarize(b["post_update_kl"]),
            },
            "k3": {
                "diagnostic_median": a["post_update_kl_single_draw_k3_median"],
                "canonical_median": b["post_update_kl_single_draw_k3_median"],
            },
        },
        "advantage_forms": advantage,
        "value_target": {
            "mean": _compare(a["value_target_mean"], b["value_target_mean"]),
            "std": _compare(a["value_target_std"], b["value_target_std"]),
        },
        "reward_total": _compare(a["reward_total"], b["reward_total"]),
        "reward_components": reward_components,
        "gae": {
            key: _compare(
                [row[key] for row in a["offline_gae"]],
                [row[key] for row in b["offline_gae"]],
            )
            for key in (
                "advantage_mean",
                "advantage_std",
                "advantage_abs_mean",
                "value_target_mean",
                "value_target_std",
                "bootstrap_abs_mean",
            )
        },
        "gradients": gradients,
        "clipping": {
            "pre_clip_gradient_norm": _compare(
                a["pre_clip_gradient_norm"], b["pre_clip_gradient_norm"]
            ),
            "post_clip_gradient_norm": _compare(
                a["post_clip_gradient_norm"], b["post_clip_gradient_norm"]
            ),
            "effective_global_clip_coefficient": _compare(
                a["effective_global_clip_coefficient"], b["effective_global_clip_coefficient"]
            ),
        },
        "parameter_delta_vs_initial_final": {
            group: {
                "diagnostic": a["parameter_delta_vs_initial_final"][group],
                "canonical": b["parameter_delta_vs_initial_final"][group],
            }
            for group in a["parameter_delta_vs_initial_final"]
        },
        "policy_ratio": {
            "change": {
                "diagnostic": a["policy_ratio_change"],
                "canonical": b["policy_ratio_change"],
            },
            "mean_median": {
                "diagnostic": a["policy_ratio_mean_median"],
                "canonical": b["policy_ratio_mean_median"],
            },
        },
        "beta": {
            "concentration_mean": _compare(
                [value for row in a["beta_concentration_mean"] for value in row],
                [value for row in b["beta_concentration_mean"] for value in row],
            ),
            "boundary_mass_max_after": {
                "diagnostic": a["probe_boundary_mass_max_after"],
                "canonical": b["probe_boundary_mass_max_after"],
            },
            "boundary_mass_max_before": {
                "diagnostic": a["probe_boundary_mass_max_before"],
                "canonical": b["probe_boundary_mass_max_before"],
            },
        },
        "heldout": {
            "initial": heldout["initial_values"],
            **{
                values["label"]: {
                    "final_values": values["final_values"],
                    **evaluate_heldout_change(
                        heldout["initial_values"], values["final_values"], study.gate
                    ),
                }
                for values in heldout["arms"].values()
            },
        },
        "gate_a_evidence": _gate_a_evidence(advantage, reward_components, gradients, a, b),
    }


def _gate_a_evidence(
    advantage: dict[str, Any],
    reward_components: dict[str, Any],
    gradients: dict[str, Any],
    diagnostic_metrics: dict[str, Any],
    canonical_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Group the matched measurements under the four predeclared mechanism categories."""

    return {
        "actor_state_advantage_geometry": {
            "advantage_forms": advantage,
            "policy_ratio": {
                "diagnostic": diagnostic_metrics["policy_ratio_change"],
                "canonical": canonical_metrics["policy_ratio_change"],
            },
        },
        "reward_value_scale": {
            "reward_total": {
                "diagnostic": _summarize(diagnostic_metrics["reward_total"]),
                "canonical": _summarize(canonical_metrics["reward_total"]),
            },
            "value_target_mean": {
                "diagnostic": _summarize(diagnostic_metrics["value_target_mean"]),
                "canonical": _summarize(canonical_metrics["value_target_mean"]),
            },
            "reward_components": reward_components,
        },
        "physical_time_credit": {
            "offline_gae": {
                key: {
                    "diagnostic": _summarize(
                        [row[key] for row in diagnostic_metrics["offline_gae"]]
                    ),
                    "canonical": _summarize([row[key] for row in canonical_metrics["offline_gae"]]),
                }
                for key in ("advantage_std", "value_target_mean", "bootstrap_abs_mean")
            }
        },
        "critic_trunk_clipping_coupling": {
            "gradients": gradients,
            "effective_global_clip_coefficient": {
                "diagnostic": _summarize(diagnostic_metrics["effective_global_clip_coefficient"]),
                "canonical": _summarize(canonical_metrics["effective_global_clip_coefficient"]),
            },
            "value_head_parameter_delta": {
                "diagnostic": diagnostic_metrics["parameter_delta_vs_initial_final"]["value_head"],
                "canonical": canonical_metrics["parameter_delta_vs_initial_final"]["value_head"],
            },
        },
    }
