"""Issue #105 Phase B causal-counterfactual attribution over matched PPO arms.

A canonical k=5 calibrated-R0 reference plus named single-mechanism diagnostic
arms share the initial policy, training seed, replay, scenario pool, batch size
and update budget. Only the declared override differs per arm. Every arm is a
matched causal diagnostic: Gate B evidence separates the proximal actor-effective
update mechanism from upstream critic/value-scale consequences, and no arm enters
the formal execution or optimizer contract.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import median
from typing import Any

from omegaconf import OmegaConf

from eco_planner.analysis import publish
from eco_planner.artifacts import write_json
from eco_planner.contracts import CLOSED_LOOP_EXECUTION_STEPS
from eco_planner.experiments.protocol.composition import compose_arm_training_config
from eco_planner.experiments.protocol.config import load_protocol
from eco_planner.jobs import run_training_job

from ..attribution import (
    arm_metrics,
    check_provenance,
    heldout_values,
    load_training_config,
    offline_gae_stats,
    summarize,
)
from ..decisions import evaluate_heldout_change, evaluate_update_gate
from .diagnostics import (
    CounterfactualArm,
    CounterfactualAttributionConfig,
    load_counterfactual_attribution,
)


def run(config_path: Path, output_dir: Path, *, figures: bool = True) -> dict[str, Any]:
    """Train the matched reference and diagnostic arms and build Gate B evidence."""

    study = load_counterfactual_attribution(config_path)
    protocol = load_protocol(study.protocol_path())
    output_dir.mkdir(parents=True, exist_ok=False)
    OmegaConf.save(OmegaConf.load(config_path), output_dir / "study_manifest.yaml", resolve=True)

    specs = study.all_arms()
    metrics_by_arm: dict[str, dict[str, Any]] = {}
    gate_by_arm: dict[str, dict[str, Any]] = {}
    for spec in specs:
        run_dir = output_dir / spec.label
        _ensure_training_run(study, protocol, run_dir, spec)
        parsed = _load_training_config(run_dir)
        metrics = _arm_metrics(study, run_dir, parsed)
        gae = _offline_gae_stats(study, run_dir, parsed)
        _check_provenance(metrics, gae)
        metrics["offline_gae"] = gae
        metrics_by_arm[spec.label] = metrics
        gate_by_arm[spec.label] = evaluate_update_gate(metrics, study.gate)
    _check_shared_initial_policy(metrics_by_arm)

    heldout = _heldout(study, protocol, output_dir, specs)
    for label, values in heldout["arms"].items():
        change = evaluate_heldout_change(
            heldout["initial_values"], values["final_values"], study.gate
        )
        metrics_by_arm[label]["heldout"] = {
            "initial_values": heldout["initial_values"],
            "final_values": values["final_values"],
            **change,
        }
        gate = gate_by_arm[label]
        gate["conditions"]["c7_heldout_exceeds_noise"] = change["exceeds_noise"]
        gate["passed_all"] = gate["passed_1_6"] and change["exceeds_noise"]
        if not change["exceeds_noise"]:
            gate["failure_reasons"].append("c7_heldout_exceeds_noise")

    arms = [
        {
            "label": spec.label,
            "mechanism": spec.mechanism,
            "description": spec.description,
            "overrides": spec.overrides,
            "diagnostic_only": spec.label != study.reference.label,
            "run_dir": str(output_dir / spec.label),
            "metrics": metrics_by_arm[spec.label],
            "gate": gate_by_arm[spec.label],
        }
        for spec in specs
    ]
    summary = {
        "status": "completed",
        "kind": "training-counterfactual-attribution",
        "study_name": study.study_name,
        "historical_comparator": study.historical_comparator,
        "reference_label": study.reference.label,
        "reference_label_mechanism": study.reference.mechanism,
        "canonical_execution_steps": CLOSED_LOOP_EXECUTION_STEPS,
        "update_count": study.update_count,
        "training_seed": study.training_seed,
        "arm_name": study.arm,
        "arms": arms,
        "heldout": heldout,
        "attribution": _attribution(study, metrics_by_arm, heldout),
    }
    write_json(output_dir / "summary.json", summary)
    publish("training", output_dir, output_dir, figures=figures)
    return {
        "status": "completed",
        "output_dir": str(output_dir),
        "reference_label": study.reference.label,
        "analytic_kl_medians": {
            label: metrics["post_update_kl_analytic_median"]
            for label, metrics in metrics_by_arm.items()
        },
    }


def _compose_arm(
    study: CounterfactualAttributionConfig, protocol: Any, spec: CounterfactualArm
) -> tuple[Any, Any]:
    overrides = [
        *study.base_overrides,
        f"training.update_count={study.update_count}",
        "ppo.gradient_diagnostics=true",
        *spec.overrides,
    ]
    _, probe = compose_arm_training_config(protocol, study.arm, study.training_seed, overrides)
    steps_per_update = probe.ppo.optimizer_steps_per_update
    overrides.append(f"ppo.scheduler_total_optimizer_steps={study.update_count * steps_per_update}")
    return compose_arm_training_config(protocol, study.arm, study.training_seed, overrides)


def _ensure_training_run(
    study: CounterfactualAttributionConfig,
    protocol: Any,
    run_dir: Path,
    spec: CounterfactualArm,
) -> None:
    if (run_dir / "summary.json").exists():
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        if summary.get("status") != "completed":
            raise RuntimeError(f"existing run is not completed: {run_dir}")
        return
    config, _ = _compose_arm(study, protocol, spec)
    summary = run_training_job(config, run_dir)
    if summary.status != "completed":
        raise RuntimeError(f"training run did not complete: {run_dir}")


def _load_training_config(run_dir: Path) -> Any:
    return load_training_config(run_dir)


def _arm_metrics(
    study: CounterfactualAttributionConfig, run_dir: Path, parsed: Any
) -> dict[str, Any]:
    return arm_metrics(
        update_count=study.update_count,
        mc_draws=study.mc_draws,
        mc_seed=study.mc_seed,
        run_dir=run_dir,
        parsed=parsed,
    )


def _offline_gae_stats(
    study: CounterfactualAttributionConfig, run_dir: Path, parsed: Any
) -> list[dict[str, Any]]:
    return offline_gae_stats(update_count=study.update_count, run_dir=run_dir, parsed=parsed)


_check_provenance = check_provenance
_heldout_values = heldout_values


def _check_shared_initial_policy(metrics_by_arm: dict[str, dict[str, Any]]) -> None:
    hashes = {metrics["initial_policy_hash"] for metrics in metrics_by_arm.values()}
    if len(hashes) != 1:
        raise ValueError("matched counterfactual arms do not share one initial policy")


def _heldout(
    study: CounterfactualAttributionConfig,
    protocol: Any,
    output_dir: Path,
    specs: list[CounterfactualArm],
) -> dict[str, Any]:
    reference = study.reference.label
    initial_path = output_dir / reference / "policy-initial.pt"
    initial_values = _heldout_values(
        protocol, study.arm, output_dir / "heldout" / "initial", "initial", initial_path
    )
    arms: dict[str, dict[str, Any]] = {}
    for spec in specs:
        checkpoint = output_dir / spec.label / "policy-final.pt"
        arms[spec.label] = {
            "label": spec.label,
            "final_values": _heldout_values(
                protocol, study.arm, output_dir / "heldout" / spec.label, "final", checkpoint
            ),
        }
    return {"initial_values": initial_values, "arms": arms}


def _scalar_medians(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        "post_update_kl_analytic": metrics["post_update_kl_analytic_median"],
        "post_update_kl_seeded_mc": metrics["post_update_kl_median"],
        "post_update_kl_k3": metrics["post_update_kl_single_draw_k3_median"],
        "policy_ratio_change": metrics["policy_ratio_change"],
        "probe_guidance_rms_shift": metrics["probe_guidance_rms_shift"],
        "actor_head_parameter_delta": metrics["parameter_delta_vs_initial_final"]["actor_head"],
        "shared_trunk_parameter_delta": metrics["parameter_delta_vs_initial_final"]["shared_trunk"],
        "value_head_parameter_delta": metrics["parameter_delta_vs_initial_final"]["value_head"],
        "actor_head_policy_gradient": median(metrics["gradient_diagnostics"]["actor_head_policy"]),
        "value_head_critic_gradient": median(metrics["gradient_diagnostics"]["value_head_critic"]),
        "shared_trunk_critic_gradient": median(
            metrics["gradient_diagnostics"]["shared_trunk_critic"]
        ),
        "pre_clip_gradient_norm": median(metrics["pre_clip_gradient_norm"]),
        "effective_global_clip_coefficient": median(metrics["effective_global_clip_coefficient"]),
        "value_target_mean": median(metrics["value_target_mean"]),
        "reward_total": median(metrics["reward_total"]),
        "raw_advantage_std": median(metrics["raw_advantage_std"]),
    }


def _ratios(reference: dict[str, float], arm: dict[str, float]) -> dict[str, float | None]:
    return {
        key: (arm[key] / reference[key] if reference[key] != 0.0 else None) for key in reference
    }


def _gate_b_evidence(
    study: CounterfactualAttributionConfig, ratios: dict[str, dict[str, float | None]]
) -> dict[str, Any]:
    categories: dict[str, list[str]] = {}
    for arm in study.arms:
        categories.setdefault(arm.mechanism, []).append(arm.label)
    return {
        mechanism: {
            "arms": labels,
            "vs_reference": {label: ratios[label] for label in labels},
        }
        for mechanism, labels in categories.items()
    }


def _attribution(
    study: CounterfactualAttributionConfig,
    metrics_by_arm: dict[str, dict[str, Any]],
    heldout: dict[str, Any],
) -> dict[str, Any]:
    reference = study.reference.label
    reference_scalars = _scalar_medians(metrics_by_arm[reference])
    ratios = {
        label: _ratios(reference_scalars, _scalar_medians(metrics))
        for label, metrics in metrics_by_arm.items()
        if label != reference
    }
    return {
        "reference_label": reference,
        "post_update_kl": {
            label: {
                "analytic": summarize(metrics["post_update_kl_analytic"]),
                "seeded_mc": summarize(metrics["post_update_kl"]),
                "k3_median": metrics["post_update_kl_single_draw_k3_median"],
            }
            for label, metrics in metrics_by_arm.items()
        },
        "value_target": {
            label: summarize(metrics["value_target_mean"])
            for label, metrics in metrics_by_arm.items()
        },
        "reward_total": {
            label: summarize(metrics["reward_total"]) for label, metrics in metrics_by_arm.items()
        },
        "advantage_forms": {
            form: {
                statistic: {
                    label: summarize(metrics[f"{form}_advantage_{statistic}"])
                    for label, metrics in metrics_by_arm.items()
                }
                for statistic in ("mean", "std")
            }
            for form in ("raw", "center", "normalized")
        },
        "gradients": {
            group: {
                label: summarize(metrics["gradient_diagnostics"][group])
                for label, metrics in metrics_by_arm.items()
            }
            for group in metrics_by_arm[reference]["gradient_diagnostics"]
        },
        "clipping": {
            "pre_clip_gradient_norm": {
                label: summarize(metrics["pre_clip_gradient_norm"])
                for label, metrics in metrics_by_arm.items()
            },
            "post_clip_gradient_norm": {
                label: summarize(metrics["post_clip_gradient_norm"])
                for label, metrics in metrics_by_arm.items()
            },
            "effective_global_clip_coefficient": {
                label: summarize(metrics["effective_global_clip_coefficient"])
                for label, metrics in metrics_by_arm.items()
            },
        },
        "parameter_delta_vs_initial_final": {
            group: {
                label: metrics["parameter_delta_vs_initial_final"][group]
                for label, metrics in metrics_by_arm.items()
            }
            for group in metrics_by_arm[reference]["parameter_delta_vs_initial_final"]
        },
        "policy_ratio": {
            "change": {
                label: metrics["policy_ratio_change"] for label, metrics in metrics_by_arm.items()
            },
            "mean_median": {
                label: metrics["policy_ratio_mean_median"]
                for label, metrics in metrics_by_arm.items()
            },
        },
        "heldout": {
            "initial": heldout["initial_values"],
            **{
                label: {
                    "final_values": values["final_values"],
                    **evaluate_heldout_change(
                        heldout["initial_values"], values["final_values"], study.gate
                    ),
                }
                for label, values in heldout["arms"].items()
            },
        },
        "scalar_medians": {
            label: _scalar_medians(metrics) for label, metrics in metrics_by_arm.items()
        },
        "ratios_vs_reference": ratios,
        "gate_b_evidence": _gate_b_evidence(study, ratios),
    }
