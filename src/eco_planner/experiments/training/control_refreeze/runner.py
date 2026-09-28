"""Issue #105 Phase C canonical k=5 PPO control re-freeze.

A canonical k=5 calibrated-R0 reference plus an ordered list of named control
candidates share the initial policy, training seed, replay, scenario pool, batch
size and update budget. Only the candidate's declared control-knob overrides
differ. The search space and the ``first_passing_in_declared_order`` selection
rule are frozen in the manifest before any result is observed; selection is
gate-based and never ranks by training return. The selected candidate is the new
canonical k=5 PPO control, recorded with its resolved config for downstream
transfer/sweep Issues.
"""

from __future__ import annotations

import json
import math
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
from .diagnostics import ControlCandidate, ControlRefreezeConfig, load_control_refreeze

_ESTIMATOR_ORDER_OF_MAGNITUDE = 10.0
_FINITE_SERIES = (
    "post_update_kl",
    "post_update_kl_analytic",
    "post_update_kl_single_draw_k1",
    "pre_clip_gradient_norm",
    "post_clip_gradient_norm",
    "policy_ratio_mean",
    "policy_ratio_std",
    "value_target_mean",
    "reward_total",
    "raw_advantage_std",
    "normalized_advantage_std",
    "effective_global_clip_coefficient",
)


def run(config_path: Path, output_dir: Path, *, figures: bool = True) -> dict[str, Any]:
    """Train the frozen Phase C candidate set and select one canonical control."""

    study = load_control_refreeze(config_path)
    protocol = load_protocol(study.protocol_path())
    output_dir.mkdir(parents=True, exist_ok=False)
    OmegaConf.save(OmegaConf.load(config_path), output_dir / "study_manifest.yaml", resolve=True)

    specs = study.all_runs()
    parsed_by_label: dict[str, Any] = {}
    metrics_by_label: dict[str, dict[str, Any]] = {}
    gate_by_label: dict[str, dict[str, Any]] = {}
    for spec in specs:
        run_dir = output_dir / spec.label
        _ensure_training_run(study, protocol, run_dir, spec)
        parsed = _load_training_config(run_dir)
        metrics = _arm_metrics(study, run_dir, parsed)
        gae = _offline_gae_stats(study, run_dir, parsed)
        _check_provenance(metrics, gae)
        _require_finite(spec.label, metrics)
        metrics["offline_gae"] = gae
        metrics["estimator_consistency"] = _estimator_consistency(metrics)
        parsed_by_label[spec.label] = parsed
        metrics_by_label[spec.label] = metrics
        gate_by_label[spec.label] = evaluate_update_gate(metrics, study.gate)
    _check_shared_initial_policy(metrics_by_label)

    heldout = _heldout(study, protocol, output_dir, specs)
    for label, values in heldout["runs"].items():
        change = evaluate_heldout_change(
            heldout["initial_values"], values["final_values"], study.gate
        )
        metrics_by_label[label]["heldout"] = {
            "initial_values": heldout["initial_values"],
            "final_values": values["final_values"],
            **change,
        }
        gate = gate_by_label[label]
        gate["conditions"]["c7_heldout_exceeds_noise"] = change["exceeds_noise"]
        gate["passed_all"] = gate["passed_1_6"] and change["exceeds_noise"]
        if not change["exceeds_noise"]:
            gate["failure_reasons"].append("c7_heldout_exceeds_noise")

    selected = _select(study, gate_by_label)
    runs = [
        {
            "label": spec.label,
            "overrides": spec.overrides,
            "description": spec.description,
            "is_reference": spec.label == study.reference.label,
            "run_dir": str(output_dir / spec.label),
            "estimator_consistency": metrics_by_label[spec.label]["estimator_consistency"],
            "metrics": metrics_by_label[spec.label],
            "gate": gate_by_label[spec.label],
        }
        for spec in specs
    ]
    summary = {
        "status": "completed",
        "kind": "training-control-refreeze",
        "study_name": study.study_name,
        "historical_comparator": study.historical_comparator,
        "selection_rule": study.selection,
        "reference_label": study.reference.label,
        "canonical_execution_steps": CLOSED_LOOP_EXECUTION_STEPS,
        "update_count": study.update_count,
        "training_seed": study.training_seed,
        "arm_name": study.arm,
        "control_override_keys": sorted(
            {key for spec in study.candidates for key in spec.overrides}
        ),
        "selected_label": selected.label if selected is not None else None,
        "selected_config": _selected_config(study, parsed_by_label, output_dir, selected),
        "runs": runs,
        "heldout": heldout,
        "attribution": _attribution(study, metrics_by_label, heldout),
    }
    write_json(output_dir / "summary.json", summary)
    publish("training", output_dir, output_dir, figures=figures)
    return {
        "status": "completed",
        "output_dir": str(output_dir),
        "selection_rule": study.selection,
        "selected_label": summary["selected_label"],
        "selected_config": summary["selected_config"],
        "analytic_kl_medians": {
            spec.label: metrics_by_label[spec.label]["post_update_kl_analytic_median"]
            for spec in specs
        },
    }


def _compose_run(
    study: ControlRefreezeConfig, protocol: Any, spec: ControlCandidate
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
    study: ControlRefreezeConfig, protocol: Any, run_dir: Path, spec: ControlCandidate
) -> None:
    if (run_dir / "summary.json").exists():
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        if summary.get("status") != "completed":
            raise RuntimeError(f"existing run is not completed: {run_dir}")
        return
    config, _ = _compose_run(study, protocol, spec)
    summary = run_training_job(config, run_dir)
    if summary.status != "completed":
        raise RuntimeError(f"training run did not complete: {run_dir}")


def _load_training_config(run_dir: Path) -> Any:
    return load_training_config(run_dir)


def _arm_metrics(study: ControlRefreezeConfig, run_dir: Path, parsed: Any) -> dict[str, Any]:
    return arm_metrics(
        update_count=study.update_count,
        mc_draws=study.mc_draws,
        mc_seed=study.mc_seed,
        run_dir=run_dir,
        parsed=parsed,
    )


def _offline_gae_stats(
    study: ControlRefreezeConfig, run_dir: Path, parsed: Any
) -> list[dict[str, Any]]:
    return offline_gae_stats(update_count=study.update_count, run_dir=run_dir, parsed=parsed)


_check_provenance = check_provenance


def _check_shared_initial_policy(metrics_by_label: dict[str, dict[str, Any]]) -> None:
    hashes = {metrics["initial_policy_hash"] for metrics in metrics_by_label.values()}
    if len(hashes) != 1:
        raise ValueError("control candidates do not share one initial policy")


def _require_finite(label: str, metrics: dict[str, Any]) -> None:
    for key in _FINITE_SERIES:
        if not all(math.isfinite(value) for value in metrics[key]):
            raise FloatingPointError(f"candidate {label} recorded a non-finite {key} value")


def _estimator_consistency(metrics: dict[str, Any]) -> dict[str, Any]:
    medians = {
        "analytic": metrics["post_update_kl_analytic_median"],
        "seeded_mc": metrics["post_update_kl_median"],
        "k3": metrics["post_update_kl_single_draw_k3_median"],
    }
    values = list(medians.values())
    finite = all(math.isfinite(value) and value > 0.0 for value in values)
    if finite:
        high = max(values)
        low = min(values)
        ratio: float | None = high / low
        consistent = high / low <= _ESTIMATOR_ORDER_OF_MAGNITUDE
    else:
        ratio = None
        consistent = False
    return {
        "medians": medians,
        "max_over_min_ratio": ratio,
        "same_order_of_magnitude": consistent,
    }


def _select(
    study: ControlRefreezeConfig, gate_by_label: dict[str, dict[str, Any]]
) -> ControlCandidate | None:
    for candidate in study.candidates:
        if gate_by_label[candidate.label]["passed_all"]:
            return candidate
    return None


def _selected_config(
    study: ControlRefreezeConfig,
    parsed_by_label: dict[str, Any],
    output_dir: Path,
    selected: ControlCandidate | None,
) -> dict[str, Any] | None:
    if selected is None:
        return None
    return {
        "label": selected.label,
        "overrides": selected.overrides,
        "run_dir": str(output_dir / selected.label),
        "resolved_ppo": parsed_by_label[selected.label].ppo.model_dump(mode="json"),
    }


def _heldout(
    study: ControlRefreezeConfig,
    protocol: Any,
    output_dir: Path,
    specs: list[ControlCandidate],
) -> dict[str, Any]:
    reference = study.reference.label
    initial_path = output_dir / reference / "policy-initial.pt"
    initial_values = _heldout_values(
        protocol, study.arm, output_dir / "heldout" / "initial", "initial", initial_path
    )
    runs: dict[str, dict[str, Any]] = {}
    for spec in specs:
        checkpoint = output_dir / spec.label / "policy-final.pt"
        runs[spec.label] = {
            "label": spec.label,
            "final_values": _heldout_values(
                protocol, study.arm, output_dir / "heldout" / spec.label, "final", checkpoint
            ),
        }
    return {"initial_values": initial_values, "runs": runs}


def _heldout_values(
    protocol: Any,
    arm: str,
    eval_dir: Path,
    label: str,
    checkpoint_path: Path,
) -> dict[str, Any]:
    return heldout_values(protocol, arm, eval_dir, label, checkpoint_path)


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


def _attribution(
    study: ControlRefreezeConfig,
    metrics_by_label: dict[str, dict[str, Any]],
    heldout: dict[str, Any],
) -> dict[str, Any]:
    reference = study.reference.label
    reference_scalars = _scalar_medians(metrics_by_label[reference])
    ratios = {
        label: _ratios(reference_scalars, _scalar_medians(metrics))
        for label, metrics in metrics_by_label.items()
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
            for label, metrics in metrics_by_label.items()
        },
        "estimator_consistency": {
            label: metrics["estimator_consistency"] for label, metrics in metrics_by_label.items()
        },
        "gradients": {
            group: {
                label: summarize(metrics["gradient_diagnostics"][group])
                for label, metrics in metrics_by_label.items()
            }
            for group in metrics_by_label[reference]["gradient_diagnostics"]
        },
        "clipping": {
            "pre_clip_gradient_norm": {
                label: summarize(metrics["pre_clip_gradient_norm"])
                for label, metrics in metrics_by_label.items()
            },
            "effective_global_clip_coefficient": {
                label: summarize(metrics["effective_global_clip_coefficient"])
                for label, metrics in metrics_by_label.items()
            },
        },
        "parameter_delta_vs_initial_final": {
            group: {
                label: metrics["parameter_delta_vs_initial_final"][group]
                for label, metrics in metrics_by_label.items()
            }
            for group in metrics_by_label[reference]["parameter_delta_vs_initial_final"]
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
                for label, values in heldout["runs"].items()
            },
        },
        "scalar_medians": {
            label: _scalar_medians(metrics) for label, metrics in metrics_by_label.items()
        },
        "ratios_vs_reference": ratios,
    }
