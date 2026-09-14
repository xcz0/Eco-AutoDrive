"""Measure the explicitly configured optimizer grid."""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Any, cast

from omegaconf import OmegaConf

from eco_planner.analysis.training import heldout_metric_values
from eco_planner.artifacts import write_json
from eco_planner.experiments.protocol.composition import (
    CheckpointLabel,
    compose_arm_training_config,
    compose_policy_evaluation_config,
)
from eco_planner.experiments.protocol.config import load_protocol
from eco_planner.experiments.training.config import TrainingGridConfig, load_training_grid
from eco_planner.experiments.training.decisions import evaluate_heldout_change, evaluate_update_gate
from eco_planner.jobs import run_evaluation_job, run_training_job
from eco_planner.rl.optimization.update_diagnostics import (
    extract_arm_metrics,
    post_update_kl_series,
)


def arm_label(learning_rate: float, epochs: int, max_gradient_norm: float) -> str:
    """Name one grid arm directory after its swept optimizer-region values."""

    return f"lr{learning_rate:.4e}-epochs{epochs}-mgn{max_gradient_norm:g}"


def compose_arm_overrides(
    study: TrainingGridConfig,
    learning_rate: float,
    epochs: int,
    max_gradient_norm: float,
) -> list[str]:
    """Assemble the matched-control plus grid overrides for one training run."""

    return [
        *study.base_overrides,
        f"ppo.learning_rate={learning_rate:.8g}",
        f"ppo.epochs={epochs}",
        f"ppo.max_gradient_norm={max_gradient_norm:g}",
        f"training.update_count={study.update_count}",
        f"ppo.scheduler_total_optimizer_steps={study.update_count * epochs}",
    ]


def run(config_path: Path, output_dir: Path, *, figures: bool = True) -> dict[str, Any]:
    """Train the pre-registered grid on the R0 anchor and evaluate update gate."""

    study = load_training_grid(config_path)
    protocol = load_protocol(study.protocol_path())
    output_dir.mkdir(parents=True, exist_ok=False)
    OmegaConf.save(OmegaConf.load(config_path), output_dir / "study_manifest.yaml", resolve=True)
    arms: list[dict[str, Any]] = []
    for learning_rate, epochs, max_gradient_norm in study.grid.combinations():
        label = arm_label(learning_rate, epochs, max_gradient_norm)
        run_dir = output_dir / label
        arm = {
            "label": label,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "max_gradient_norm": max_gradient_norm,
            "run_dir": str(run_dir),
            "status": "failed",
        }
        arms.append(arm)
        try:
            arm["status"] = _ensure_training_run(
                study, protocol, run_dir, learning_rate, epochs, max_gradient_norm
            )
        finally:
            # Preserve partial evidence even when training raises; propagate the original error.
            error = sys.exc_info()[1]
            if error is not None:
                arm["failure"] = {
                    "exception_type": type(error).__name__,
                    "message": str(error),
                    "traceback": traceback.format_exc(),
                }
            write_json(
                output_dir / "summary.json",
                {
                    "kind": "training-grid",
                    "status": "incomplete",
                    "study_name": study.study_name,
                    "selection_rule": study.selection,
                    "update_gate_passed": False,
                    "selected_config": None,
                    "arms": arms,
                },
            )
        print(f"training arm {label}: {arm['status']}", flush=True)
    completed = [arm for arm in arms if arm["status"] == "completed"]
    if not completed:
        raise RuntimeError("no grid arm completed training")
    initial_hashes = {
        json.loads((Path(arm["run_dir"]) / "summary.json").read_text(encoding="utf-8"))[
            "initial_policy_hash"
        ]
        for arm in completed
    }
    if len(initial_hashes) != 1:
        raise RuntimeError("grid arms must share one initial policy hash")
    initial_hash = initial_hashes.pop()
    for arm in arms:
        metrics = None
        if arm["status"] == "completed":
            metrics = extract_arm_metrics(
                Path(arm["run_dir"]),
                max_gradient_norm=arm["max_gradient_norm"],
                update_count=study.update_count,
            )
            metrics.update(
                post_update_kl_series(
                    Path(arm["run_dir"]),
                    update_count=study.update_count,
                    mc_draws=study.mc_draws,
                    mc_seed=study.mc_seed,
                )
            )
            if metrics["initial_policy_hash"] != initial_hash:
                raise RuntimeError(f"arm {arm['label']} has a different initial policy")
            arm["metrics"] = metrics
        arm["gate"] = evaluate_update_gate(metrics, study.gate)
    candidates = [arm for arm in arms if arm["status"] == "completed" and arm["gate"]["passed_1_6"]]
    initial_values = (
        _heldout_values(
            protocol,
            study,
            output_dir / "heldout" / "initial",
            "initial",
            Path(completed[0]["run_dir"]) / "policy-initial.pt",
        )
        if candidates
        else None
    )
    for candidate in candidates:
        assert initial_values is not None
        final_values = _heldout_values(
            protocol,
            study,
            output_dir / "heldout" / candidate["label"],
            "final",
            Path(candidate["run_dir"]) / "policy-final.pt",
        )
        candidate["heldout"] = {
            "initial_values": initial_values,
            "final_values": final_values,
            **evaluate_heldout_change(initial_values, final_values, study.gate),
        }
        candidate["gate"]["conditions"]["c7_heldout_exceeds_noise"] = candidate["heldout"][
            "exceeds_noise"
        ]
        candidate["gate"]["passed_all"] = candidate["heldout"]["exceeds_noise"]
        if not candidate["heldout"]["exceeds_noise"]:
            candidate["gate"]["failure_reasons"].append("c7_heldout_exceeds_noise")
    selected = min(
        (arm for arm in candidates if arm["gate"].get("passed_all")),
        key=lambda arm: (arm["learning_rate"], arm["epochs"], arm["max_gradient_norm"]),
        default=None,
    )
    summary = {
        "status": "completed",
        "kind": "training-grid",
        "study_name": study.study_name,
        "selection_rule": study.selection,
        "update_gate_passed": selected is not None,
        "selected_config": (
            {
                "label": selected["label"],
                "learning_rate": selected["learning_rate"],
                "epochs": selected["epochs"],
                "max_gradient_norm": selected["max_gradient_norm"],
                "run_dir": selected["run_dir"],
            }
            if selected is not None
            else None
        ),
        "initial_policy_hash": initial_hash,
        "arms": arms,
    }
    write_json(output_dir / "summary.json", summary)
    from eco_planner.analysis import publish

    publish("training", output_dir, output_dir, figures=figures)
    return {
        "status": "completed",
        "update_gate_passed": summary["update_gate_passed"],
        "selected_config": summary["selected_config"],
        "completed_arms": len(completed),
        "failed_arms": [arm["label"] for arm in arms if arm["status"] == "failed"],
        "output_dir": str(output_dir),
    }


def _ensure_training_run(
    study: TrainingGridConfig,
    protocol: Any,
    run_dir: Path,
    learning_rate: float,
    epochs: int,
    max_gradient_norm: float,
) -> str:
    """Train one grid arm with the declared budget."""

    config, parsed = compose_arm_training_config(
        protocol,
        study.arm,
        study.training_seed,
        compose_arm_overrides(study, learning_rate, epochs, max_gradient_norm),
    )
    if parsed.ppo.batch_size != parsed.ppo.minibatch_size:
        raise RuntimeError(
            "effective-update grid assumes one minibatch per epoch (batch_size == minibatch_size)"
        )
    if parsed.ppo.optimizer_steps_per_update != epochs:
        raise RuntimeError("effective-update grid assumes optimizer_steps_per_update == epochs")
    summary = run_training_job(config, run_dir)
    return summary.status


def _heldout_values(
    protocol: Any,
    study: TrainingGridConfig,
    eval_dir: Path,
    label: str,
    checkpoint_path: Path,
) -> dict[str, Any]:
    """Run (or reuse) one matched held-out policy evaluation and aggregate it."""

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
