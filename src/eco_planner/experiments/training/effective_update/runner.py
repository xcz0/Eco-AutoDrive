"""Run the Issue #94 Task F effective-update region search."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from omegaconf import OmegaConf

from eco_planner.artifacts import write_json
from eco_planner.experiments.reward.scalar.composition import (
    CheckpointLabel,
    compose_arm_training_config,
    compose_policy_evaluation_config,
)
from eco_planner.experiments.reward.scalar.config import load_scalar_reward_protocol
from eco_planner.jobs import run_evaluation_job, run_training_job

from .config import EffectiveUpdateStudyConfig, load_effective_update_study
from .diagnostics import (
    evaluate_gate_f,
    evaluate_heldout_change,
    extract_arm_metrics,
    heldout_metric_values,
    post_update_kl_series,
)


def arm_label(learning_rate: float, epochs: int, max_gradient_norm: float) -> str:
    """Name one grid arm directory after its swept optimizer-region values."""

    return f"lr{learning_rate:.4e}-epochs{epochs}-mgn{max_gradient_norm:g}"


def compose_arm_overrides(
    study: EffectiveUpdateStudyConfig,
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
    """Train the pre-registered grid on the R0 anchor and evaluate Gate F."""

    study = load_effective_update_study(config_path)
    protocol = load_scalar_reward_protocol(study.protocol_path())
    output_dir.mkdir(parents=True, exist_ok=True)
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
            "status": _ensure_training_run(
                study, protocol, run_dir, learning_rate, epochs, max_gradient_norm
            ),
        }
        arms.append(arm)
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
                post_update_kl_series(Path(arm["run_dir"]), update_count=study.update_count)
            )
            if metrics["initial_policy_hash"] != initial_hash:
                raise RuntimeError(f"arm {arm['label']} has a different initial policy")
            arm["metrics"] = metrics
        arm["gate"] = evaluate_gate_f(metrics, study.gate)
    candidates = [arm for arm in arms if arm["status"] == "completed" and arm["gate"]["passed_1_6"]]
    initial_values = _heldout_values(
        protocol,
        study,
        output_dir / "heldout" / "initial",
        "initial",
        Path(completed[0]["run_dir"]) / "policy-initial.pt",
    )
    for candidate in candidates:
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
        "study_name": study.study_name,
        "selection_rule": study.selection,
        "gate_f_passed": selected is not None,
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
    return {
        "status": "completed",
        "gate_f_passed": summary["gate_f_passed"],
        "selected_config": summary["selected_config"],
        "completed_arms": len(completed),
        "failed_arms": [arm["label"] for arm in arms if arm["status"] == "failed"],
        "output_dir": str(output_dir),
    }


def _ensure_training_run(
    study: EffectiveUpdateStudyConfig,
    protocol: Any,
    run_dir: Path,
    learning_rate: float,
    epochs: int,
    max_gradient_norm: float,
) -> str:
    """Train one grid arm unless a persisted result already exists."""

    if (run_dir / "failure.json").exists():
        return "failed"
    if (run_dir / "summary.json").exists():
        return "completed"
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
    try:
        run_training_job(config, run_dir)
    except Exception as error:
        write_json(
            run_dir / "failure.json",
            {"error": f"{type(error).__name__}: {error}"},
        )
        return "failed"
    return "completed"


def _heldout_values(
    protocol: Any,
    study: EffectiveUpdateStudyConfig,
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
