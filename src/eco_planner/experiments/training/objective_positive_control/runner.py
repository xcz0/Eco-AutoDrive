"""Run the Issue #94 Task G objective positive-control study."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from omegaconf import OmegaConf

from eco_planner.artifacts import write_json
from eco_planner.experiments.reward.scalar.composition import (
    CheckpointLabel,
    compose_policy_evaluation_config,
)
from eco_planner.experiments.reward.scalar.config import (
    ScalarRewardProtocolConfig,
    load_scalar_reward_protocol,
)
from eco_planner.experiments.training.effective_update.diagnostics import (
    extract_arm_metrics,
    post_update_kl_series,
)
from eco_planner.jobs import run_evaluation_job, run_training_job

from .composition import (
    compose_frozen_ppo_overrides,
    compose_positive_control_training_config,
    heldout_dir,
    run_label,
)
from .config import (
    ObjectivePositiveControlStudyConfig,
    load_objective_positive_control_study,
)
from .diagnostics import (
    evaluate_gate_g,
    heldout_values,
    load_probe_field,
    probe_rms,
)

_ARM_KEYS = ("r0", "rstress")


def run(config_path: Path, output_dir: Path, *, figures: bool = True) -> dict[str, Any]:
    """Train the matched R0 vs Rstress arms and evaluate Gate G."""

    study = load_objective_positive_control_study(config_path)
    protocol = load_scalar_reward_protocol(study.protocol_path())
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.load(config_path), output_dir / "study_manifest.yaml", resolve=True)
    arm_configs = {"r0": study.r0, "rstress": study.rstress}
    overrides = [
        *study.base_overrides,
        *compose_frozen_ppo_overrides(
            learning_rate=study.ppo.learning_rate,
            epochs=study.ppo.epochs,
            max_gradient_norm=study.ppo.max_gradient_norm,
            update_count=study.update_count,
        ),
    ]
    runs: list[dict[str, Any]] = []
    for arm_key in _ARM_KEYS:
        arm_config = arm_configs[arm_key]
        for seed in study.training_seeds:
            label = run_label(arm_key, seed)
            run_dir = output_dir / label
            runs.append(
                {
                    "arm": arm_key,
                    "label": arm_config.label,
                    "reward_profile": arm_config.reward_profile,
                    "seed": seed,
                    "run_dir": str(run_dir),
                    "status": _ensure_training_run(
                        protocol, arm_config.reward_profile, seed, run_dir, overrides, study
                    ),
                }
            )
            print(f"training run {label}: {runs[-1]['status']}", flush=True)
    failed_runs = [run_label(run["arm"], run["seed"]) for run in runs if run["status"] == "failed"]
    if failed_runs:
        # A crashed training run (e.g. Beta boundary collapse) is itself the
        # decisive outcome; record it and stop instead of salvaging a partial
        # pairing.
        summary = {
            "status": "completed",
            "study_name": study.study_name,
            "gate_g_passed": False,
            "gate_g": None,
            "failure_reasons": ["training_run_failed"],
            "failed_runs": failed_runs,
            "runs": runs,
        }
        write_json(output_dir / "summary.json", summary)
        return {
            "status": "failed",
            "gate_g_passed": False,
            "failure_reasons": ["training_run_failed"],
            "failed_runs": failed_runs,
            "output_dir": str(output_dir),
        }
    for run in runs:
        metrics = extract_arm_metrics(
            Path(run["run_dir"]),
            max_gradient_norm=study.ppo.max_gradient_norm,
            update_count=study.update_count,
        )
        metrics.update(post_update_kl_series(Path(run["run_dir"]), update_count=study.update_count))
        run["metrics"] = metrics
    _require_matched_pairing(runs, study)
    completed = {run_label(run["arm"], run["seed"]): run for run in runs}
    records: dict[int, dict[str, dict[str, Any]]] = {}
    for seed in study.training_seeds:
        records[seed] = {}
        initial_values = _heldout_values(
            protocol,
            heldout_dir(output_dir, seed, "initial"),
            "initial",
            Path(completed[run_label("r0", seed)]["run_dir"]) / "policy-initial.pt",
        )
        for arm_key in _ARM_KEYS:
            run = completed[run_label(arm_key, seed)]
            run_dir = Path(run["run_dir"])
            final_values = _heldout_values(
                protocol,
                heldout_dir(output_dir, seed, arm_key),
                "final",
                run_dir / "policy-final.pt",
            )
            records[seed][arm_key] = {
                "metrics": run["metrics"],
                "heldout": final_values,
                "probe_before_guidance_mean": load_probe_field(
                    run_dir, "probe_before", "guidance_mean"
                ),
                "probe_after_guidance_mean": load_probe_field(
                    run_dir, "probe_after", "guidance_mean"
                ),
            }
        records[seed]["initial_heldout"] = initial_values
    _require_matched_probe_contexts(records, study.gate.probe_match_tolerance)
    gate = evaluate_gate_g(records, study.gate)
    summary = {
        "status": "completed",
        "study_name": study.study_name,
        "gate_g_passed": gate["passed"],
        "gate_g": gate,
        "initial_policy_hashes": {
            str(seed): completed[run_label("r0", seed)]["metrics"]["initial_policy_hash"]
            for seed in study.training_seeds
        },
        "runs": runs,
        "heldout": {
            str(seed): {
                "initial": records[seed]["initial_heldout"],
                **{arm_key: records[seed][arm_key]["heldout"] for arm_key in _ARM_KEYS},
            }
            for seed in study.training_seeds
        },
    }
    write_json(output_dir / "summary.json", summary)
    return {
        "status": "completed",
        "gate_g_passed": gate["passed"],
        "gate_conditions": gate["conditions"],
        "failure_reasons": gate["failure_reasons"],
        "failed_runs": [],
        "output_dir": str(output_dir),
    }


def _ensure_training_run(
    protocol: ScalarRewardProtocolConfig,
    reward_profile: str,
    seed: int,
    run_dir: Path,
    overrides: list[str],
    study: ObjectivePositiveControlStudyConfig,
) -> str:
    """Train one paired arm run unless a persisted result already exists."""

    if (run_dir / "failure.json").exists():
        return "failed"
    if (run_dir / "summary.json").exists():
        return "completed"
    config, parsed = compose_positive_control_training_config(
        protocol, reward_profile, seed, overrides
    )
    if parsed.ppo.batch_size != parsed.ppo.minibatch_size:
        raise RuntimeError(
            "positive-control study assumes one minibatch per epoch (batch_size == minibatch_size)"
        )
    if parsed.ppo.optimizer_steps_per_update != study.ppo.epochs:
        raise RuntimeError("positive-control study assumes optimizer_steps_per_update == epochs")
    try:
        run_training_job(config, run_dir)
    except Exception as error:
        write_json(
            run_dir / "failure.json",
            {"error": f"{type(error).__name__}: {error}"},
        )
        return "failed"
    return "completed"


def _require_matched_pairing(
    runs: list[dict[str, Any]], study: ObjectivePositiveControlStudyConfig
) -> None:
    """Require every seed's two arms to share one initial policy and budget."""

    completed = {
        run_label(run["arm"], run["seed"]): run for run in runs if run["status"] == "completed"
    }
    for seed in study.training_seeds:
        r0 = completed[run_label("r0", seed)]
        rstress = completed[run_label("rstress", seed)]
        if r0["metrics"]["initial_policy_hash"] != rstress["metrics"]["initial_policy_hash"]:
            raise RuntimeError(f"seed {seed} arms must share one initial policy hash")
        if r0["metrics"]["update_count"] != rstress["metrics"]["update_count"]:
            raise RuntimeError(f"seed {seed} arms must share one update budget")


def _require_matched_probe_contexts(
    records: dict[int, dict[str, dict[str, Any]]], tolerance: float
) -> None:
    """Require the frozen probe contexts to be identical within each seed pair."""

    for seed in sorted(records):
        r0_before = records[seed]["r0"]["probe_before_guidance_mean"]
        rstress_before = records[seed]["rstress"]["probe_before_guidance_mean"]
        delta = [
            b - a
            for row_a, row_b in zip(r0_before, rstress_before, strict=True)
            for a, b in zip(row_a, row_b, strict=True)
        ]
        if probe_rms(delta) > tolerance:
            raise RuntimeError(
                f"seed {seed} arms have different probe_before fields; "
                "the matched pairing is broken"
            )


def _heldout_values(
    protocol: ScalarRewardProtocolConfig,
    eval_dir: Path,
    label: str,
    checkpoint_path: Path,
) -> dict[str, Any]:
    """Run (or reuse) one matched held-out policy evaluation and aggregate it."""

    if (eval_dir / "summary.json").exists():
        summary = json.loads((eval_dir / "summary.json").read_text(encoding="utf-8"))
    else:
        config, _ = compose_policy_evaluation_config(
            protocol, "a1", cast(CheckpointLabel, label), checkpoint_path
        )
        run_evaluation_job(config, eval_dir)
        summary = json.loads((eval_dir / "summary.json").read_text(encoding="utf-8"))
    if summary["status"] != "completed":
        raise RuntimeError(f"held-out evaluation {eval_dir} did not complete")
    return heldout_values(summary["episodes"])
