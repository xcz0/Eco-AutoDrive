"""Run the Issue #94 Task H evaluation-diagnostics study."""

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
from eco_planner.experiments.training.objective_positive_control.diagnostics import (
    heldout_values,
)
from eco_planner.jobs import run_evaluation_job

from .config import (
    EvaluationDiagnosticsStudyConfig,
    load_evaluation_diagnostics_study,
)
from .diagnostics import ARMS, POLICIES, build_diagnostics_summary

_SOURCE_HELDOUT_POLICY_KEYS = {"initial": "initial", "r0": "r0", "rstress": "rstress"}


def run(source_dir: Path, config_path: Path, output_dir: Path, *, figures: bool = True) -> dict:
    """Diagnose one frozen positive-control study with stochastic evaluations.

    The matched deterministic held-out results are reused from the source study;
    this run only adds diagnostic stochastic evaluations with fixed
    policy-action seeds plus the offline Beta-distribution comparison.
    """

    study = load_evaluation_diagnostics_study(config_path)
    protocol = load_scalar_reward_protocol(study.protocol_path())
    _require_source_layout(source_dir, study)
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.load(config_path), output_dir / "study_manifest.yaml", resolve=True)
    records: dict[int, dict[str, dict[str, Any]]] = {}
    for seed in study.training_seeds:
        records[seed] = {}
        for policy in POLICIES:
            checkpoint = _source_checkpoint(source_dir, seed, policy)
            records[seed][policy] = {
                "checkpoint": str(checkpoint),
                "deterministic": _deterministic_values(source_dir, seed, policy, checkpoint),
                "stochastic": {
                    action_seed: _ensure_stochastic_evaluation(
                        protocol,
                        checkpoint,
                        policy,
                        action_seed,
                        output_dir
                        / "stochastic"
                        / f"seed-{seed}"
                        / policy
                        / f"action-{action_seed}",
                    )
                    for action_seed in study.policy_action_seeds
                },
            }
            print(
                f"stochastic evaluations seed-{seed} {policy}: completed",
                flush=True,
            )
    beta_records = {
        seed: {arm: _beta_record(source_dir, seed, arm) for arm in ARMS}
        for seed in study.training_seeds
    }
    summary = build_diagnostics_summary(records, beta_records, study.policy_action_seeds)
    summary["study_name"] = study.study_name
    summary["source_study_dir"] = str(source_dir)
    write_json(output_dir / "summary.json", summary)
    return {
        "status": "completed",
        "study_name": study.study_name,
        "source_study_dir": str(source_dir),
        "output_dir": str(output_dir),
    }


def _require_source_layout(source_dir: Path, study: EvaluationDiagnosticsStudyConfig) -> None:
    """Require the frozen positive-control artifacts the diagnostics depend on."""

    for seed in study.training_seeds:
        for arm in ARMS:
            run_dir = source_dir / f"{arm}-seed-{seed}"
            for name in ("summary.json", "policy-initial.pt", "policy-final.pt"):
                if not (run_dir / name).exists():
                    raise RuntimeError(f"source run {run_dir} is missing {name}")
        for policy in POLICIES:
            heldout = source_dir / "heldout" / f"seed-{seed}" / policy
            if not (heldout / "summary.json").exists():
                raise RuntimeError(f"source held-out evaluation {heldout} is missing summary.json")


def _source_checkpoint(source_dir: Path, seed: int, policy: str) -> Path:
    """Locate the source-study checkpoint evaluated for one policy entry."""

    if policy == "initial":
        return source_dir / f"r0-seed-{seed}" / "policy-initial.pt"
    return source_dir / f"{policy}-seed-{seed}" / "policy-final.pt"


def _deterministic_values(
    source_dir: Path, seed: int, policy: str, checkpoint: Path
) -> dict[str, Any]:
    """Reuse one matched deterministic held-out evaluation from the source study."""

    summary = _load_evaluation_summary(
        source_dir / "heldout" / f"seed-{seed}" / _SOURCE_HELDOUT_POLICY_KEYS[policy]
    )
    provenance = summary["policy_checkpoint"]
    if provenance is None or Path(provenance["path"]).resolve() != checkpoint.resolve():
        raise RuntimeError(
            f"source held-out evaluation for seed-{seed} {policy} was produced from a "
            "different checkpoint than the diagnosed one"
        )
    return heldout_values(summary["episodes"])


def _ensure_stochastic_evaluation(
    protocol: ScalarRewardProtocolConfig,
    checkpoint: Path,
    policy: str,
    action_seed: int,
    eval_dir: Path,
) -> dict[str, Any]:
    """Run (or reuse) one diagnostic stochastic held-out evaluation."""

    if not (eval_dir / "summary.json").exists():
        config, _ = compose_policy_evaluation_config(
            protocol,
            "a1",
            cast(CheckpointLabel, "initial" if policy == "initial" else "final"),
            checkpoint,
            overrides=[
                "evaluation.policy_checkpoint.action_mode=sample",
                f"evaluation.policy_checkpoint.policy_action_seed={action_seed}",
            ],
        )
        run_evaluation_job(config, eval_dir)
    summary = _load_evaluation_summary(eval_dir)
    if summary["policy_checkpoint"] is None:
        raise RuntimeError(f"stochastic evaluation {eval_dir} has no policy-checkpoint provenance")
    return heldout_values(summary["episodes"])


def _load_evaluation_summary(eval_dir: Path) -> dict[str, Any]:
    summary = json.loads((eval_dir / "summary.json").read_text(encoding="utf-8"))
    if summary["status"] != "completed":
        raise RuntimeError(f"held-out evaluation {eval_dir} did not complete")
    return summary


def _beta_record(source_dir: Path, seed: int, arm: str) -> dict[str, Any]:
    """Read one arm's fixed-context probe summaries from the source training run."""

    summary = json.loads(
        (source_dir / f"{arm}-seed-{seed}" / "summary.json").read_text(encoding="utf-8")
    )
    return {
        "probe_before": summary["probe_before"],
        "probe_after": summary["probe_after"],
    }
