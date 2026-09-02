"""Hydra-facing orchestration for the staged PPO stability experiment."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Literal

import optuna
from optuna.trial import FrozenTrial, TrialState

from eco_planner.artifacts import write_json
from eco_planner.experiments.ppo_stability.config import (
    TrialParameters,
    load_stability_config,
)
from eco_planner.experiments.ppo_stability.report import rank_validation_configs, summarize_stage_a
from eco_planner.experiments.ppo_stability.search import (
    compose_trial_training_config,
    create_study,
    make_objective,
    prepare_study_root,
)
from eco_planner.experiments.ppo_stability.validation import run_validation
from eco_planner.jobs import run_training_job
from eco_planner.rl.artifacts import TrainingUpdateSummary


def run_stage_a(study_path: Path, output_root: Path) -> dict[str, object]:
    """Run remaining Stage A search trials and write its aggregate report."""

    config = load_stability_config(study_path)
    prepare_study_root(study_path, output_root)
    study = create_study(config, output_root)
    remaining = max(0, config.trial_count - len(study.trials))
    if remaining:
        study.optimize(make_objective(config, output_root), n_trials=remaining, catch=(Exception,))
    summary = summarize_stage_a(study, config)
    write_json(output_root / "stage-a-summary.json", summary)
    return summary


def run_validation_stage(
    study_path: Path, output_root: Path, stage: Literal["b", "c"]
) -> dict[str, object]:
    """Train/evaluate the selected candidates for one validation stage."""

    config = load_stability_config(study_path)
    study = create_study(config, output_root)
    stage_config = config.stage_b if stage == "b" else config.stage_c
    candidates = _validation_candidates(study, config.stage_b.top_config_count, output_root, stage)
    stage_root = output_root / f"stage-{stage}"
    stage_root.mkdir(parents=True, exist_ok=False)
    records: list[dict[str, object]] = []
    for trial in candidates[: stage_config.top_config_count]:
        parameters = TrialParameters.model_validate(trial.params)
        for training_seed in stage_config.training_seeds:
            records.append(
                run_validation(
                    config,
                    parameters,
                    config_id=trial.number,
                    stage=stage,
                    training_seed=training_seed,
                    update_count=stage_config.update_count,
                    output_dir=stage_root / f"config-{trial.number:04d}" / f"seed-{training_seed}",
                )
            )
    promoted = rank_validation_configs(records, len(stage_config.training_seeds))
    summary = {
        "stage": stage,
        "records": records,
        "ranked_config_ids": promoted,
        "promoted_config_ids": promoted[: config.stage_c.top_config_count] if stage == "b" else [],
    }
    write_json(stage_root / "summary.json", summary)
    return summary


def run_diagnostics(
    study_path: Path, output_root: Path, diagnostic: Literal["gradient", "guidance"]
) -> dict[str, object]:
    """Run the selected candidate with the requested diagnostic perturbation."""

    config = load_stability_config(study_path)
    study = create_study(config, output_root)
    candidates = sorted(
        (trial for trial in study.trials if trial.state == TrialState.COMPLETE),
        key=lambda item: (-(item.value or -math.inf), item.number),
    )
    if not candidates:
        raise ValueError("diagnostics require at least one completed Stage A trial")
    trial = candidates[0]
    parameters = TrialParameters.model_validate(trial.params)
    diagnostic_root = output_root / "diagnostics" / diagnostic
    diagnostic_root.mkdir(parents=True, exist_ok=False)
    runs: list[dict[str, object]] = []
    variants = ["gradient"] if diagnostic == "gradient" else ["current", "reduced"]
    for variant in variants:
        guidance_range = (
            (
                config.diagnostics.lateral_max_offset_m,
                config.diagnostics.longitudinal_max_speed_fraction,
            )
            if variant == "reduced"
            else None
        )
        raw, _ = compose_trial_training_config(
            config,
            parameters,
            training_seed=config.diagnostics.training_seed,
            update_count=config.diagnostics.update_count,
            gradient_diagnostics=diagnostic == "gradient",
            guidance_range=guidance_range,
        )
        run_dir = diagnostic_root / variant
        run_dir.mkdir(parents=True, exist_ok=False)
        summary = run_training_job(raw, run_dir)
        runs.append(
            {
                "variant": variant,
                "config_id": trial.number,
                "summary_path": str(run_dir / "summary.json"),
                "minimum_episode_length_retention": _summary_retention(summary.updates),
            }
        )
    payload = {"diagnostic": diagnostic, "runs": runs}
    write_json(diagnostic_root / "summary.json", payload)
    return payload


def run_command(
    command: Literal["stage-a", "stage-b", "stage-c", "diagnose", "summarize"],
    study_path: Path,
    output_root: Path,
    diagnostic: Literal["gradient", "guidance"] | None = None,
) -> dict[str, object]:
    """Dispatch the CLI command to one explicit experiment operation."""

    if command == "stage-a":
        return run_stage_a(study_path, output_root)
    if command == "stage-b":
        return run_validation_stage(study_path, output_root, "b")
    if command == "stage-c":
        return run_validation_stage(study_path, output_root, "c")
    if command == "diagnose":
        if diagnostic is None:
            raise ValueError("diagnose requires a diagnostic")
        return run_diagnostics(study_path, output_root, diagnostic)
    if command == "summarize":
        config = load_stability_config(study_path)
        payload = summarize_stage_a(create_study(config, output_root), config)
        write_json(output_root / "stage-a-summary.json", payload)
        return payload
    raise ValueError(f"unsupported study command {command!r}")


def _validation_candidates(
    study: optuna.Study,
    stage_b_top_config_count: int,
    output_root: Path,
    stage: Literal["b", "c"],
) -> list[FrozenTrial]:
    if stage == "b":
        return sorted(
            (trial for trial in study.trials if trial.state == TrialState.COMPLETE),
            key=lambda item: (-(item.value or -math.inf), item.number),
        )[:stage_b_top_config_count]
    payload = json.loads((output_root / "stage-b" / "summary.json").read_text(encoding="utf-8"))
    trials = {trial.number: trial for trial in study.trials}
    return [trials[int(number)] for number in payload["promoted_config_ids"]]


def _summary_retention(updates: tuple[TrainingUpdateSummary, ...]) -> float:
    baseline = updates[0].mean_episode_length
    return min(item.mean_episode_length / baseline for item in updates)
