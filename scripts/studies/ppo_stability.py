"""Run the staged PPO stability study registered for Issue #76."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import optuna
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf, open_dict
from optuna.importance import get_param_importances
from optuna.trial import FrozenTrial, Trial, TrialState
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictFloat,
    StrictInt,
    model_validator,
)

from eco_planner.artifacts import write_json
from eco_planner.configuration import load_local_environment, load_resolved_yaml_mapping
from eco_planner.evaluation.config import ScenarioConfig
from eco_planner.rl.artifacts import TrainingUpdateSummary
from eco_planner.rl.config import TrainingJobConfig, parse_training_config
from eco_planner.rl.evaluation import (
    PolicyEvaluationComparison,
    compare_policy_evaluations,
    evaluate_policy_checkpoint,
)
from eco_planner.rl.trainer import train
from scripts._paths import CONFIG_ROOT, LOCAL_ENVIRONMENT_PATH

DEFAULT_STUDY = CONFIG_ROOT / "studies" / "ppo" / "stability.yaml"


class _StrictModel(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid", allow_inf_nan=False)


class SearchSpace(_StrictModel):
    learning_rate_min: StrictFloat = Field(gt=0.0)
    learning_rate_max: StrictFloat = Field(gt=0.0)
    epochs: list[StrictInt] = Field(min_length=1)
    batch_sizes: list[StrictInt] = Field(min_length=1)
    minibatch_sizes: list[StrictInt] = Field(min_length=1)
    target_kl_min: StrictFloat = Field(gt=0.0)
    target_kl_max: StrictFloat = Field(gt=0.0)


class PruningConfig(_StrictModel):
    report_interval_updates: StrictInt = Field(gt=0)
    consecutive_update_count: StrictInt = Field(gt=0)
    minimum_episode_length_retention: StrictFloat = Field(gt=0.0, le=1.0)
    out_of_road_fraction: StrictFloat = Field(gt=0.0, le=1.0)
    clip_fraction: StrictFloat | None = Field(default=None, gt=0.0, le=1.0)
    median_startup_trials: StrictInt = Field(ge=0)
    median_warmup_updates: StrictInt = Field(ge=0)


class StageConfig(_StrictModel):
    update_count: StrictInt = Field(gt=0)
    training_seeds: list[StrictInt] = Field(min_length=1)
    top_config_count: StrictInt = Field(gt=0)


class EvaluationConfig(_StrictModel):
    seed: StrictInt = Field(ge=0)
    maps: list[str] = Field(min_length=1)
    map_seeds: list[StrictInt] = Field(min_length=1)
    transitions_per_scenario: StrictInt = Field(gt=0)
    minimum_retention: StrictFloat = Field(gt=0.0, le=1.0)


class DiagnosticConfig(_StrictModel):
    update_count: StrictInt = Field(gt=0)
    training_seed: StrictInt = Field(ge=0)
    lateral_max_offset_m: StrictFloat = Field(gt=0.0)
    longitudinal_max_speed_fraction: StrictFloat = Field(gt=0.0, lt=1.0)


class PPOStabilityStudyConfig(_StrictModel):
    version: Literal[1]
    study_name: str = Field(min_length=1)
    base_training_config: str = Field(min_length=1)
    sampler_seed: StrictInt = Field(ge=0)
    trial_count: StrictInt = Field(gt=0)
    stage_a_update_count: StrictInt = Field(gt=0)
    stage_a_training_seed: StrictInt = Field(ge=0)
    transitions_per_scenario: StrictInt = Field(gt=0)
    training_maps: list[str] = Field(min_length=1)
    training_map_seeds: list[StrictInt] = Field(min_length=1)
    search: SearchSpace
    pruning: PruningConfig
    stage_b: StageConfig
    stage_c: StageConfig
    evaluation: EvaluationConfig
    diagnostics: DiagnosticConfig

    @model_validator(mode="after")
    def validate_study(self) -> PPOStabilityStudyConfig:
        if self.search.learning_rate_min >= self.search.learning_rate_max:
            raise ValueError("learning-rate search bounds must be increasing")
        if self.search.target_kl_min >= self.search.target_kl_max:
            raise ValueError("target-KL search bounds must be increasing")
        if len(set(self.stage_b.training_seeds)) != len(self.stage_b.training_seeds):
            raise ValueError("Stage B training seeds must be unique")
        if len(set(self.stage_c.training_seeds)) != len(self.stage_c.training_seeds):
            raise ValueError("Stage C training seeds must be unique")
        return self


class TrialParameters(_StrictModel):
    learning_rate: StrictFloat = Field(gt=0.0)
    epochs: StrictInt = Field(gt=0)
    batch_size: StrictInt = Field(gt=0)
    minibatch_size: StrictInt = Field(gt=0)
    target_kl: StrictFloat = Field(gt=0.0)

    @property
    def valid_minibatch(self) -> bool:
        return self.minibatch_size <= self.batch_size and not (
            self.batch_size % self.minibatch_size
        )


class StabilityViolation(RuntimeError):
    """Expected domain-specific instability detected during a validation run."""


class StabilityMonitor:
    """Accumulate the registered stability objective and hard-prune conditions."""

    def __init__(self, config: PruningConfig) -> None:
        self.config = config
        self.updates: list[TrainingUpdateSummary] = []
        self.baseline_episode_length: float | None = None
        self.minimum_episode_retention = math.inf

    def add(self, update: TrainingUpdateSummary) -> str | None:
        self.updates.append(update)
        if self.baseline_episode_length is None:
            self.baseline_episode_length = update.mean_episode_length
        retention = update.mean_episode_length / self.baseline_episode_length
        self.minimum_episode_retention = min(self.minimum_episode_retention, retention)
        if retention < self.config.minimum_episode_length_retention:
            return "episode_length_below_minimum_retention"
        window = self.updates[-self.config.consecutive_update_count :]
        if len(window) < self.config.consecutive_update_count:
            return None
        if all(
            item.out_of_road_count / item.sample_count >= self.config.out_of_road_fraction
            for item in window
        ):
            return "sustained_out_of_road"
        if all(item.kl_early_stopped for item in window):
            return "sustained_target_kl_early_stop"
        if self.config.clip_fraction is not None and all(
            item.mean_clip_fraction >= self.config.clip_fraction for item in window
        ):
            return "sustained_clip_fraction"
        return None

    def intermediate_payload(self) -> dict[str, object]:
        latest = self.updates[-1]
        return {
            "update": latest.update_index,
            "minimum_episode_length_retention": self.minimum_episode_retention,
            "out_of_road_fraction": latest.out_of_road_count / latest.sample_count,
            "mean_approximate_kl": latest.mean_approximate_kl,
            "mean_clip_fraction": latest.mean_clip_fraction,
            "kl_early_stopped": latest.kl_early_stopped,
        }


def load_stability_config(path: Path) -> PPOStabilityStudyConfig:
    return PPOStabilityStudyConfig.model_validate(load_resolved_yaml_mapping(path))


def sample_trial_parameters(trial: Trial, config: SearchSpace) -> TrialParameters:
    return TrialParameters(
        learning_rate=trial.suggest_float(
            "learning_rate", config.learning_rate_min, config.learning_rate_max, log=True
        ),
        epochs=trial.suggest_categorical("epochs", config.epochs),
        batch_size=trial.suggest_categorical("batch_size", config.batch_sizes),
        minibatch_size=trial.suggest_categorical("minibatch_size", config.minibatch_sizes),
        target_kl=trial.suggest_float(
            "target_kl", config.target_kl_min, config.target_kl_max, log=True
        ),
    )


def compose_trial_training_config(
    study: PPOStabilityStudyConfig,
    parameters: TrialParameters,
    *,
    training_seed: int,
    update_count: int,
    gradient_diagnostics: bool = False,
    guidance_range: tuple[float, float] | None = None,
) -> tuple[DictConfig, TrainingJobConfig]:
    scenario_count = parameters.batch_size // study.transitions_per_scenario
    scenarios = _scenarios(
        study.training_maps,
        study.training_map_seeds,
        limit=scenario_count,
    )
    if len(scenarios) != scenario_count:
        raise ValueError("study does not define enough independent training scenarios")
    overrides = [f"runtime.seed={training_seed}", "training.replay_id=0"]
    if GlobalHydra.instance().is_initialized():
        config = compose(config_name=study.base_training_config, overrides=overrides)
    else:
        with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_ROOT.resolve())):
            config = compose(config_name=study.base_training_config, overrides=overrides)
    optimizer_steps = (
        update_count * parameters.epochs * (parameters.batch_size // parameters.minibatch_size)
    )
    maximum_map_seed = max((*study.training_map_seeds, *study.evaluation.map_seeds))
    with open_dict(config):
        config.training.update_count = update_count
        config.training.transitions_per_environment = study.transitions_per_scenario
        config.ppo.learning_rate = parameters.learning_rate
        config.ppo.epochs = parameters.epochs
        config.ppo.batch_size = parameters.batch_size
        config.ppo.minibatch_size = parameters.minibatch_size
        config.ppo.target_kl = parameters.target_kl
        config.ppo.gradient_diagnostics = gradient_diagnostics
        config.ppo.scheduler_total_optimizer_steps = optimizer_steps
        config.scenarios = [item.model_dump(mode="python") for item in scenarios]
        config.env.horizon = study.transitions_per_scenario
        config.env.num_scenarios = maximum_map_seed + 1
        if guidance_range is not None:
            config.guidance.lateral_max_offset_m = guidance_range[0]
            config.guidance.longitudinal_max_speed_fraction = guidance_range[1]
    return config, parse_training_config(config)


def run_stage_a(study_path: Path, output_root: Path) -> dict[str, object]:
    config = load_stability_config(study_path)
    _prepare_study_root(study_path, output_root)
    study = _create_study(config, output_root)
    remaining = max(0, config.trial_count - len(study.trials))
    if remaining:
        study.optimize(
            _objective(config, output_root),
            n_trials=remaining,
            # Single-trial failures are recorded as FAIL trials with their
            # failure reason; they must not abort the whole search.
            catch=(Exception,),
        )
    summary = summarize_stage_a(study, config)
    write_json(output_root / "stage-a-summary.json", summary)
    return summary


def summarize_stage_a(study: optuna.Study, config: PPOStabilityStudyConfig) -> dict[str, object]:
    completed = sorted(
        (trial for trial in study.trials if trial.state == TrialState.COMPLETE),
        key=lambda item: (-(item.value or -math.inf), item.number),
    )
    importances: dict[str, float] | None = None
    importance_error: str | None = None
    if len(completed) >= 2:
        try:
            importances = get_param_importances(study)
        except (ValueError, ImportError) as error:
            importance_error = str(error)
    counts = {
        state.name.lower(): sum(trial.state == state for trial in study.trials)
        for state in (TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL)
    }
    return {
        "study_name": study.study_name,
        "trial_count": len(study.trials),
        "state_counts": counts,
        "stability_counts": {
            "stable": counts["complete"],
            "unstable": counts["pruned"] + counts["fail"],
        },
        "top_configs": [
            _trial_payload(item) for item in completed[: config.stage_b.top_config_count]
        ],
        "parameter_importances": importances,
        "parameter_importance_error": importance_error,
        "stability_region": [_trial_payload(item) for item in study.trials],
    }


def run_validation_stage(
    study_path: Path,
    output_root: Path,
    stage: Literal["b", "c"],
) -> dict[str, object]:
    config = load_stability_config(study_path)
    optuna_study = _create_study(config, output_root)
    stage_config = config.stage_b if stage == "b" else config.stage_c
    candidates = _validation_candidates(optuna_study, config, output_root, stage)
    stage_root = output_root / f"stage-{stage}"
    stage_root.mkdir(parents=True, exist_ok=False)
    records: list[dict[str, object]] = []
    for trial in candidates[: stage_config.top_config_count]:
        parameters = TrialParameters.model_validate(trial.params)
        for training_seed in stage_config.training_seeds:
            records.append(
                _run_validation(
                    config,
                    parameters,
                    config_id=trial.number,
                    stage=stage,
                    training_seed=training_seed,
                    update_count=stage_config.update_count,
                    output_dir=stage_root / f"config-{trial.number:04d}" / f"seed-{training_seed}",
                )
            )
    promoted = _rank_validation_configs(records, len(stage_config.training_seeds))
    summary = {
        "stage": stage,
        "records": records,
        "ranked_config_ids": promoted,
        "promoted_config_ids": (
            promoted[: config.stage_c.top_config_count] if stage == "b" else []
        ),
    }
    write_json(stage_root / "summary.json", summary)
    return summary


def run_diagnostics(
    study_path: Path,
    output_root: Path,
    diagnostic: Literal["gradient", "guidance"],
) -> dict[str, object]:
    config = load_stability_config(study_path)
    optuna_study = _create_study(config, output_root)
    candidates = sorted(
        (trial for trial in optuna_study.trials if trial.state == TrialState.COMPLETE),
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
        guidance_range = None
        if variant == "reduced":
            guidance_range = (
                config.diagnostics.lateral_max_offset_m,
                config.diagnostics.longitudinal_max_speed_fraction,
            )
        raw, parsed = compose_trial_training_config(
            config,
            parameters,
            training_seed=config.diagnostics.training_seed,
            update_count=config.diagnostics.update_count,
            gradient_diagnostics=diagnostic == "gradient",
            guidance_range=guidance_range,
        )
        run_dir = diagnostic_root / variant
        run_dir.mkdir(parents=True, exist_ok=False)
        OmegaConf.save(raw, run_dir / "resolved_config.yaml", resolve=True)
        summary = train(parsed, run_dir)
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


def _objective(config: PPOStabilityStudyConfig, output_root: Path) -> Callable[[Trial], float]:
    def objective(trial: Trial) -> float:
        trial_dir = output_root / "stage-a" / f"trial-{trial.number:04d}"
        trial_dir.mkdir(parents=True, exist_ok=False)
        parameters = sample_trial_parameters(trial, config.search)
        trial.set_user_attr("output_dir", str(trial_dir))
        if not parameters.valid_minibatch:
            trial.set_user_attr("prune_reason", "invalid_batch_minibatch_combination")
            _write_trial_record(trial_dir, trial, parameters, "pruned")
            raise optuna.TrialPruned("invalid_batch_minibatch_combination")
        raw, parsed = compose_trial_training_config(
            config,
            parameters,
            training_seed=config.stage_a_training_seed,
            update_count=config.stage_a_update_count,
        )
        OmegaConf.save(raw, trial_dir / "resolved_config.yaml", resolve=True)
        monitor = StabilityMonitor(config.pruning)

        def observe(update: TrainingUpdateSummary) -> None:
            reason = monitor.add(update)
            if reason is not None:
                trial.set_user_attr("prune_reason", reason)
                trial.set_user_attr("last_stability_metrics", monitor.intermediate_payload())
                raise optuna.TrialPruned(reason)
            update_number = update.update_index + 1
            if (
                update_number % config.pruning.report_interval_updates == 0
                or update_number == config.stage_a_update_count
            ):
                trial.report(monitor.minimum_episode_retention, step=update_number)
                trial.set_user_attr("last_stability_metrics", monitor.intermediate_payload())
                if trial.should_prune():
                    trial.set_user_attr("prune_reason", "median_pruner")
                    raise optuna.TrialPruned("median_pruner")

        try:
            summary = train(parsed, trial_dir, update_observer=observe)
        except FloatingPointError as error:
            trial.set_user_attr("prune_reason", "non_finite")
            trial.set_user_attr("failure", str(error))
            _write_trial_record(trial_dir, trial, parameters, "pruned")
            raise optuna.TrialPruned("non_finite") from error
        except optuna.TrialPruned:
            _write_trial_record(trial_dir, trial, parameters, "pruned")
            raise
        except Exception as error:
            trial.set_user_attr("failure", f"{type(error).__name__}: {error}")
            _write_trial_record(trial_dir, trial, parameters, "failed")
            raise
        trial.set_user_attr("summary_path", str(trial_dir / "summary.json"))
        trial.set_user_attr("final_policy_path", str(trial_dir / "policy-final.pt"))
        trial.set_user_attr("minimum_episode_length_retention", monitor.minimum_episode_retention)
        _write_trial_record(trial_dir, trial, parameters, "complete")
        if len(summary.updates) != config.stage_a_update_count:
            raise RuntimeError("completed Stage A trial has an unexpected update count")
        return monitor.minimum_episode_retention

    return objective


def _run_validation(
    config: PPOStabilityStudyConfig,
    parameters: TrialParameters,
    *,
    config_id: int,
    stage: Literal["b", "c"],
    training_seed: int,
    update_count: int,
    output_dir: Path,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=False)
    raw, parsed = compose_trial_training_config(
        config,
        parameters,
        training_seed=training_seed,
        update_count=update_count,
    )
    OmegaConf.save(raw, output_dir / "resolved_config.yaml", resolve=True)
    monitor = StabilityMonitor(config.pruning)

    def observe(update: TrainingUpdateSummary) -> None:
        reason = monitor.add(update)
        if reason is not None:
            raise StabilityViolation(reason)

    try:
        train(parsed, output_dir, update_observer=observe)
    except StabilityViolation as error:
        return {
            "stage": stage,
            "config_id": config_id,
            "training_seed": training_seed,
            "state": "unstable",
            "reason": str(error),
            "minimum_episode_length_retention": monitor.minimum_episode_retention,
            "evaluation": None,
            "output_dir": str(output_dir),
        }
    except Exception as error:
        return {
            "stage": stage,
            "config_id": config_id,
            "training_seed": training_seed,
            "state": "failed",
            "reason": f"{type(error).__name__}: {error}",
            "minimum_episode_length_retention": monitor.minimum_episode_retention,
            "evaluation": None,
            "output_dir": str(output_dir),
        }
    evaluation = _evaluate_validation_policy(config, parsed, output_dir)
    return {
        "stage": stage,
        "config_id": config_id,
        "training_seed": training_seed,
        "state": "complete",
        "reason": None,
        "minimum_episode_length_retention": monitor.minimum_episode_retention,
        "evaluation": evaluation.model_dump(mode="json"),
        "output_dir": str(output_dir),
    }


def _evaluate_validation_policy(
    study: PPOStabilityStudyConfig,
    training: TrainingJobConfig,
    run_dir: Path,
) -> PolicyEvaluationComparison:
    scenarios = _scenarios(
        study.evaluation.maps,
        study.evaluation.map_seeds,
        limit=None,
    )
    evaluation_root = run_dir / "evaluation"
    initial = evaluate_policy_checkpoint(
        training,
        run_dir / "policy-initial.pt",
        label="initial",
        scenarios=scenarios,
        transitions_per_scenario=study.evaluation.transitions_per_scenario,
        evaluation_seed=study.evaluation.seed,
        output_dir=evaluation_root / "initial",
    )
    final = evaluate_policy_checkpoint(
        training,
        run_dir / "policy-final.pt",
        label="final",
        scenarios=scenarios,
        transitions_per_scenario=study.evaluation.transitions_per_scenario,
        evaluation_seed=study.evaluation.seed,
        output_dir=evaluation_root / "final",
    )
    comparison = compare_policy_evaluations(
        initial,
        final,
        minimum_retention=study.evaluation.minimum_retention,
    )
    write_json(evaluation_root / "comparison.json", comparison)
    return comparison


def _rank_validation_configs(
    records: list[dict[str, object]], required_seed_count: int
) -> list[int]:
    by_config: dict[int, list[dict[str, object]]] = {}
    for record in records:
        by_config.setdefault(int(record["config_id"]), []).append(record)
    ranked: list[tuple[float, float, int]] = []
    for config_id, items in by_config.items():
        if len(items) != required_seed_count or any(item["state"] != "complete" for item in items):
            continue
        evaluations = [item["evaluation"] for item in items]
        if any(not isinstance(value, dict) or not value["passed"] for value in evaluations):
            continue
        worst_training = min(float(item["minimum_episode_length_retention"]) for item in items)
        mean_route = sum(float(value["route_progress_retention"]) for value in evaluations) / len(
            evaluations
        )
        ranked.append((-worst_training, -mean_route, config_id))
    return [item[2] for item in sorted(ranked)]


def _validation_candidates(
    study: optuna.Study,
    config: PPOStabilityStudyConfig,
    output_root: Path,
    stage: Literal["b", "c"],
) -> list[FrozenTrial]:
    if stage == "b":
        return sorted(
            (trial for trial in study.trials if trial.state == TrialState.COMPLETE),
            key=lambda item: (-(item.value or -math.inf), item.number),
        )[: config.stage_b.top_config_count]
    summary_path = output_root / "stage-b" / "summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    promoted = payload["promoted_config_ids"]
    trials = {trial.number: trial for trial in study.trials}
    return [trials[int(number)] for number in promoted]


def _create_study(config: PPOStabilityStudyConfig, output_root: Path) -> optuna.Study:
    database_path = (output_root / "study.db").resolve().as_posix()
    storage = f"sqlite:///{database_path}"
    return optuna.create_study(
        study_name=config.study_name,
        storage=storage,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=config.sampler_seed),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=config.pruning.median_startup_trials,
            n_warmup_steps=config.pruning.median_warmup_updates,
            interval_steps=config.pruning.report_interval_updates,
        ),
        load_if_exists=True,
    )


def _prepare_study_root(study_path: Path, output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "study_manifest.yaml"
    incoming = OmegaConf.to_container(OmegaConf.load(study_path), resolve=True)
    if manifest_path.exists():
        existing = OmegaConf.to_container(OmegaConf.load(manifest_path), resolve=True)
        if existing != incoming:
            raise ValueError("existing study root uses a different manifest")
    else:
        OmegaConf.save(OmegaConf.load(study_path), manifest_path, resolve=True)
    (output_root / "stage-a").mkdir(exist_ok=True)


def _write_trial_record(
    trial_dir: Path,
    trial: Trial,
    parameters: TrialParameters,
    state: Literal["complete", "pruned", "failed"],
) -> None:
    write_json(
        trial_dir / "trial.json",
        {
            "trial_number": trial.number,
            "state": state,
            "parameters": parameters.model_dump(mode="json"),
            "user_attributes": dict(trial.user_attrs),
            "resolved_config_path": str(trial_dir / "resolved_config.yaml"),
            "training_state_checkpoint": str(trial_dir / "training-state.ckpt"),
        },
    )


def _trial_payload(trial: FrozenTrial) -> dict[str, object]:
    return {
        "trial_number": trial.number,
        "state": trial.state.name.lower(),
        "value": trial.value,
        "parameters": dict(trial.params),
        "user_attributes": dict(trial.user_attrs),
    }


def _scenarios(
    maps: list[str], seeds: list[int], *, limit: int | None
) -> tuple[ScenarioConfig, ...]:
    values = tuple(
        ScenarioConfig(name=f"{map_name.lower()}_s{seed}", map=map_name, seed=seed)
        for seed in seeds
        for map_name in maps
    )
    return values if limit is None else values[:limit]


def _summary_retention(updates: tuple[TrainingUpdateSummary, ...]) -> float:
    baseline = updates[0].mean_episode_length
    return min(item.mean_episode_length / baseline for item in updates)


def main() -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    load_local_environment(LOCAL_ENVIRONMENT_PATH)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", choices=("stage-a", "stage-b", "stage-c", "diagnose", "summarize")
    )
    parser.add_argument("--study", type=Path, default=DEFAULT_STUDY)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--diagnostic", choices=("gradient", "guidance"))
    args = parser.parse_args()
    if args.command == "stage-a":
        payload = run_stage_a(args.study.resolve(), args.output_root.resolve())
    elif args.command == "stage-b":
        payload = run_validation_stage(args.study.resolve(), args.output_root.resolve(), "b")
    elif args.command == "stage-c":
        payload = run_validation_stage(args.study.resolve(), args.output_root.resolve(), "c")
    elif args.command == "diagnose":
        if args.diagnostic is None:
            parser.error("diagnose requires --diagnostic")
        payload = run_diagnostics(args.study.resolve(), args.output_root.resolve(), args.diagnostic)
    else:
        study_config = load_stability_config(args.study.resolve())
        payload = summarize_stage_a(
            _create_study(study_config, args.output_root.resolve()), study_config
        )
        write_json(args.output_root.resolve() / "stage-a-summary.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
