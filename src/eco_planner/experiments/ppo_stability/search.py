"""Optuna search and typed PPO job composition for stability candidates."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Literal

import optuna
from omegaconf import DictConfig, OmegaConf, open_dict
from optuna.trial import Trial

from eco_planner.artifacts import write_json
from eco_planner.experiments.ppo_stability.config import (
    PPOStabilityStudyConfig,
    SearchSpace,
    TrialParameters,
    scenarios,
)
from eco_planner.experiments.ppo_stability.monitor import StabilityMonitor
from eco_planner.jobs import compose_job_config, run_training_job
from eco_planner.rl.artifacts import TrainingUpdateSummary
from eco_planner.rl.config import TrainingJobConfig, parse_training_config


def sample_trial_parameters(trial: Trial, config: SearchSpace) -> TrialParameters:
    """Sample one concrete candidate from the registered search space."""

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
    """Compose and strictly parse the job used for one candidate run."""

    scenario_count = parameters.batch_size // study.transitions_per_scenario
    selected_scenarios = scenarios(
        study.training_maps, study.training_map_seeds, limit=scenario_count
    )
    if len(selected_scenarios) != scenario_count:
        raise ValueError("study does not define enough independent training scenarios")
    config = compose_job_config(
        study.base_training_config, [f"runtime.seed={training_seed}", "training.replay_id=0"]
    )
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
        config.scenarios = [item.model_dump(mode="python") for item in selected_scenarios]
        config.env.horizon = study.transitions_per_scenario
        config.env.num_scenarios = maximum_map_seed + 1
        if guidance_range is not None:
            config.guidance.lateral_max_offset_m = guidance_range[0]
            config.guidance.longitudinal_max_speed_fraction = guidance_range[1]
    return config, parse_training_config(config)


def create_study(config: PPOStabilityStudyConfig, output_root: Path) -> optuna.Study:
    """Open the persistent Optuna study with its registered sampler/pruner."""

    database_path = (output_root / "study.db").resolve().as_posix()
    return optuna.create_study(
        study_name=config.study_name,
        storage=f"sqlite:///{database_path}",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=config.sampler_seed),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=config.pruning.median_startup_trials,
            n_warmup_steps=config.pruning.median_warmup_updates,
            interval_steps=config.pruning.report_interval_updates,
        ),
        load_if_exists=True,
    )


def prepare_study_root(study_path: Path, output_root: Path) -> None:
    """Create or verify the persistent root for an identical manifest."""

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


def make_objective(config: PPOStabilityStudyConfig, output_root: Path) -> Callable[[Trial], float]:
    """Return the Stage A objective with registered runtime pruning."""

    def objective(trial: Trial) -> float:
        trial_dir = output_root / "stage-a" / f"trial-{trial.number:04d}"
        trial_dir.mkdir(parents=True, exist_ok=False)
        parameters = sample_trial_parameters(trial, config.search)
        trial.set_user_attr("output_dir", str(trial_dir))
        if not parameters.valid_minibatch:
            trial.set_user_attr("prune_reason", "invalid_batch_minibatch_combination")
            write_trial_record(trial_dir, trial, parameters, "pruned")
            raise optuna.TrialPruned("invalid_batch_minibatch_combination")
        raw, _ = compose_trial_training_config(
            config,
            parameters,
            training_seed=config.stage_a_training_seed,
            update_count=config.stage_a_update_count,
        )
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
            summary = run_training_job(raw, trial_dir, update_observer=observe)
        except FloatingPointError as error:
            trial.set_user_attr("prune_reason", "non_finite")
            trial.set_user_attr("failure", str(error))
            write_trial_record(trial_dir, trial, parameters, "pruned")
            raise optuna.TrialPruned("non_finite") from error
        except optuna.TrialPruned:
            write_trial_record(trial_dir, trial, parameters, "pruned")
            raise
        except Exception as error:
            trial.set_user_attr("failure", f"{type(error).__name__}: {error}")
            write_trial_record(trial_dir, trial, parameters, "failed")
            raise
        trial.set_user_attr("summary_path", str(trial_dir / "summary.json"))
        trial.set_user_attr("final_policy_path", str(trial_dir / "policy-final.pt"))
        trial.set_user_attr("minimum_episode_length_retention", monitor.minimum_episode_retention)
        write_trial_record(trial_dir, trial, parameters, "complete")
        if len(summary.updates) != config.stage_a_update_count:
            raise RuntimeError("completed Stage A trial has an unexpected update count")
        return monitor.minimum_episode_retention

    return objective


def write_trial_record(
    trial_dir: Path,
    trial: Trial,
    parameters: TrialParameters,
    state: Literal["complete", "pruned", "failed"],
) -> None:
    """Write one readable companion record for a persistent Optuna trial."""

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
