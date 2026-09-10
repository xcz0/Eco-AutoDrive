from __future__ import annotations

from omegaconf import DictConfig, open_dict

from eco_planner.jobs import compose_job_config
from eco_planner.rl import TrainingJobConfig, parse_training_config

from .config import (
    PPOStabilityStudyConfig,
    TrialParameters,
    scenarios,
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
