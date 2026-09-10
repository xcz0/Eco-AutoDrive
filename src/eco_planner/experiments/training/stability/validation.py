"""PPO-stability acceptance rules over generic closed-loop evaluation artifacts."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Literal

import numpy as np

from eco_planner.artifacts import write_json
from eco_planner.configuration import ScenarioConfig
from eco_planner.evaluation import (
    CompletedEpisodeSummary,
    EvaluationJobConfig,
    JobSummary,
    PolicyCheckpointEvaluationAgent,
    PolicyCheckpointProvenance,
    run_evaluation_agent,
)
from eco_planner.jobs import run_training_job
from eco_planner.rl import (
    TrainingJobConfig,
    TrainingUpdateSummary,
    create_fabric_rollout_runtime,
    policy_state_hash,
)
from eco_planner.rl.optimization import load_exploration_policy_checkpoint

from .comparison import (
    PolicyEvaluationComparison,
    PolicyEvaluationSummary,
    compare_policy_evaluations,
)
from .composition import compose_trial_training_config
from .config import (
    PPOStabilityStudyConfig,
    TrialParameters,
    scenarios,
)
from .monitor import StabilityMonitor, StabilityViolation

_EVALUATION_SEED_NAMESPACE = 7_602_024


def evaluate_policy_checkpoint(
    config: TrainingJobConfig,
    checkpoint_path: Path,
    *,
    label: Literal["initial", "final"],
    scenarios: tuple[ScenarioConfig, ...],
    transitions_per_scenario: int,
    evaluation_seed: int,
    output_dir: Path,
) -> PolicyEvaluationSummary:
    """Persist a deterministic policy-mean checkpoint run through generic evaluation."""

    if not scenarios:
        raise ValueError("policy evaluation requires at least one scenario")
    if type(transitions_per_scenario) is not int or transitions_per_scenario <= 0:
        raise ValueError("transitions_per_scenario must be a positive integer")
    noise_seeds, policy_seeds = _derive_evaluation_seeds(evaluation_seed, len(scenarios))
    runtime = create_fabric_rollout_runtime(
        config.runtime,
        config.sampler,
        config.guidance,
        config.policy,
        Path(config.model.args_path),
        Path(config.model.checkpoint_path),
        policy_seeds[0],
        planner_compile_mode=config.training.planner_compile_mode,
    )
    load_exploration_policy_checkpoint(checkpoint_path, runtime.policy)
    agent = PolicyCheckpointEvaluationAgent(
        runtime=runtime,
        noise_seeds=noise_seeds,
        policy_checkpoint=PolicyCheckpointProvenance(
            label=label,
            path=str(checkpoint_path),
            policy_hash=policy_state_hash(runtime.policy),
        ),
    )
    job = run_evaluation_agent(
        _evaluation_job_config(config, scenarios, transitions_per_scenario),
        output_dir,
        agent,
    )
    return _summary_from_job(
        job,
        label=label,
        checkpoint_path=checkpoint_path,
        policy_hash=policy_state_hash(runtime.policy),
        evaluation_seed=evaluation_seed,
        scenarios=scenarios,
        noise_seeds=noise_seeds,
    )


def _evaluation_job_config(
    training: TrainingJobConfig,
    scenarios: tuple[ScenarioConfig, ...],
    transitions_per_scenario: int,
) -> EvaluationJobConfig:
    environment = dict(training.env)
    environment.update(
        {
            "horizon": training.training.history_warmup_steps + transitions_per_scenario,
        }
    )
    return EvaluationJobConfig.model_validate(
        {
            "name": f"{training.name}-ppo-checkpoint-evaluation",
            "map_query_radius_m": training.map_query_radius_m,
            "evaluation": {
                "mode": training.training.mode,
                "profile": "ppo_checkpoint",
                "history_warmup_steps": training.training.history_warmup_steps,
                "evaluated_horizon_steps": transitions_per_scenario,
                "execution": {
                    "topology": "serial",
                    "deterministic": training.training.deterministic,
                },
            },
            "env": environment,
            "model": training.model.model_dump(mode="python"),
            "runtime": training.runtime.model_dump(mode="python"),
            "sampler": asdict(training.sampler),
            "guidance": asdict(training.guidance),
            "scenarios": [item.model_dump(mode="python") for item in scenarios],
            "video": {
                "enabled": False,
                "fps": 2,
                "screen_width": 32,
                "screen_height": 32,
                "film_width": 32,
                "film_height": 32,
                "scaling": 1.0,
            },
        }
    )


def _summary_from_job(
    job: JobSummary,
    *,
    label: Literal["initial", "final"],
    checkpoint_path: Path,
    policy_hash: str,
    evaluation_seed: int,
    scenarios: tuple[ScenarioConfig, ...],
    noise_seeds: tuple[int, ...],
) -> PolicyEvaluationSummary:
    episodes = tuple(item for item in job.episodes if isinstance(item, CompletedEpisodeSummary))
    if len(episodes) != len(job.episodes):
        raise RuntimeError("PPO checkpoint evaluation did not complete every generic episode")
    transition_count = sum(item.simulator_steps for item in episodes)
    return PolicyEvaluationSummary(
        checkpoint_label=label,
        checkpoint_path=str(checkpoint_path),
        policy_hash=policy_hash,
        evaluation_seed=evaluation_seed,
        scenarios=tuple(f"{item.name}:{item.map}:{item.seed}" for item in scenarios),
        noise_seeds=noise_seeds,
        transition_count=transition_count,
        episode_count=len(episodes),
        mean_episode_length=float(transition_count / len(episodes)),
        collision_count=sum(item.metrics.collision for item in episodes),
        out_of_road_count=sum(item.metrics.out_of_road for item in episodes),
        route_completion_delta=float(sum(item.metrics.route_completion for item in episodes)),
        distance_m=float(sum(item.metrics.distance_m for item in episodes)),
        mean_speed_mps=float(np.mean([item.metrics.speed_mps.mean for item in episodes])),
        stopped_fraction=float(np.mean([item.metrics.stopped_fraction for item in episodes])),
    )


def _derive_evaluation_seeds(
    evaluation_seed: int, scenario_count: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    sequence = np.random.SeedSequence([_EVALUATION_SEED_NAMESPACE, evaluation_seed])
    values = tuple(
        int(child.generate_state(1, dtype=np.uint32)[0])
        for child in sequence.spawn(2 * scenario_count)
    )
    return values[:scenario_count], values[scenario_count:]


def run_validation(
    config: PPOStabilityStudyConfig,
    parameters: TrialParameters,
    *,
    config_id: int,
    stage: Literal["b", "c"],
    training_seed: int,
    update_count: int,
    output_dir: Path,
) -> dict[str, object]:
    """Train a candidate and apply the registered stability acceptance gates."""

    output_dir.mkdir(parents=True, exist_ok=False)
    raw, parsed = compose_trial_training_config(
        config, parameters, training_seed=training_seed, update_count=update_count
    )
    monitor = StabilityMonitor(config.pruning)

    def observe(update: TrainingUpdateSummary) -> None:
        reason = monitor.add(update)
        if reason is not None:
            raise StabilityViolation(reason)

    try:
        run_training_job(raw, output_dir, update_observer=observe)
    except StabilityViolation as error:
        return _validation_record(
            stage, config_id, training_seed, "unstable", str(error), monitor, output_dir
        )
    except Exception as error:
        return _validation_record(
            stage,
            config_id,
            training_seed,
            "failed",
            f"{type(error).__name__}: {error}",
            monitor,
            output_dir,
        )
    comparison = evaluate_validation_policy(config, parsed, output_dir)
    return {
        **_validation_record(
            stage, config_id, training_seed, "complete", None, monitor, output_dir
        ),
        "evaluation": comparison.model_dump(mode="json"),
    }


def evaluate_validation_policy(
    study: PPOStabilityStudyConfig,
    training: TrainingJobConfig,
    run_dir: Path,
) -> PolicyEvaluationComparison:
    """Evaluate initial/final policy artifacts through the shared engine."""

    held_out_scenarios = scenarios(study.evaluation.maps, study.evaluation.map_seeds, limit=None)
    evaluation_root = run_dir / "evaluation"
    initial = evaluate_policy_checkpoint(
        training,
        run_dir / "policy-initial.pt",
        label="initial",
        scenarios=held_out_scenarios,
        transitions_per_scenario=study.evaluation.transitions_per_scenario,
        evaluation_seed=study.evaluation.seed,
        output_dir=evaluation_root / "initial",
    )
    final = evaluate_policy_checkpoint(
        training,
        run_dir / "policy-final.pt",
        label="final",
        scenarios=held_out_scenarios,
        transitions_per_scenario=study.evaluation.transitions_per_scenario,
        evaluation_seed=study.evaluation.seed,
        output_dir=evaluation_root / "final",
    )
    comparison = compare_policy_evaluations(
        initial, final, minimum_retention=study.evaluation.minimum_retention
    )
    write_json(evaluation_root / "comparison.json", comparison)
    return comparison


def _validation_record(
    stage: Literal["b", "c"],
    config_id: int,
    training_seed: int,
    state: str,
    reason: str | None,
    monitor: StabilityMonitor,
    output_dir: Path,
) -> dict[str, object]:
    return {
        "stage": stage,
        "config_id": config_id,
        "training_seed": training_seed,
        "state": state,
        "reason": reason,
        "minimum_episode_length_retention": monitor.minimum_episode_retention,
        "evaluation": None,
        "output_dir": str(output_dir),
    }
