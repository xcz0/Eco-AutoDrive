"""PPO-stability acceptance rules over generic closed-loop evaluation artifacts."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt

from eco_planner.artifacts import write_json
from eco_planner.envs.array_types import BatchObservation
from eco_planner.evaluation import (
    CompletedEpisodeSummary,
    EvaluationDecision,
    EvaluationJobConfig,
    JobSummary,
    ScenarioConfig,
    run_evaluation_agent,
)
from eco_planner.experiments.ppo_stability.config import (
    PPOStabilityStudyConfig,
    TrialParameters,
    scenarios,
)
from eco_planner.experiments.ppo_stability.monitor import StabilityMonitor, StabilityViolation
from eco_planner.experiments.ppo_stability.search import compose_trial_training_config
from eco_planner.jobs import run_training_job
from eco_planner.models import (
    CheckpointLoadReport,
    GuidanceConfig,
    OfficialDiffusionPlannerConfig,
    SamplerReport,
)
from eco_planner.rl.artifacts import TrainingUpdateSummary, policy_state_hash
from eco_planner.rl.config import TrainingJobConfig
from eco_planner.rl.optimization import load_exploration_policy_checkpoint
from eco_planner.rl.rollout import FabricRolloutRuntime, create_fabric_rollout_runtime
from eco_planner.runtime.fabric import InferenceRuntimeReport

_EVALUATION_SEED_NAMESPACE = 7_602_024


@dataclass(frozen=True)
class _PPOCheckpointEvaluationAgent:
    """Adapt one PPO guidance checkpoint to the generic evaluation engine."""

    runtime: FabricRolloutRuntime
    noise_seeds: tuple[int, ...]

    @property
    def planner_config(self) -> OfficialDiffusionPlannerConfig:
        return self.runtime.planner_config

    @property
    def report(self) -> InferenceRuntimeReport:
        return self.runtime.report

    @property
    def checkpoint_report(self) -> CheckpointLoadReport:
        return self.runtime.checkpoint_report

    @property
    def sampler_report(self) -> SamplerReport:
        return self.runtime.sampler_report

    @property
    def guidance_config(self) -> GuidanceConfig:
        return self.runtime.guidance_config

    @property
    def guided(self) -> bool:
        return True

    def new_noise_generator(self, scenario_index: int) -> torch.Generator:
        return self.runtime.new_noise_generator(self.noise_seed(scenario_index))

    def noise_seed(self, scenario_index: int) -> int:
        return self.noise_seeds[scenario_index]

    def decide_batch(
        self,
        observation: BatchObservation,
        generators: Sequence[torch.Generator],
    ) -> EvaluationDecision:
        return self.runtime.decide_batch_mean(observation, tuple(generators))


class _ArtifactModel(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid", allow_inf_nan=False)


class PolicyEvaluationSummary(_ArtifactModel):
    """Acceptance inputs projected from one ordinary generic evaluation job."""

    checkpoint_label: Literal["initial", "final"]
    checkpoint_path: str
    policy_hash: str = Field(min_length=64, max_length=64)
    evaluation_seed: StrictInt = Field(ge=0)
    scenarios: tuple[str, ...]
    noise_seeds: tuple[StrictInt, ...]
    transition_count: StrictInt = Field(gt=0)
    episode_count: StrictInt = Field(gt=0)
    mean_episode_length: StrictFloat = Field(gt=0.0)
    collision_count: StrictInt = Field(ge=0)
    out_of_road_count: StrictInt = Field(ge=0)
    route_completion_delta: StrictFloat
    distance_m: StrictFloat = Field(ge=0.0)
    mean_speed_mps: StrictFloat = Field(ge=0.0)
    stopped_fraction: StrictFloat = Field(ge=0.0, le=1.0)


class PolicyEvaluationComparison(_ArtifactModel):
    initial: PolicyEvaluationSummary
    final: PolicyEvaluationSummary
    episode_length_retention: StrictFloat = Field(ge=0.0)
    route_progress_retention: StrictFloat = Field(ge=0.0)
    collision_count_not_increased: bool
    out_of_road_count_not_increased: bool
    passed: bool


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
    job = run_evaluation_agent(
        _evaluation_job_config(config, scenarios, transitions_per_scenario),
        output_dir,
        _PPOCheckpointEvaluationAgent(runtime, noise_seeds),
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


def compare_policy_evaluations(
    initial: PolicyEvaluationSummary,
    final: PolicyEvaluationSummary,
    *,
    minimum_retention: float = 0.9,
) -> PolicyEvaluationComparison:
    """Apply PPO's retention and safety gates; generic execution owns all metrics."""

    if (
        initial.evaluation_seed != final.evaluation_seed
        or initial.scenarios != final.scenarios
        or initial.noise_seeds != final.noise_seeds
    ):
        raise ValueError("policy evaluations must use identical scenarios and diffusion seeds")
    if initial.route_completion_delta <= 0.0:
        raise ValueError("initial policy evaluation must make positive route progress")
    episode_retention = final.mean_episode_length / initial.mean_episode_length
    route_retention = final.route_completion_delta / initial.route_completion_delta
    collision_ok = final.collision_count <= initial.collision_count
    out_of_road_ok = final.out_of_road_count <= initial.out_of_road_count
    return PolicyEvaluationComparison(
        initial=initial,
        final=final,
        episode_length_retention=float(episode_retention),
        route_progress_retention=float(route_retention),
        collision_count_not_increased=collision_ok,
        out_of_road_count_not_increased=out_of_road_ok,
        passed=(
            collision_ok
            and out_of_road_ok
            and episode_retention >= minimum_retention
            and route_retention >= minimum_retention
        ),
    )


def _evaluation_job_config(
    training: TrainingJobConfig,
    scenarios: tuple[ScenarioConfig, ...],
    transitions_per_scenario: int,
) -> EvaluationJobConfig:
    environment = dict(training.env)
    environment.update(
        {
            "execution_mode": "evaluation",
            "trajectory_execution_steps": 5,
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
