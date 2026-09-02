from __future__ import annotations

from collections.abc import Callable

import pytest
from omegaconf import DictConfig

from eco_planner.benchmarking.config import (
    EnvironmentBenchmarkJobConfig,
    RolloutBenchmarkConfig,
    ScalingBenchmarkConfig,
    parse_environment_job,
    split_benchmark_config,
)
from eco_planner.evaluation import EvaluationJobConfig, parse_evaluation_config
from eco_planner.rl.config import (
    RolloutJobConfig,
    TrainingJobConfig,
    parse_rollout_config,
    parse_training_config,
)

ComposeConfig = Callable[[str, list[str] | None], DictConfig]
RESOURCE_OVERRIDE = "components/resources=rtx3050_laptop"


@pytest.mark.parametrize(
    "config_name",
    [
        "jobs/evaluation/no_traffic/smoke",
        "jobs/evaluation/traffic/smoke",
        "jobs/evaluation/traffic/matrix",
    ],
)
def test_evaluation_jobs_compose_into_typed_boundary(
    compose_config: ComposeConfig,
    config_name: str,
) -> None:
    parsed = parse_evaluation_config(compose_config(config_name, [RESOURCE_OVERRIDE]))

    assert isinstance(parsed, EvaluationJobConfig)


@pytest.mark.smoke
def test_training_jobs_compose_into_typed_boundaries(
    compose_config: ComposeConfig,
) -> None:
    training = compose_config(
        "jobs/training/ppo/smoke",
        [RESOURCE_OVERRIDE, "runtime.seed=0", "training.replay_id=0"],
    )
    rollout = compose_config("jobs/training/rollout/smoke", None)

    parsed_training = parse_training_config(training)
    assert isinstance(parsed_training, TrainingJobConfig)
    assert parsed_training.training.planner_compile_mode == "eager"
    compiled_training = parse_training_config(
        compose_config(
            "jobs/training/ppo/smoke",
            [
                RESOURCE_OVERRIDE,
                "runtime.seed=0",
                "training.replay_id=0",
                "training.planner_compile_mode=dit_reduce_overhead",
            ],
        )
    )
    assert compiled_training.training.planner_compile_mode == "dit_reduce_overhead"
    with pytest.raises(ValueError, match="planner_compile_mode"):
        parse_training_config(
            compose_config(
                "jobs/training/ppo/smoke",
                [
                    RESOURCE_OVERRIDE,
                    "runtime.seed=0",
                    "training.replay_id=0",
                    "training.planner_compile_mode=automatic",
                ],
            )
        )
    assert isinstance(parse_rollout_config(rollout), RolloutJobConfig)


def test_conservative_training_job_composes_into_typed_boundaries(
    compose_config: ComposeConfig,
) -> None:
    training = compose_config(
        "jobs/training/ppo/conservative",
        [RESOURCE_OVERRIDE, "runtime.seed=0", "training.replay_id=0"],
    )
    parsed = parse_training_config(training)

    assert isinstance(parsed, TrainingJobConfig)
    assert parsed.ppo.learning_rate == 2.5e-5
    assert parsed.ppo.epochs == 1
    assert parsed.ppo.batch_size == 256
    assert parsed.ppo.minibatch_size == 128
    assert parsed.ppo.scheduler_total_optimizer_steps == 40
    assert parsed.training.update_count == 20
    assert len(parsed.scenarios) == 16
    expected_batch = len(parsed.scenarios) * parsed.training.transitions_per_environment
    assert parsed.ppo.batch_size == expected_batch


def test_benchmark_jobs_compose_into_typed_boundaries(
    compose_config: ComposeConfig,
) -> None:
    environment = parse_environment_job(compose_config("jobs/benchmark/environment", None))
    throughput_job, throughput = split_benchmark_config(
        compose_config("jobs/benchmark/throughput", [RESOURCE_OVERRIDE]),
        ScalingBenchmarkConfig,
    )
    rollout_job, rollout = split_benchmark_config(
        compose_config("jobs/benchmark/rollout", [RESOURCE_OVERRIDE]),
        RolloutBenchmarkConfig,
    )

    assert isinstance(environment, EnvironmentBenchmarkJobConfig)
    assert isinstance(throughput, ScalingBenchmarkConfig)
    assert isinstance(parse_evaluation_config(throughput_job), EvaluationJobConfig)
    assert isinstance(rollout, RolloutBenchmarkConfig)
    assert rollout.ppo_epochs == 4
    assert rollout.ppo_minibatch_size == 16
    assert rollout.transitions_per_slot == 16
    assert isinstance(parse_training_config(rollout_job), TrainingJobConfig)


def test_semantic_jobs_compose_without_a_machine_profile(
    monkeypatch: pytest.MonkeyPatch,
    compose_config: ComposeConfig,
) -> None:
    monkeypatch.delenv("MACHINE_NAME", raising=False)

    evaluation = parse_evaluation_config(compose_config("jobs/evaluation/no_traffic/smoke", None))
    parallel_evaluation = parse_evaluation_config(
        compose_config("jobs/evaluation/traffic/matrix", None)
    )
    training = parse_training_config(
        compose_config(
            "jobs/training/ppo/smoke",
            ["runtime.seed=0", "training.replay_id=0"],
        )
    )
    throughput_job, _ = split_benchmark_config(
        compose_config("jobs/benchmark/throughput", None), ScalingBenchmarkConfig
    )
    environment = parse_environment_job(compose_config("jobs/benchmark/environment", None))

    assert evaluation.resources is None
    assert evaluation.evaluation.execution.topology == "serial"
    assert parallel_evaluation.resources is None
    assert parallel_evaluation.evaluation.execution.topology == "job_parallel"
    assert training.resources is None
    assert parse_evaluation_config(throughput_job).resources is None
    assert environment.resources is None
