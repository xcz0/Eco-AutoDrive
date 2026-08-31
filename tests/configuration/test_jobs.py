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
from eco_planner.evaluation.config import EvaluationJobConfig, parse_evaluation_config
from eco_planner.rl.config import (
    RolloutJobConfig,
    TrainingJobConfig,
    parse_rollout_config,
    parse_training_config,
)

ComposeConfig = Callable[[str, list[str] | None], DictConfig]


@pytest.mark.parametrize(
    "config_name",
    [
        "jobs/evaluation/no_traffic/smoke",
        "jobs/evaluation/traffic/smoke",
        "jobs/evaluation/traffic/matrix",
    ],
)
def test_evaluation_jobs_compose_into_typed_boundary(
    monkeypatch: pytest.MonkeyPatch,
    compose_config: ComposeConfig,
    config_name: str,
) -> None:
    monkeypatch.setenv("MACHINE_NAME", "rtx3050_laptop")

    parsed = parse_evaluation_config(compose_config(config_name, None))

    assert isinstance(parsed, EvaluationJobConfig)


@pytest.mark.smoke
def test_training_jobs_compose_into_typed_boundaries(
    monkeypatch: pytest.MonkeyPatch,
    compose_config: ComposeConfig,
) -> None:
    monkeypatch.setenv("MACHINE_NAME", "rtx3050_laptop")

    training = compose_config(
        "jobs/training/ppo/smoke",
        ["runtime.seed=0", "training.replay_id=0"],
    )
    rollout = compose_config("jobs/training/rollout/smoke", None)

    parsed_training = parse_training_config(training)
    assert isinstance(parsed_training, TrainingJobConfig)
    assert parsed_training.training.planner_compile_mode == "eager"
    compiled_training = parse_training_config(
        compose_config(
            "jobs/training/ppo/smoke",
            [
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
                    "runtime.seed=0",
                    "training.replay_id=0",
                    "training.planner_compile_mode=automatic",
                ],
            )
        )
    assert isinstance(parse_rollout_config(rollout), RolloutJobConfig)


def test_conservative_training_job_composes_into_typed_boundaries(
    monkeypatch: pytest.MonkeyPatch,
    compose_config: ComposeConfig,
) -> None:
    monkeypatch.setenv("MACHINE_NAME", "rtx3050_laptop")

    training = compose_config(
        "jobs/training/ppo/conservative",
        ["runtime.seed=0", "training.replay_id=0"],
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
    monkeypatch: pytest.MonkeyPatch,
    compose_config: ComposeConfig,
) -> None:
    monkeypatch.setenv("MACHINE_NAME", "rtx3050_laptop")

    environment = parse_environment_job(compose_config("jobs/benchmark/environment", None))
    throughput_job, throughput = split_benchmark_config(
        compose_config("jobs/benchmark/throughput", None), ScalingBenchmarkConfig
    )
    rollout_job, rollout = split_benchmark_config(
        compose_config("jobs/benchmark/rollout", None), RolloutBenchmarkConfig
    )

    assert isinstance(environment, EnvironmentBenchmarkJobConfig)
    assert isinstance(throughput, ScalingBenchmarkConfig)
    assert isinstance(parse_evaluation_config(throughput_job), EvaluationJobConfig)
    assert isinstance(rollout, RolloutBenchmarkConfig)
    assert rollout.ppo_epochs == 4
    assert rollout.ppo_minibatch_size == 16
    assert rollout.transitions_per_slot == 16
    assert isinstance(parse_training_config(rollout_job), TrainingJobConfig)
