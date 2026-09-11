from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import pytest
from hydra.errors import MissingConfigException
from omegaconf import DictConfig, OmegaConf

from eco_planner import jobs
from eco_planner.benchmarking.config import (
    EnvironmentBenchmarkJobConfig,
    RolloutBenchmarkConfig,
    ScalingBenchmarkConfig,
    parse_environment_job,
    split_benchmark_config,
)
from eco_planner.configuration import (
    load_local_environment,
    with_machine_resource_override,
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


def test_missing_local_environment_is_optional(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("MACHINE_NAME", raising=False)

    load_local_environment(tmp_path / "missing.env")

    assert "MACHINE_NAME" not in os.environ


def test_process_environment_wins_over_local_environment(monkeypatch, tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("MACHINE_NAME=rtx3050_laptop\n", encoding="utf-8")
    monkeypatch.setenv("MACHINE_NAME", "rtx_a4000")

    load_local_environment(env_path)

    assert with_machine_resource_override([]) == ["components/resources=rtx_a4000"]


def test_compose_job_config_uses_the_shared_hydra_boundary(monkeypatch) -> None:
    monkeypatch.setenv("MACHINE_NAME", "rtx3050_laptop")

    config = jobs.compose_job_config(
        "jobs/training/ppo",
        ("runtime.seed=17", "training.replay_id=3"),
    )

    assert config.runtime.seed == 17
    assert config.training.replay_id == 3
    assert config.resources.name == "rtx3050_laptop"
    assert config.resources.rollout_worker_count == 4
    assert config.resources.evaluation_job_worker_count == 2
    assert config.resources.evaluation_vector_env_slots == 4
    assert config.resources.torch_threads_per_worker == 8


def test_compose_job_config_preserves_an_explicit_resource_override(monkeypatch) -> None:
    monkeypatch.setenv("MACHINE_NAME", "rtx3050_laptop")

    config = jobs.compose_job_config(
        "jobs/training/ppo",
        (
            "components/resources=rtx_a4000",
            "runtime.seed=17",
            "training.replay_id=3",
        ),
    )

    assert config.resources.name == "rtx_a4000"


def test_compose_job_config_without_a_machine_profile(monkeypatch) -> None:
    monkeypatch.delenv("MACHINE_NAME", raising=False)

    config = jobs.compose_job_config(
        "jobs/training/ppo",
        ("runtime.seed=17", "training.replay_id=3"),
    )

    assert "resources" not in config


def test_unknown_machine_profile_is_reported_by_hydra(monkeypatch) -> None:
    monkeypatch.setenv("MACHINE_NAME", "unknown-machine")

    with pytest.raises(MissingConfigException, match="components/resources/unknown-machine"):
        jobs.compose_job_config(
            "jobs/training/ppo",
            ("runtime.seed=17", "training.replay_id=3"),
        )


def test_typed_job_runners_parse_before_invoking_domain_execution(
    monkeypatch, tmp_path: Path
) -> None:
    evaluation_config = OmegaConf.create({"evaluation": "raw"})
    training_config = OmegaConf.create({"training": "raw"})
    resource_profile = object()
    evaluation_summary = SimpleNamespace(resources=resource_profile)
    training_summary = SimpleNamespace(resources=resource_profile)
    seen: dict[str, object] = {}

    monkeypatch.setattr(jobs, "parse_evaluation_config", lambda config: evaluation_summary)
    monkeypatch.setattr(
        jobs,
        "run_evaluation",
        lambda config, output_dir: (
            seen.update(evaluation=(config, output_dir)) or evaluation_summary
        ),
    )
    monkeypatch.setattr(jobs, "parse_training_config", lambda config: training_summary)
    monkeypatch.setattr(
        jobs,
        "train",
        lambda config, output_dir, update_observer=None: (
            seen.update(training=(config, output_dir, update_observer)) or training_summary
        ),
    )

    assert jobs.run_evaluation_job(evaluation_config, tmp_path / "evaluation") is evaluation_summary
    assert jobs.run_training_job(training_config, tmp_path / "training") is training_summary
    assert seen["evaluation"] == (evaluation_summary, tmp_path / "evaluation")
    assert seen["training"] == (training_summary, tmp_path / "training", None)
    assert (tmp_path / "training" / "resolved_config.yaml").is_file()


@pytest.mark.parametrize("runner_name", ["run_evaluation_job", "run_training_job"])
def test_job_execution_requires_a_resource_profile(
    monkeypatch, tmp_path: Path, runner_name: str
) -> None:
    config = OmegaConf.create({"job": "raw"})
    parsed = SimpleNamespace(resources=None)
    parse_name = (
        "parse_evaluation_config"
        if runner_name == "run_evaluation_job"
        else "parse_training_config"
    )
    monkeypatch.setattr(jobs, parse_name, lambda raw: parsed)

    with pytest.raises(ValueError, match="execution requires a resource profile"):
        getattr(jobs, runner_name)(config, tmp_path / runner_name)


@pytest.mark.parametrize(
    "config_name",
    [
        "jobs/evaluation/no_traffic_smoke",
        "jobs/evaluation/no_traffic",
        "jobs/evaluation/no_traffic_matrix",
        "jobs/evaluation/traffic_smoke",
        "jobs/evaluation/traffic",
        "jobs/evaluation/traffic_matrix",
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
        "jobs/training/ppo",
        [RESOURCE_OVERRIDE, "runtime.seed=0", "training.replay_id=0"],
    )
    rollout = compose_config("jobs/training/rollout_smoke", None)

    parsed_training = parse_training_config(training)
    assert isinstance(parsed_training, TrainingJobConfig)
    assert parsed_training.training.planner_compile_mode == "eager"
    compiled_training = parse_training_config(
        compose_config(
            "jobs/training/ppo",
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
                "jobs/training/ppo",
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
        "jobs/training/ppo_conservative",
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


@pytest.mark.parametrize(
    "config_name",
    [
        "jobs/training/ppo_energy_smoke",
    ],
)
def test_ppo_job_profiles_compose_into_typed_boundaries(
    compose_config: ComposeConfig,
    config_name: str,
) -> None:
    parsed = parse_training_config(
        compose_config(
            config_name,
            [RESOURCE_OVERRIDE, "runtime.seed=0", "training.replay_id=0"],
        )
    )

    assert isinstance(parsed, TrainingJobConfig)


def test_benchmark_jobs_compose_into_typed_boundaries(
    compose_config: ComposeConfig,
) -> None:
    environment = parse_environment_job(compose_config("jobs/benchmark/environment", None))
    throughput_job, throughput = split_benchmark_config(
        compose_config("jobs/benchmark/throughput", [RESOURCE_OVERRIDE]),
        ScalingBenchmarkConfig,
    )
    throughput_traffic_job, throughput_traffic = split_benchmark_config(
        compose_config("jobs/benchmark/throughput_traffic", [RESOURCE_OVERRIDE]),
        ScalingBenchmarkConfig,
    )
    rollout_job, rollout = split_benchmark_config(
        compose_config("jobs/benchmark/rollout", [RESOURCE_OVERRIDE]),
        RolloutBenchmarkConfig,
    )

    assert isinstance(environment, EnvironmentBenchmarkJobConfig)
    assert isinstance(throughput, ScalingBenchmarkConfig)
    assert isinstance(parse_evaluation_config(throughput_job), EvaluationJobConfig)
    assert isinstance(throughput_traffic, ScalingBenchmarkConfig)
    assert parse_evaluation_config(throughput_traffic_job).evaluation.mode == "traffic"
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

    evaluation = parse_evaluation_config(compose_config("jobs/evaluation/no_traffic_smoke", None))
    parallel_evaluation = parse_evaluation_config(
        compose_config("jobs/evaluation/traffic_matrix", None)
    )
    training = parse_training_config(
        compose_config(
            "jobs/training/ppo",
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
