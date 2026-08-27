from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from eco_planner.evaluation.config import EvaluationJobConfig, parse_evaluation_config
from eco_planner.rl.config import (
    RLTrainingJobConfig,
    RolloutJobConfig,
    parse_rollout_config,
    parse_training_config,
)
from scripts.benchmarking.common import (
    EnvironmentBenchmarkJobConfig,
    RolloutBenchmarkConfig,
    ScalingBenchmarkConfig,
    parse_environment_job,
    split_benchmark_config,
)
from scripts.studies.energy_matrix import load_energy_study
from scripts.studies.ppo_reward_ab import load_ab_config
from scripts.studies.reward_sanity import evaluate_sanity, load_sanity_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs"


def _compose(config_name: str, overrides: list[str] | None = None):
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_ROOT)):
        return compose(config_name=config_name, overrides=overrides or [])


@pytest.mark.parametrize(
    "config_name",
    [
        "jobs/evaluation/no_traffic/smoke",
        "jobs/evaluation/traffic/smoke",
        "jobs/evaluation/traffic/matrix",
    ],
)
def test_evaluation_jobs_compose_into_typed_boundary(
    monkeypatch: pytest.MonkeyPatch, config_name: str
) -> None:
    monkeypatch.setenv("MACHINE_NAME", "rtx3050_laptop")

    parsed = parse_evaluation_config(_compose(config_name))

    assert isinstance(parsed, EvaluationJobConfig)


def test_training_jobs_compose_into_typed_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MACHINE_NAME", "rtx3050_laptop")

    training = _compose(
        "jobs/training/ppo/smoke",
        overrides=["runtime.seed=0", "training.replay_id=0"],
    )
    rollout = _compose("jobs/training/rollout/smoke")

    assert isinstance(parse_training_config(training), RLTrainingJobConfig)
    assert isinstance(parse_rollout_config(rollout), RolloutJobConfig)


def test_benchmark_jobs_compose_into_typed_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MACHINE_NAME", "rtx3050_laptop")

    environment = parse_environment_job(_compose("jobs/benchmark/environment"))
    throughput_job, throughput = split_benchmark_config(
        _compose("jobs/benchmark/throughput"), ScalingBenchmarkConfig
    )
    rollout_job, rollout = split_benchmark_config(
        _compose("jobs/benchmark/rollout"), RolloutBenchmarkConfig
    )

    assert isinstance(environment, EnvironmentBenchmarkJobConfig)
    assert isinstance(throughput, ScalingBenchmarkConfig)
    assert isinstance(parse_evaluation_config(throughput_job), EvaluationJobConfig)
    assert isinstance(rollout, RolloutBenchmarkConfig)
    assert isinstance(parse_training_config(rollout_job), RLTrainingJobConfig)


def test_study_manifests_are_strict_and_reference_composable_jobs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MACHINE_NAME", "rtx3050_laptop")
    energy = load_energy_study(CONFIG_ROOT / "studies" / "energy" / "matrix.yaml")
    reward_ab = load_ab_config(CONFIG_ROOT / "studies" / "reward" / "ppo_ab.yaml")

    for job in energy.jobs:
        for guidance in energy.guidance_profiles:
            config = _compose(
                job.config_name,
                overrides=[f"components/guidance={guidance.config}"],
            )
            assert isinstance(parse_evaluation_config(config), EvaluationJobConfig)
    training = _compose(
        reward_ab.base_training_config,
        overrides=["runtime.seed=0", "training.replay_id=0"],
    )
    assert isinstance(parse_training_config(training), RLTrainingJobConfig)


def test_reward_sanity_config_covers_anti_hacking_and_gate_cases() -> None:
    config = load_sanity_config(CONFIG_ROOT / "studies" / "reward" / "sanity.yaml")

    assert {item.name for item in config.cases} == {
        "cruise",
        "stationary",
        "extremely_low_speed",
        "slower_progress",
        "low_route_progress",
        "overspeed",
        "following_non_closing",
        "approaching_collision",
        "uncomfortable",
        "collision",
        "out_of_road",
        "wrong_direction",
    }
    assert len(config.comparisons) == 7


def test_reward_sanity_report_requires_every_declared_check_to_pass() -> None:
    config = load_sanity_config(CONFIG_ROOT / "studies" / "reward" / "sanity.yaml")

    report = evaluate_sanity(config)

    assert report["status"] == "passed"
    assert report["case_count"] == 12
    assert all(item["passed"] for item in report["checks"])
