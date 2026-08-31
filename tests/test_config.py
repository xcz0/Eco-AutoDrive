from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from eco_planner.evaluation.config import EvaluationJobConfig, parse_evaluation_config
from eco_planner.rl.config import (
    RolloutJobConfig,
    TrainingJobConfig,
    parse_rollout_config,
    parse_training_config,
)
from scripts.analysis.ppo_reward_ab import aggregate_pair_reports
from scripts.benchmarking.common import (
    EnvironmentBenchmarkJobConfig,
    RolloutBenchmarkConfig,
    ScalingBenchmarkConfig,
    parse_environment_job,
    split_benchmark_config,
)
from scripts.studies.energy_matrix import load_energy_study
from scripts.studies.ppo_reward_ab import build_training_command, load_ab_config
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

    parsed_training = parse_training_config(training)
    assert isinstance(parsed_training, TrainingJobConfig)
    assert parsed_training.training.planner_compile_mode == "eager"
    compiled_training = parse_training_config(
        _compose(
            "jobs/training/ppo/smoke",
            overrides=[
                "runtime.seed=0",
                "training.replay_id=0",
                "training.planner_compile_mode=dit_reduce_overhead",
            ],
        )
    )
    assert compiled_training.training.planner_compile_mode == "dit_reduce_overhead"
    with pytest.raises(ValueError, match="planner_compile_mode"):
        parse_training_config(
            _compose(
                "jobs/training/ppo/smoke",
                overrides=[
                    "runtime.seed=0",
                    "training.replay_id=0",
                    "training.planner_compile_mode=automatic",
                ],
            )
        )
    assert isinstance(parse_rollout_config(rollout), RolloutJobConfig)


def test_conservative_training_job_composes_into_typed_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MACHINE_NAME", "rtx3050_laptop")

    training = _compose(
        "jobs/training/ppo/conservative",
        overrides=["runtime.seed=0", "training.replay_id=0"],
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
    assert rollout.ppo_epochs == 4
    assert rollout.ppo_minibatch_size == 16
    assert rollout.transitions_per_slot == 16
    assert isinstance(parse_training_config(rollout_job), TrainingJobConfig)


def test_study_manifests_are_strict_and_reference_composable_jobs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MACHINE_NAME", "rtx3050_laptop")
    energy = load_energy_study(CONFIG_ROOT / "studies" / "energy" / "matrix.yaml")
    reward_studies = [
        load_ab_config(CONFIG_ROOT / "studies" / "reward" / name)
        for name in ("ppo_ab.yaml", "ppo_ab_long_term.yaml")
    ]

    for job in energy.jobs:
        for guidance in energy.guidance_profiles:
            config = _compose(
                job.config_name,
                overrides=[f"components/guidance={guidance.config}"],
            )
            assert isinstance(parse_evaluation_config(config), EvaluationJobConfig)
    for reward_ab in reward_studies:
        profile = reward_ab.profiles[0]
        matched = reward_ab.matched_training
        training = _compose(
            reward_ab.base_training_config,
            overrides=[
                f"components/reward={profile.reward_config}",
                "runtime.seed=0",
                "training.replay_id=0",
                f"training.update_count={matched.update_count}",
                (f"training.transitions_per_environment={matched.transitions_per_environment}"),
                (f"ppo.scheduler_total_optimizer_steps={matched.scheduler_total_optimizer_steps}"),
            ],
        )
        parsed = parse_training_config(training)
        assert isinstance(parsed, TrainingJobConfig)
        assert parsed.training.update_count == matched.update_count
        assert parsed.ppo.scheduler_total_optimizer_steps == matched.scheduler_total_optimizer_steps


def test_long_term_reward_ab_builds_explicit_duration_and_scheduler_overrides() -> None:
    config = load_ab_config(CONFIG_ROOT / "studies" / "reward" / "ppo_ab_long_term.yaml")
    command = build_training_command(
        config,
        config.profiles[1],
        training_seed=2,
        replay_id=0,
        run_dir=Path("phase-b") / "energy",
    )

    assert config.matched_training.update_count == 20
    assert config.matched_training.training_seeds == [0, 1, 2]
    assert "training.update_count=20" in command
    assert "ppo.scheduler_total_optimizer_steps=160" in command


def test_long_term_reward_ab_aggregates_one_effect_estimate_per_seed() -> None:
    pairs = [
        {
            "training_seed": seed,
            "replay_id": 0,
            "changes": {"energy_intensity_change_fraction": value},
            "review_flags": {"progress_regressed": seed == 2},
        }
        for seed, value in enumerate((-0.2, -0.1, 0.0))
    ]

    aggregate = aggregate_pair_reports(pairs)
    statistics = aggregate["change_statistics"]

    assert aggregate["pair_count"] == 3
    assert aggregate["training_seed_count"] == 3
    assert isinstance(statistics, dict)
    assert statistics["energy_intensity_change_fraction"] == {
        "mean": pytest.approx(-0.1),
        "sample_standard_deviation": pytest.approx(0.1),
        "minimum": pytest.approx(-0.2),
        "maximum": pytest.approx(0.0),
    }
    assert aggregate["review_flag_counts"] == {"progress_regressed": 1}


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
