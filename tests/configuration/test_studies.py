from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from omegaconf import DictConfig

from eco_planner.evaluation.config import EvaluationJobConfig, parse_evaluation_config
from eco_planner.rl.config import TrainingJobConfig, parse_training_config
from scripts.analysis.ppo_reward_ab import aggregate_pair_reports
from scripts.studies.energy_matrix import load_energy_study
from scripts.studies.ppo_reward_ab import build_training_command, load_ab_config
from scripts.studies.reward_sanity import evaluate_sanity, load_sanity_config

ComposeConfig = Callable[[str, list[str] | None], DictConfig]


def test_study_manifests_are_strict_and_reference_composable_jobs(
    monkeypatch: pytest.MonkeyPatch,
    compose_config: ComposeConfig,
    config_root: Path,
) -> None:
    monkeypatch.setenv("MACHINE_NAME", "rtx3050_laptop")
    energy = load_energy_study(config_root / "studies" / "energy" / "matrix.yaml")
    reward_studies = [
        load_ab_config(config_root / "studies" / "reward" / name)
        for name in ("ppo_ab.yaml", "ppo_ab_long_term.yaml")
    ]

    for job in energy.jobs:
        for guidance in energy.guidance_profiles:
            config = compose_config(
                job.config_name,
                [f"components/guidance={guidance.config}"],
            )
            assert isinstance(parse_evaluation_config(config), EvaluationJobConfig)
    for reward_ab in reward_studies:
        profile = reward_ab.profiles[0]
        matched = reward_ab.matched_training
        training = compose_config(
            reward_ab.base_training_config,
            [
                f"components/reward={profile.reward_config}",
                "runtime.seed=0",
                "training.replay_id=0",
                f"training.update_count={matched.update_count}",
                f"training.transitions_per_environment={matched.transitions_per_environment}",
                f"ppo.scheduler_total_optimizer_steps={matched.scheduler_total_optimizer_steps}",
            ],
        )
        parsed = parse_training_config(training)
        assert isinstance(parsed, TrainingJobConfig)
        assert parsed.training.update_count == matched.update_count
        assert parsed.ppo.scheduler_total_optimizer_steps == matched.scheduler_total_optimizer_steps


def test_long_term_reward_ab_builds_explicit_duration_and_scheduler_overrides(
    config_root: Path,
) -> None:
    config = load_ab_config(config_root / "studies" / "reward" / "ppo_ab_long_term.yaml")
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


def test_reward_sanity_config_covers_anti_hacking_and_gate_cases(config_root: Path) -> None:
    config = load_sanity_config(config_root / "studies" / "reward" / "sanity.yaml")

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


def test_reward_sanity_report_requires_every_declared_check_to_pass(config_root: Path) -> None:
    config = load_sanity_config(config_root / "studies" / "reward" / "sanity.yaml")

    report = evaluate_sanity(config)

    assert report["status"] == "passed"
    assert report["case_count"] == 12
    assert all(item["passed"] for item in report["checks"])
