from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from omegaconf import DictConfig

from eco_planner.evaluation import EvaluationJobConfig, parse_evaluation_config
from eco_planner.experiments.guidance.sweep import load_energy_study
from eco_planner.reward_validation import evaluate_sanity, load_sanity_config

ComposeConfig = Callable[[str, list[str] | None], DictConfig]


def test_experiment_manifests_are_strict_and_reference_composable_jobs(
    monkeypatch: pytest.MonkeyPatch,
    compose_config: ComposeConfig,
    config_root: Path,
) -> None:
    monkeypatch.setenv("MACHINE_NAME", "rtx3050_laptop")
    energy = load_energy_study(
        config_root / "experiments" / "guidance" / "energy-sweep" / "matrix.yaml"
    )

    for job in energy.jobs:
        for guidance in energy.guidance_profiles:
            config = compose_config(
                job.config_name,
                [f"components/guidance={guidance.config}"],
            )
            parsed = parse_evaluation_config(config)
            assert isinstance(parsed, EvaluationJobConfig)
            assert parsed.runtime.seed == 0
            assert parsed.sampler.name == "ddim5"
            assert parsed.env["random_agent_model"] is False


def test_reward_sanity_config_covers_anti_hacking_and_gate_cases(config_root: Path) -> None:
    config = load_sanity_config(config_root / "validation" / "reward.yaml")

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
    config = load_sanity_config(config_root / "validation" / "reward.yaml")

    report = evaluate_sanity(config)

    assert report["status"] == "passed"
    assert report["case_count"] == 12
    assert all(item["passed"] for item in report["checks"])


def test_no_energy_reward_sanity_report_passes_and_pins_the_r0_cruise_total(
    config_root: Path,
) -> None:
    config = load_sanity_config(config_root / "validation" / "reward" / "sanity_no_energy.yaml")

    report = evaluate_sanity(config)

    assert "plannerrft_no_energy_v1.yaml" in config.reward_config
    assert report["reward_profile"] == "plannerrft_no_energy_v1"
    assert report["status"] == "passed"
    assert report["case_count"] == 12
    assert all(item["passed"] for item in report["checks"])
