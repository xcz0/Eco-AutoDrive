from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from reward_sanity import evaluate_sanity, load_sanity_config

from eco_planner.evaluation.config import EvaluationJobConfig, parse_evaluation_config
from eco_planner.rl.config import RLTrainingJobConfig, parse_training_config

ROOT = Path(__file__).resolve().parents[1]


def test_canonical_jobs_compose_into_typed_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MACHINE_NAME", "rtx3050_laptop")
    config_dir = Path(__file__).resolve().parents[1] / "configs"

    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        evaluation = compose(
            config_name="jobs/evaluation/no_traffic_smoke",
            overrides=["runtime.seed=0", "resources=rtx3050_laptop"],
        )
        training = compose(
            config_name="jobs/training/ppo_smoke",
            overrides=[
                "runtime.seed=0",
                "training.replay_id=0",
                "resources=rtx3050_laptop",
            ],
        )

    assert isinstance(parse_evaluation_config(evaluation), EvaluationJobConfig)
    assert isinstance(parse_training_config(training), RLTrainingJobConfig)


def test_reward_sanity_config_covers_anti_hacking_and_gate_cases() -> None:
    config = load_sanity_config(ROOT / "configs" / "matrices" / "reward_sanity.yaml")

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
    config = load_sanity_config(ROOT / "configs" / "matrices" / "reward_sanity.yaml")

    report = evaluate_sanity(config)

    assert report["status"] == "passed"
    assert report["case_count"] == 12
    assert all(item["passed"] for item in report["checks"])
