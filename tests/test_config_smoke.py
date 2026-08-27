from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from eco_planner.evaluation.config import EvaluationJobConfig, parse_evaluation_config
from eco_planner.rl.config import RLTrainingJobConfig, parse_training_config


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
