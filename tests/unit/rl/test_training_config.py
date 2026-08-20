from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from eco_planner.rl import parse_training_config


def _config():
    config_dir = Path(__file__).parents[3] / "configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        return compose(
            config_name="experiment/train_ppo_smoke",
            overrides=["runtime.seed=0", "training.replay_id=0"],
        )


def test_ppo_smoke_config_resolves_a_general_closed_loop_job() -> None:
    parsed = parse_training_config(_config())
    assert [item.map for item in parsed.scenarios] == ["S", "SC"]
    assert parsed.training.update_count == 4
    assert parsed.rl.batch_size == 32
    assert parsed.rl.scheduler_total_optimizer_steps == 32


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        ("training.transitions_per_environment", 0, "greater than 0"),
        ("env.trajectory_execution_steps", 5, "trajectory_execution_steps"),
        ("rl.batch_size", 16, "must equal all closed-loop"),
        ("rl.scheduler_total_optimizer_steps", 8, "cover every configured"),
    ],
)
def test_training_config_rejects_invalid_cross_field_values(
    path: str, value: object, message: str
) -> None:
    config = _config()
    OmegaConf.update(config, path, value, merge=False)
    with pytest.raises((TypeError, ValueError), match=message):
        parse_training_config(config)
