from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from eco_planner.rl import parse_stage6_training_config


def _config():
    config_dir = Path(__file__).parents[3] / "configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        return compose(
            config_name="training/stage6_smoke",
            overrides=["runtime.seed=0", "training.replay_id=0"],
        )


def test_stage6_smoke_config_locks_closed_loop_batch_and_scheduler() -> None:
    parsed = parse_stage6_training_config(_config())
    assert parsed.runtime.accelerator == "cuda"
    assert parsed.runtime.precision == "bf16-mixed"
    assert [item.map for item in parsed.scenarios] == ["S", "SC"]
    assert parsed.training.update_count == 4
    assert parsed.train.batch_size == 32
    assert parsed.train.optimizer_steps_per_update == 8
    assert parsed.train.scheduler_total_optimizer_steps == 32


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        ("runtime.precision", "32-true", "CUDA BF16"),
        ("training.transitions_per_environment", 15, "literal_error"),
        ("reward.speed_reward", 0.2, "literal_error"),
        ("env.trajectory_execution_steps", 5, "trajectory_execution_steps"),
        ("train.batch_size", 16, "32 closed-loop"),
    ],
)
def test_stage6_smoke_config_rejects_contract_changes(
    path: str, value: object, message: str
) -> None:
    config = _config()
    OmegaConf.update(config, path, value, merge=False)
    with pytest.raises((TypeError, ValueError), match=message):
        parse_stage6_training_config(config)
