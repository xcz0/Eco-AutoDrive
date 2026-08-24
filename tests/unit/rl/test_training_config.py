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
            config_name="jobs/training/ppo_smoke",
            overrides=["runtime.seed=0", "training.replay_id=0"],
        )


def test_ppo_smoke_config_resolves_a_general_closed_loop_job() -> None:
    parsed = parse_training_config(_config())
    assert [item.map for item in parsed.scenarios] == ["S", "SC"]
    assert parsed.training.update_count == 4
    assert parsed.rl.batch_size == 32
    assert parsed.rl.scheduler_total_optimizer_steps == 32
    assert parsed.resources.name == "rtx3050_laptop"


def test_training_resource_profile_changes_physical_workers_without_changing_ppo_batch() -> None:
    baseline = parse_training_config(_config())
    config = _config()
    OmegaConf.update(config, "resources.rollout_worker_count", 1, merge=False)
    profiled = parse_training_config(config)

    assert profiled.resources.rollout_worker_count == 1
    assert profiled.rl == baseline.rl
    assert profiled.training == baseline.training


def test_training_reward_coefficients_remain_sweepable_and_match_metadrive_env() -> None:
    config = _config()
    OmegaConf.update(config, "reward.speed_reward", 0.2, merge=False)

    parsed = parse_training_config(config)

    assert parsed.reward.speed_reward == pytest.approx(0.2)
    assert parsed.env["speed_reward"] == pytest.approx(0.2)


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
