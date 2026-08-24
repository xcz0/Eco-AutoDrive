from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from eco_planner.rl.config import parse_ppo_config


def _config() -> object:
    config_dir = Path(__file__).parents[3] / "configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        config = compose(
            config_name="jobs/training/ppo_smoke",
            overrides=["runtime.seed=0", "training.replay_id=0"],
        )
    return OmegaConf.create(OmegaConf.to_container(config.rl, resolve=True))


def test_ppo_smoke_config_is_strict_and_complete() -> None:
    parsed = parse_ppo_config(_config())  # type: ignore[arg-type]
    assert parsed.gamma == pytest.approx(0.99)
    assert parsed.gae_lambda == pytest.approx(0.95)
    assert parsed.optimizer_steps_per_update == 8
    assert parsed.scheduler_total_optimizer_steps == 32


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("minibatch_size", 3, "divisible"),
        ("scheduler_total_optimizer_steps", 3, "horizon"),
        ("normalize_advantage", True, "Extra inputs"),
        ("value_loss", "l2", "Extra inputs"),
        ("clip_value", False, "Extra inputs"),
        ("optimizer", "adam", "Extra inputs"),
        ("scheduler", "cosine", "Extra inputs"),
        ("adam_epsilon", 0.0, "greater than 0"),
    ],
)
def test_ppo_config_rejects_unsupported_or_inconsistent_values(
    field: str,
    value: object,
    message: str,
) -> None:
    config = _config()
    config[field] = value  # type: ignore[index]
    with pytest.raises((TypeError, ValueError), match=message):
        parse_ppo_config(config)  # type: ignore[arg-type]


def test_ppo_config_rejects_extra_keys() -> None:
    config = _config()
    config["unknown"] = 1  # type: ignore[index]
    with pytest.raises(ValueError, match="Extra inputs"):
        parse_ppo_config(config)  # type: ignore[arg-type]
