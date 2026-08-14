from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from eco_planner.rl import parse_ppo_optimization_config


def _config() -> object:
    path = Path(__file__).parents[3] / "configs" / "train" / "ppo_smoke.yaml"
    return OmegaConf.load(path)


def test_ppo_smoke_config_is_strict_and_complete() -> None:
    parsed = parse_ppo_optimization_config(_config())  # type: ignore[arg-type]
    assert parsed.gamma == pytest.approx(0.99)
    assert parsed.gae_lambda == pytest.approx(0.95)
    assert parsed.optimizer_steps_per_update == 4
    assert parsed.scheduler_total_optimizer_steps == 4
    assert parsed.value_loss == "l2"
    assert not parsed.clip_value


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("minibatch_size", 3, "divisible"),
        ("scheduler_total_optimizer_steps", 3, "horizon"),
        ("clip_value", True, "clip_value=false"),
        ("normalize_advantage", False, "normalize_advantage=true"),
        ("adam_epsilon", 0.0, "positive"),
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
        parse_ppo_optimization_config(config)  # type: ignore[arg-type]


def test_ppo_config_rejects_extra_keys() -> None:
    config = _config()
    config["unknown"] = 1  # type: ignore[index]
    with pytest.raises(ValueError, match="unexpected"):
        parse_ppo_optimization_config(config)  # type: ignore[arg-type]
