from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from eco_planner.rl.config import parse_exploration_policy_config, parse_rollout_config


def _policy_mapping() -> dict[str, object]:
    return {
        "hidden_dim": 192,
        "reference_mixer_depth": 2,
        "reference_token_mlp_hidden_dim": 128,
        "reference_channel_mlp_hidden_dim": 384,
        "cross_attention_heads": 6,
        "cross_attention_dropout": 0.0,
        "fusion_mlp_depth": 2,
        "fusion_hidden_dim": 256,
        "initial_concentration": 2.0,
        "minimum_concentration": 0.0001,
    }


def test_hydra_policy_profile_is_complete_and_strict() -> None:
    config_dir = Path(__file__).resolve().parents[3] / "configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        config = compose(config_name="rl/exploration_policy")

    parsed = parse_exploration_policy_config(config.policy)
    assert parsed.hidden_dim == 192
    assert parsed.initial_concentration == 2.0


@pytest.mark.parametrize(
    ("update", "error"),
    [
        ({"hidden_dim": 191}, "divisible"),
        ({"reference_horizon": 79}, "Extra inputs"),
        ({"cross_attention_dropout": 1.0}, "less than 1"),
        ({"initial_concentration": 0.0001}, "exceed"),
        ({"fusion_mlp_depth": 0}, "greater than 0"),
    ],
)
def test_policy_config_rejects_invalid_values(update: dict[str, object], error: str) -> None:
    raw = _policy_mapping()
    raw.update(update)
    with pytest.raises(ValueError, match=error):
        parse_exploration_policy_config(OmegaConf.create(raw))


def test_policy_config_rejects_missing_and_extra_fields() -> None:
    raw = _policy_mapping()
    del raw["fusion_hidden_dim"]
    raw["undocumented_default"] = 1
    with pytest.raises(ValueError, match="Field required|Extra inputs"):
        parse_exploration_policy_config(OmegaConf.create(raw))


def test_rollout_profile_excludes_fixed_runtime_semantics() -> None:
    config_dir = Path(__file__).resolve().parents[3] / "configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        config = compose(config_name="experiment/rollout_smoke")

    parsed = parse_rollout_config(config)
    assert parsed.env["trajectory_execution_steps"] == 1
    assert set(config.rollout) == {
        "mode",
        "max_transitions",
        "history_warmup_steps",
        "policy_action_seed",
        "stopped_speed_threshold_mps",
    }
