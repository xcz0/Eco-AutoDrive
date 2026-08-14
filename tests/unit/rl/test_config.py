from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from eco_planner.rl import parse_exploration_policy_config


def _policy_mapping() -> dict[str, object]:
    return {
        "name": "exploration_beta",
        "hidden_dim": 192,
        "reference_horizon": 80,
        "reference_state_dim": 4,
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
        config = compose(config_name="policy/exploration_beta")

    parsed = parse_exploration_policy_config(config.policy)
    assert parsed.hidden_dim == 192
    assert parsed.reference_horizon == 80
    assert parsed.initial_concentration == 2.0


@pytest.mark.parametrize(
    ("update", "error"),
    [
        ({"hidden_dim": 191}, "divisible"),
        ({"reference_horizon": 79}, r"\[B, 80, 4\]"),
        ({"cross_attention_dropout": 1.0}, r"\[0, 1\)"),
        ({"initial_concentration": 0.0001}, "exceed"),
        ({"fusion_mlp_depth": 0}, "positive"),
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
    with pytest.raises(ValueError, match="keys mismatch"):
        parse_exploration_policy_config(OmegaConf.create(raw))
