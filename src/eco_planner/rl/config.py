"""Strict Hydra-facing configuration for the exploration policy."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from omegaconf import DictConfig, OmegaConf


@dataclass(frozen=True)
class ExplorationPolicyConfig:
    """All reproduction-chosen architecture and Beta initialization parameters."""

    name: Literal["exploration_beta"]
    hidden_dim: int
    reference_horizon: int
    reference_state_dim: int
    reference_mixer_depth: int
    reference_token_mlp_hidden_dim: int
    reference_channel_mlp_hidden_dim: int
    cross_attention_heads: int
    cross_attention_dropout: float
    fusion_mlp_depth: int
    fusion_hidden_dim: int
    initial_concentration: float
    minimum_concentration: float


def parse_exploration_policy_config(config: DictConfig) -> ExplorationPolicyConfig:
    """Resolve one policy profile without defaults or ignored fields."""

    if not isinstance(config, DictConfig):
        raise TypeError("policy configuration must be a DictConfig")
    raw = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    if not isinstance(raw, dict):
        raise TypeError("policy configuration must resolve to a dictionary")
    expected = {
        "name",
        "hidden_dim",
        "reference_horizon",
        "reference_state_dim",
        "reference_mixer_depth",
        "reference_token_mlp_hidden_dim",
        "reference_channel_mlp_hidden_dim",
        "cross_attention_heads",
        "cross_attention_dropout",
        "fusion_mlp_depth",
        "fusion_hidden_dim",
        "initial_concentration",
        "minimum_concentration",
    }
    keys = set(raw)
    if keys != expected:
        raise ValueError(
            "exploration policy keys mismatch; "
            f"missing={sorted(expected - keys)}, "
            f"unexpected={sorted(str(key) for key in keys - expected)}"
        )
    if raw["name"] != "exploration_beta":
        raise ValueError("policy.name must equal 'exploration_beta'")

    integer_fields = expected - {
        "name",
        "cross_attention_dropout",
        "initial_concentration",
        "minimum_concentration",
    }
    values: dict[str, int] = {}
    for name in integer_fields:
        value = raw[name]
        if type(value) is not int:
            raise TypeError(f"policy.{name} must be an integer")
        if value <= 0:
            raise ValueError(f"policy.{name} must be positive")
        values[name] = value
    if values["reference_horizon"] != 80 or values["reference_state_dim"] != 4:
        raise ValueError("policy reference trajectory contract must be [B, 80, 4]")
    if values["hidden_dim"] % values["cross_attention_heads"] != 0:
        raise ValueError("policy.hidden_dim must be divisible by cross_attention_heads")

    dropout = _finite_float(raw["cross_attention_dropout"], "cross_attention_dropout")
    if not 0.0 <= dropout < 1.0:
        raise ValueError("policy.cross_attention_dropout must be in [0, 1)")
    minimum = _positive_float(raw["minimum_concentration"], "minimum_concentration")
    initial = _positive_float(raw["initial_concentration"], "initial_concentration")
    if initial <= minimum:
        raise ValueError("policy.initial_concentration must exceed minimum_concentration")

    return ExplorationPolicyConfig(
        name="exploration_beta",
        hidden_dim=values["hidden_dim"],
        reference_horizon=values["reference_horizon"],
        reference_state_dim=values["reference_state_dim"],
        reference_mixer_depth=values["reference_mixer_depth"],
        reference_token_mlp_hidden_dim=values["reference_token_mlp_hidden_dim"],
        reference_channel_mlp_hidden_dim=values["reference_channel_mlp_hidden_dim"],
        cross_attention_heads=values["cross_attention_heads"],
        cross_attention_dropout=dropout,
        fusion_mlp_depth=values["fusion_mlp_depth"],
        fusion_hidden_dim=values["fusion_hidden_dim"],
        initial_concentration=initial,
        minimum_concentration=minimum,
    )


def _finite_float(value: object, name: str) -> float:
    if type(value) not in {int, float}:
        raise TypeError(f"policy.{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"policy.{name} must be finite")
    return result


def _positive_float(value: object, name: str) -> float:
    result = _finite_float(value, name)
    if result <= 0.0:
        raise ValueError(f"policy.{name} must be positive")
    return result
