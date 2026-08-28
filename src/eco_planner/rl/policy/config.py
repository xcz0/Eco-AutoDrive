"""Strict Exploration Policy architecture configuration."""

from __future__ import annotations

from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, model_validator


class ExplorationPolicyConfig(BaseModel):
    """Architecture and affine-Beta initialization parameters."""

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid", allow_inf_nan=False)

    hidden_dim: StrictInt = Field(gt=0)
    reference_mixer_depth: StrictInt = Field(gt=0)
    reference_token_mlp_hidden_dim: StrictInt = Field(gt=0)
    reference_channel_mlp_hidden_dim: StrictInt = Field(gt=0)
    cross_attention_heads: StrictInt = Field(gt=0)
    cross_attention_dropout: StrictFloat = Field(ge=0.0, lt=1.0)
    fusion_mlp_depth: StrictInt = Field(gt=0)
    fusion_hidden_dim: StrictInt = Field(gt=0)
    initial_concentration: StrictFloat = Field(gt=0.0)
    minimum_concentration: StrictFloat = Field(gt=0.0)

    @model_validator(mode="after")
    def validate_policy_contract(self) -> ExplorationPolicyConfig:
        if self.hidden_dim % self.cross_attention_heads:
            raise ValueError("policy.hidden_dim must be divisible by cross_attention_heads")
        if self.initial_concentration <= self.minimum_concentration:
            raise ValueError("policy.initial_concentration must exceed minimum_concentration")
        return self


def parse_exploration_policy_config(config: DictConfig) -> ExplorationPolicyConfig:
    """Parse one resolved Hydra policy component at the configuration boundary."""

    return ExplorationPolicyConfig.model_validate(
        dict(OmegaConf.to_container(config, resolve=True, throw_on_missing=True))
    )
