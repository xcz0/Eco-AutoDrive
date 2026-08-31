"""Strict PPO optimization configuration."""

from __future__ import annotations

from omegaconf import DictConfig
from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, model_validator

from eco_planner.configuration import resolve_config_mapping


class PPOConfig(BaseModel):
    """GAE, PPO, optimizer, and scheduler parameters for one update profile."""

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid", allow_inf_nan=False)

    name: str = Field(min_length=1)
    gamma: StrictFloat = Field(gt=0.0, lt=1.0)
    gae_lambda: StrictFloat = Field(gt=0.0, lt=1.0)
    clip_epsilon: StrictFloat = Field(gt=0.0, lt=1.0)
    value_coefficient: StrictFloat = Field(ge=0.0)
    entropy_coefficient: StrictFloat = Field(ge=0.0)
    learning_rate: StrictFloat = Field(gt=0.0)
    adam_epsilon: StrictFloat = Field(gt=0.0)
    weight_decay: StrictFloat = Field(ge=0.0)
    max_gradient_norm: StrictFloat = Field(gt=0.0)
    epochs: StrictInt = Field(gt=0)
    batch_size: StrictInt = Field(gt=0)
    minibatch_size: StrictInt = Field(gt=0)
    minibatch_seed: StrictInt = Field(ge=0)
    scheduler_total_optimizer_steps: StrictInt = Field(gt=0)
    scheduler_minimum_learning_rate: StrictFloat = Field(ge=0.0)

    @property
    def optimizer_steps_per_update(self) -> int:
        return self.epochs * (self.batch_size // self.minibatch_size)

    @model_validator(mode="after")
    def validate_optimization_contract(self) -> PPOConfig:
        if self.batch_size % self.minibatch_size:
            raise ValueError("ppo.batch_size must be divisible by ppo.minibatch_size")
        if self.scheduler_minimum_learning_rate >= self.learning_rate:
            raise ValueError("scheduler minimum learning rate must be below the initial rate")
        if self.scheduler_total_optimizer_steps < self.optimizer_steps_per_update:
            raise ValueError("scheduler horizon must cover at least one complete PPO update")
        return self


def parse_ppo_config(config: DictConfig) -> PPOConfig:
    """Parse one resolved Hydra PPO component at the configuration boundary."""

    return PPOConfig.model_validate(resolve_config_mapping(config))
