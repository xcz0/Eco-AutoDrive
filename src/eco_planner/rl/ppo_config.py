"""Strict Hydra-facing configuration for Stage-5 PPO mathematics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from omegaconf import DictConfig, OmegaConf


@dataclass(frozen=True)
class PPOOptimizationConfig:
    """All explicitly selected GAE, PPO, optimizer, and scheduler parameters."""

    name: Literal["ppo_stage5_smoke", "ppo_stage6_smoke"]
    gamma: float
    gae_lambda: float
    normalize_advantage: bool
    clip_epsilon: float
    value_loss: Literal["l2"]
    clip_value: Literal[False]
    value_coefficient: float
    entropy_coefficient: float
    optimizer: Literal["adam"]
    learning_rate: float
    adam_epsilon: float
    weight_decay: float
    max_gradient_norm: float
    epochs: int
    batch_size: int
    minibatch_size: int
    minibatch_seed: int
    scheduler: Literal["cosine"]
    scheduler_total_optimizer_steps: int
    scheduler_minimum_learning_rate: float

    @property
    def optimizer_steps_per_update(self) -> int:
        return self.epochs * (self.batch_size // self.minibatch_size)


def parse_ppo_optimization_config(config: DictConfig) -> PPOOptimizationConfig:
    """Resolve one Stage-5 profile without defaults or ignored fields."""

    if not isinstance(config, DictConfig):
        raise TypeError("PPO optimization configuration must be a DictConfig")
    raw = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    if not isinstance(raw, dict):
        raise TypeError("PPO optimization configuration must resolve to a dictionary")
    expected = {
        "name",
        "gamma",
        "gae_lambda",
        "normalize_advantage",
        "clip_epsilon",
        "value_loss",
        "clip_value",
        "value_coefficient",
        "entropy_coefficient",
        "optimizer",
        "learning_rate",
        "adam_epsilon",
        "weight_decay",
        "max_gradient_norm",
        "epochs",
        "batch_size",
        "minibatch_size",
        "minibatch_seed",
        "scheduler",
        "scheduler_total_optimizer_steps",
        "scheduler_minimum_learning_rate",
    }
    keys = set(raw)
    if keys != expected:
        raise ValueError(
            "PPO optimization keys mismatch; "
            f"missing={sorted(expected - keys)}, "
            f"unexpected={sorted(str(key) for key in keys - expected)}"
        )
    if raw["name"] not in {"ppo_stage5_smoke", "ppo_stage6_smoke"}:
        raise ValueError("train.name must select a supported PPO smoke profile")
    for name, expected_value in (
        ("value_loss", "l2"),
        ("optimizer", "adam"),
        ("scheduler", "cosine"),
    ):
        if raw[name] != expected_value:
            raise ValueError(f"train.{name} must equal {expected_value!r}")
    if type(raw["normalize_advantage"]) is not bool or not raw["normalize_advantage"]:
        raise ValueError("Stage-5 PPO requires normalize_advantage=true")
    if type(raw["clip_value"]) is not bool or raw["clip_value"]:
        raise ValueError("Stage-5 PPO requires clip_value=false")

    integer_fields = {
        "epochs",
        "batch_size",
        "minibatch_size",
        "minibatch_seed",
        "scheduler_total_optimizer_steps",
    }
    integers: dict[str, int] = {}
    for name in integer_fields:
        value = raw[name]
        if type(value) is not int:
            raise TypeError(f"train.{name} must be an integer")
        minimum = 0 if name == "minibatch_seed" else 1
        if value < minimum:
            raise ValueError(f"train.{name} must be at least {minimum}")
        integers[name] = value
    if integers["batch_size"] % integers["minibatch_size"]:
        raise ValueError("train.batch_size must be divisible by train.minibatch_size")

    gamma = _open_unit_float(raw["gamma"], "gamma")
    gae_lambda = _open_unit_float(raw["gae_lambda"], "gae_lambda")
    clip_epsilon = _open_unit_float(raw["clip_epsilon"], "clip_epsilon")
    value_coefficient = _non_negative_float(raw["value_coefficient"], "value_coefficient")
    entropy_coefficient = _non_negative_float(raw["entropy_coefficient"], "entropy_coefficient")
    learning_rate = _positive_float(raw["learning_rate"], "learning_rate")
    adam_epsilon = _positive_float(raw["adam_epsilon"], "adam_epsilon")
    weight_decay = _non_negative_float(raw["weight_decay"], "weight_decay")
    max_gradient_norm = _positive_float(raw["max_gradient_norm"], "max_gradient_norm")
    minimum_learning_rate = _non_negative_float(
        raw["scheduler_minimum_learning_rate"], "scheduler_minimum_learning_rate"
    )
    if minimum_learning_rate >= learning_rate:
        raise ValueError("scheduler minimum learning rate must be below the initial rate")

    result = PPOOptimizationConfig(
        name=raw["name"],
        gamma=gamma,
        gae_lambda=gae_lambda,
        normalize_advantage=True,
        clip_epsilon=clip_epsilon,
        value_loss="l2",
        clip_value=False,
        value_coefficient=value_coefficient,
        entropy_coefficient=entropy_coefficient,
        optimizer="adam",
        learning_rate=learning_rate,
        adam_epsilon=adam_epsilon,
        weight_decay=weight_decay,
        max_gradient_norm=max_gradient_norm,
        epochs=integers["epochs"],
        batch_size=integers["batch_size"],
        minibatch_size=integers["minibatch_size"],
        minibatch_seed=integers["minibatch_seed"],
        scheduler="cosine",
        scheduler_total_optimizer_steps=integers["scheduler_total_optimizer_steps"],
        scheduler_minimum_learning_rate=minimum_learning_rate,
    )
    if result.scheduler_total_optimizer_steps < result.optimizer_steps_per_update:
        raise ValueError("scheduler horizon must cover at least one complete PPO update")
    return result


def _finite_float(value: object, name: str) -> float:
    if type(value) not in {int, float}:
        raise TypeError(f"train.{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"train.{name} must be finite")
    return result


def _positive_float(value: object, name: str) -> float:
    result = _finite_float(value, name)
    if result <= 0.0:
        raise ValueError(f"train.{name} must be positive")
    return result


def _non_negative_float(value: object, name: str) -> float:
    result = _finite_float(value, name)
    if result < 0.0:
        raise ValueError(f"train.{name} must be non-negative")
    return result


def _open_unit_float(value: object, name: str) -> float:
    result = _finite_float(value, name)
    if not 0.0 < result < 1.0:
        raise ValueError(f"train.{name} must be strictly inside (0, 1)")
    return result
