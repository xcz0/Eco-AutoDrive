"""Strict manifest for the Issue #105 Phase C canonical PPO control re-freeze.

Phase C re-locates one canonical k=5 effective-update control from a search space
that was frozen before any result was observed. The search space is a declared,
ordered list of named candidates restricted to the Phase C optimizer knobs
(learning rate, epochs/minibatch, time-consistent discount/GAE, value-loss
scaling, clipping). Reward representation, lambda, policy features and execution
cadence are invariant and rejected if a candidate tries to override them. A
canonical calibrated-R0 reference is trained for matched comparison.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from eco_planner._repository import CONFIG_ROOT
from eco_planner.configuration import load_resolved_yaml_mapping

from ..config import GateConfig

# The full Phase C control search space. A candidate may only override these
# optimizer knobs; every other override would change the frozen experiment
# semantics (reward representation / lambda / policy feature / cadence / batch).
ALLOWED_CONTROL_OVERRIDE_KEYS = frozenset(
    {
        "ppo.learning_rate",
        "ppo.epochs",
        "ppo.minibatch_size",
        "ppo.gamma",
        "ppo.gae_lambda",
        "ppo.value_coefficient",
        "ppo.max_gradient_norm",
        "ppo.clip_epsilon",
    }
)

SelectionRule = Literal["first_passing_in_declared_order"]


class ControlCandidate(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid", allow_inf_nan=False)

    label: str = Field(min_length=1)
    overrides: list[str]
    description: str = Field(min_length=1)


class ControlRefreezeConfig(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid", allow_inf_nan=False)

    version: Literal[1]
    study_name: str = Field(min_length=1)
    protocol: str = Field(min_length=1)
    arm: str
    training_seed: int = Field(ge=0)
    update_count: int = Field(gt=0)
    base_overrides: list[str]
    mc_draws: int = Field(gt=0)
    mc_seed: int = Field(ge=0)
    gate: GateConfig
    selection: SelectionRule
    historical_comparator: str = Field(min_length=1)
    reference: ControlCandidate
    candidates: list[ControlCandidate] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_design(self) -> ControlRefreezeConfig:
        if self.reference.overrides:
            raise ValueError("reference arm must not carry overrides")
        labels = [self.reference.label, *(candidate.label for candidate in self.candidates)]
        if len(set(labels)) != len(labels):
            raise ValueError("candidate labels must be unique")
        for candidate in self.candidates:
            if not candidate.overrides:
                raise ValueError(f"candidate {candidate.label} must carry at least one override")
            if candidate.label == self.reference.label:
                raise ValueError("candidate labels must differ from the reference label")
            if any("runtime.seed" in override for override in candidate.overrides):
                raise ValueError(f"candidate {candidate.label} must not override runtime.seed")
            keys = [override.split("=", 1)[0] for override in candidate.overrides]
            if len(set(keys)) != len(keys):
                raise ValueError(f"candidate {candidate.label} repeats an override key")
            outside = sorted(key for key in keys if key not in ALLOWED_CONTROL_OVERRIDE_KEYS)
            if outside:
                raise ValueError(
                    f"candidate {candidate.label} overrides non-control keys: {outside}"
                )
        return self

    def protocol_path(self) -> Path:
        return CONFIG_ROOT / self.protocol

    def all_runs(self) -> list[ControlCandidate]:
        return [self.reference, *self.candidates]


def load_control_refreeze(path: Path) -> ControlRefreezeConfig:
    """Load one explicit Phase C control re-freeze manifest."""

    return ControlRefreezeConfig.model_validate(load_resolved_yaml_mapping(path))
