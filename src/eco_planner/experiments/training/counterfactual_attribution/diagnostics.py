"""Strict manifest for the Issue #105 Phase B causal-counterfactual study.

The study trains a canonical k=5 calibrated-R0 reference plus named single-mechanism
diagnostic arms that keep k=5 cadence and vary one critic/reward/credit mechanism.
Every arm is a matched causal diagnostic; none is a formal execution or optimizer
contract.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from eco_planner._repository import CONFIG_ROOT
from eco_planner.configuration import load_resolved_yaml_mapping

from ..config import GateConfig

Mechanism = Literal[
    "canonical_k5_reference",
    "cadence_reference",
    "reward_value_scale",
    "physical_time_credit",
    "critic_coupling",
]


class CounterfactualArm(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid", allow_inf_nan=False)

    label: str = Field(min_length=1)
    mechanism: Mechanism
    overrides: list[str]
    description: str = Field(min_length=1)


class CounterfactualAttributionConfig(BaseModel):
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
    historical_comparator: str = Field(min_length=1)
    reference: CounterfactualArm
    arms: list[CounterfactualArm] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_design(self) -> CounterfactualAttributionConfig:
        if self.reference.overrides:
            raise ValueError("reference arm must not carry overrides")
        labels = [self.reference.label, *(arm.label for arm in self.arms)]
        if len(set(labels)) != len(labels):
            raise ValueError("arm labels must be unique")
        for arm in self.arms:
            if not arm.overrides:
                raise ValueError(f"arm {arm.label} must carry at least one override")
            if arm.label == self.reference.label:
                raise ValueError("arm labels must differ from the reference label")
            if any("runtime.seed" in override for override in arm.overrides):
                raise ValueError(f"arm {arm.label} must not override runtime.seed")
            keys = [override.split("=", 1)[0] for override in arm.overrides]
            if len(set(keys)) != len(keys):
                raise ValueError(f"arm {arm.label} repeats an override key")
        return self

    def protocol_path(self) -> Path:
        return CONFIG_ROOT / self.protocol

    def all_arms(self) -> list[CounterfactualArm]:
        return [self.reference, *self.arms]


def load_counterfactual_attribution(path: Path) -> CounterfactualAttributionConfig:
    """Load one explicit Phase B causal-counterfactual manifest."""

    return CounterfactualAttributionConfig.model_validate(load_resolved_yaml_mapping(path))
