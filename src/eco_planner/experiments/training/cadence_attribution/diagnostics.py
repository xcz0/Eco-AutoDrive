"""Strict manifest for the Issue #105 Phase A matched cadence attribution study.

The study runs the same calibrated policy/environment twice, differing only in
the execution-prefix length (a diagnostic k=1 arm and the canonical k=5 arm),
and compares the full diagnostic set. The non-canonical arm is a matched causal
diagnostic; it never becomes the formal execution contract.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

from eco_planner._repository import CONFIG_ROOT
from eco_planner.configuration import load_resolved_yaml_mapping
from eco_planner.contracts import CLOSED_LOOP_EXECUTION_STEPS, PLANNER_HORIZON

from ..config import GateConfig


class CadenceAttributionConfig(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid", allow_inf_nan=False)

    version: Literal[1]
    study_name: str = Field(min_length=1)
    protocol: str = Field(min_length=1)
    arm: str
    training_seed: StrictInt = Field(ge=0)
    update_count: StrictInt = Field(gt=0)
    base_overrides: list[str]
    execution_steps: list[StrictInt] = Field(min_length=2)
    mc_draws: StrictInt = Field(gt=0)
    mc_seed: StrictInt = Field(ge=0)
    gate: GateConfig
    historical_comparator: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_design(self) -> CadenceAttributionConfig:
        if list(self.execution_steps) != sorted(set(self.execution_steps)):
            raise ValueError("execution_steps must be sorted and unique")
        if any(not (1 <= steps < PLANNER_HORIZON) for steps in self.execution_steps):
            raise ValueError("execution_steps must lie in [1, PLANNER_HORIZON)")
        if CLOSED_LOOP_EXECUTION_STEPS not in self.execution_steps:
            raise ValueError("execution_steps must include the canonical closed-loop cadence")
        if all(steps == CLOSED_LOOP_EXECUTION_STEPS for steps in self.execution_steps):
            raise ValueError("execution_steps must include one non-canonical diagnostic arm")
        return self

    def protocol_path(self) -> Path:
        return CONFIG_ROOT / self.protocol

    def diagnostic_steps(self) -> int:
        return next(steps for steps in self.execution_steps if steps != CLOSED_LOOP_EXECUTION_STEPS)

    def arm_label(self, execution_steps: int) -> str:
        return f"k{execution_steps}"


def load_cadence_attribution(path: Path) -> CadenceAttributionConfig:
    """Load one explicit matched cadence-attribution manifest."""

    return CadenceAttributionConfig.model_validate(load_resolved_yaml_mapping(path))
