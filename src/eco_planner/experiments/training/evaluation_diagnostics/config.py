"""Typed manifest for the Task H evaluation-diagnostics study."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

from eco_planner._repository import CONFIG_ROOT
from eco_planner.configuration import load_resolved_yaml_mapping

DEFAULT_STUDY = CONFIG_ROOT / "experiments" / "training" / "evaluation-diagnostics.yaml"


class _StrictModel(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid", allow_inf_nan=False)


class EvaluationDiagnosticsStudyConfig(_StrictModel):
    """Reference one frozen positive-control study and its stochastic seeds."""

    version: Literal[1]
    study_name: str = Field(min_length=1)
    protocol: str = Field(min_length=1)
    training_seeds: list[StrictInt] = Field(min_length=1)
    policy_action_seeds: list[StrictInt] = Field(min_length=1)

    @model_validator(mode="after")
    def _require_unique_non_negative_seeds(self) -> EvaluationDiagnosticsStudyConfig:
        if len(set(self.training_seeds)) != len(self.training_seeds):
            raise ValueError("training_seeds must be unique")
        if len(set(self.policy_action_seeds)) != len(self.policy_action_seeds):
            raise ValueError("policy_action_seeds must be unique")
        if any(seed < 0 for seed in self.policy_action_seeds):
            raise ValueError("policy_action_seeds must be non-negative")
        return self

    def protocol_path(self) -> Path:
        return CONFIG_ROOT / self.protocol


def load_evaluation_diagnostics_study(path: Path) -> EvaluationDiagnosticsStudyConfig:
    """Load one explicit evaluation-diagnostics study manifest."""

    return EvaluationDiagnosticsStudyConfig.model_validate(load_resolved_yaml_mapping(path))
