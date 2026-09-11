"""Typed manifest for the Task G objective positive-control study."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, model_validator

from eco_planner._repository import CONFIG_ROOT
from eco_planner.configuration import load_resolved_yaml_mapping

DEFAULT_STUDY = CONFIG_ROOT / "experiments" / "training" / "objective-positive-control.yaml"

R0ProfileName = Literal["plannerrft_no_energy_calibrated_v1"]
RstressProfileName = Literal["plannerrft_energy_band_lam64_v1"]

SeparationMetricName = Literal["mean_speed_mps", "energy_ml_per_km", "route_completion"]


class _StrictModel(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid", allow_inf_nan=False)


class R0ArmConfig(_StrictModel):
    """Calibrated no-energy R0 anchor (E-038 decomposition R0 arm as a profile)."""

    label: str = Field(min_length=1)
    reward_profile: R0ProfileName


class RstressArmConfig(_StrictModel):
    """Calibrated band-energy lambda=64 stress arm (E-038 lambda64 arm as a profile)."""

    label: str = Field(min_length=1)
    reward_profile: RstressProfileName


class PpoFreezeConfig(_StrictModel):
    """The Task F frozen effective PPO control (E-039 selection rule output)."""

    learning_rate: StrictFloat = Field(gt=0.0)
    epochs: StrictInt = Field(gt=0)
    max_gradient_norm: StrictFloat = Field(gt=0.0)


class GateGConfig(_StrictModel):
    guidance_rms_floor: StrictFloat = Field(gt=0.0)
    probe_match_tolerance: StrictFloat = Field(gt=0.0)
    paired_relative_effect_floor: StrictFloat = Field(gt=0.0)
    separation_metrics: list[SeparationMetricName] = Field(min_length=1)
    collision_budget: StrictInt = Field(ge=0)
    arrive_dest_drop_ceiling: StrictFloat = Field(ge=0.0, le=1.0)
    stopped_fraction_increase_ceiling: StrictFloat = Field(ge=0.0, le=1.0)
    route_completion_drop_ceiling: StrictFloat = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _require_speed_or_energy_metric(self) -> GateGConfig:
        required = {"mean_speed_mps", "energy_ml_per_km"}
        if not required & set(self.separation_metrics):
            raise ValueError(
                "gate.separation_metrics must include mean_speed_mps or energy_ml_per_km "
                "so Gate G condition 4 has an objective-direction metric"
            )
        return self


class ObjectivePositiveControlStudyConfig(_StrictModel):
    version: Literal[1]
    study_name: str = Field(min_length=1)
    protocol: str = Field(min_length=1)
    arms: Literal["r0_vs_rstress"] = "r0_vs_rstress"
    r0: R0ArmConfig
    rstress: RstressArmConfig
    training_seeds: list[StrictInt] = Field(min_length=2)
    update_count: StrictInt = Field(gt=0)
    ppo: PpoFreezeConfig
    base_overrides: list[str] = Field(min_length=1)
    gate: GateGConfig

    @model_validator(mode="after")
    def _require_unique_seeds(self) -> ObjectivePositiveControlStudyConfig:
        if len(set(self.training_seeds)) != len(self.training_seeds):
            raise ValueError("training_seeds must be unique")
        return self

    def protocol_path(self) -> Path:
        return CONFIG_ROOT / self.protocol


def load_objective_positive_control_study(path: Path) -> ObjectivePositiveControlStudyConfig:
    """Load one explicit objective positive-control study manifest."""

    return ObjectivePositiveControlStudyConfig.model_validate(load_resolved_yaml_mapping(path))
