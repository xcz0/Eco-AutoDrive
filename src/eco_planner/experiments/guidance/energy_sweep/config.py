"""Configuration for the fixed-seed energy benchmark matrix."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, field_validator

from eco_planner._repository import CONFIG_ROOT
from eco_planner.configuration import load_resolved_yaml_mapping

DEFAULT_STUDY = CONFIG_ROOT / "experiments" / "guidance" / "energy-sweep" / "matrix.yaml"


class _StrictModel(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid", allow_inf_nan=False)


class EnergyMetricSpec(_StrictModel):
    name: Literal["metadrive_fuel_proxy"]
    implementation: Literal["recompute_metadrive_base_vehicle_formula_on_executed_trace"]
    formula: str = Field(min_length=1)
    unit: Literal["mL"]
    sampling_interval_s: float = Field(gt=0.0)
    interpretation: str = Field(min_length=1)


class GuidanceProfileSpec(_StrictModel):
    id: Literal[
        "baseline",
        "longitudinal_negative",
        "longitudinal_zero",
        "longitudinal_positive",
    ]
    config: str = Field(min_length=1)
    longitudinal_scale: StrictFloat | None


class EvaluationJobSpec(_StrictModel):
    id: str = Field(min_length=1)
    config_name: str = Field(min_length=1)


class EnergyStudyConfig(_StrictModel):
    version: Literal[1]
    energy_metric: EnergyMetricSpec
    guidance_profiles: tuple[GuidanceProfileSpec, ...]
    jobs: tuple[EvaluationJobSpec, ...]

    @field_validator("guidance_profiles", "jobs", mode="before")
    @classmethod
    def tuple_items(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    def validate_study_contract(self) -> None:
        profile_ids = tuple(profile.id for profile in self.guidance_profiles)
        if profile_ids != (
            "baseline",
            "longitudinal_negative",
            "longitudinal_zero",
            "longitudinal_positive",
        ):
            raise ValueError("energy study guidance profiles are incomplete or out of order")


def load_energy_study(path: Path) -> EnergyStudyConfig:
    config = EnergyStudyConfig.model_validate(load_resolved_yaml_mapping(path))
    config.validate_study_contract()
    return config
