"""Explicit objective and temporal-credit axes for fixed-batch diagnostics."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, model_validator

from eco_planner.reward import CalibrationTargets, EnergyBandConfig

from .decisions import AttributionThresholds, GateThresholds


class RewardArm(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid", allow_inf_nan=False)
    label: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    weight: StrictFloat | Literal["energy_only"]

    @model_validator(mode="after")
    def valid_weight(self) -> RewardArm:
        if isinstance(self.weight, float) and self.weight < 0:
            raise ValueError("reward weight must be nonnegative")
        return self


class CreditStudyConfig(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid", allow_inf_nan=False)
    arms: list[RewardArm] = Field(min_length=2)
    advantage_forms: list[Literal["raw", "center", "z"]] = Field(min_length=1)
    credit_forms: list[Literal["standard_gae", "reward_only_gae", "discounted_return"]] = Field(
        min_length=1
    )
    quantiles: list[StrictFloat] = Field(min_length=2)
    value_target_ddof: Literal[0, 1]
    calibration: CalibrationTargets | None
    energy_band: EnergyBandConfig | None
    objective_gate: GateThresholds | None
    attribution_gate: AttributionThresholds | None

    @model_validator(mode="after")
    def validate_design(self) -> CreditStudyConfig:
        for axis in ([a.label for a in self.arms], self.advantage_forms, self.credit_forms):
            if len(axis) != len(set(axis)):
                raise ValueError("diagnostic axes must be unique")
        if (
            self.quantiles != sorted(set(self.quantiles))
            or self.quantiles[0] != 0
            or self.quantiles[-1] != 1
        ):
            raise ValueError("quantiles must increase from zero to one")
        if self.objective_gate or self.attribution_gate:
            endpoints = {a.label: a.weight for a in self.arms}
            if endpoints.get("r0") != 0.0 or endpoints.get("energy_only") != "energy_only":
                raise ValueError("gates require declared r0 and energy_only endpoints")
            if set(self.advantage_forms) != {"raw", "center", "z"}:
                raise ValueError("gates require raw, center and z objectives")
            if "standard_gae" not in self.credit_forms:
                raise ValueError("gates require standard GAE")
        if self.attribution_gate and set(self.credit_forms) != {
            "standard_gae",
            "reward_only_gae",
            "discounted_return",
        }:
            raise ValueError("attribution requires all three temporal-credit forms")
        if (
            self.objective_gate
            and sum(isinstance(a.weight, float) and a.weight > 0 for a in self.arms) < 2
        ):
            raise ValueError("objective gate requires at least two positive lambda arms")
        return self
