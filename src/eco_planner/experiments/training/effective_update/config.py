"""Typed manifest for the Task F effective-update region search."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, model_validator

from eco_planner._repository import CONFIG_ROOT
from eco_planner.configuration import load_resolved_yaml_mapping

DEFAULT_STUDY = CONFIG_ROOT / "experiments" / "training" / "effective-update.yaml"

_HELDOUT_METRICS = (
    "mean_speed_mps",
    "distance_m",
    "route_completion",
    "energy_total_ml",
    "energy_ml_per_km",
)

HeldoutMetricName = Literal[
    "mean_speed_mps",
    "distance_m",
    "route_completion",
    "energy_total_ml",
    "energy_ml_per_km",
]


class _StrictModel(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid", allow_inf_nan=False)


class GridConfig(_StrictModel):
    learning_rates: list[StrictFloat] = Field(min_length=1)
    epochs: list[StrictInt] = Field(min_length=1)
    max_gradient_norms: list[StrictFloat] = Field(min_length=1)

    @model_validator(mode="after")
    def _require_positive_sorted_unique(self) -> GridConfig:
        fields = (
            ("learning_rates", self.learning_rates),
            ("epochs", self.epochs),
            ("max_gradient_norms", self.max_gradient_norms),
        )
        for name, values in fields:
            if any(value <= 0 for value in values):
                raise ValueError(f"grid.{name} entries must be positive")
            if list(values) != sorted(set(values)):
                raise ValueError(f"grid.{name} entries must be sorted and unique")
        return self

    def combinations(self) -> tuple[tuple[float, int, float], ...]:
        """Enumerate the pre-registered grid in (lr, epochs, mgn) order."""

        return tuple(
            (learning_rate, epochs, max_gradient_norm)
            for learning_rate in self.learning_rates
            for epochs in self.epochs
            for max_gradient_norm in self.max_gradient_norms
        )


class GateConfig(_StrictModel):
    target_kl: StrictFloat = Field(gt=0.0)
    kl_floor: StrictFloat = Field(gt=0.0)
    within_target_kl_fraction: StrictFloat = Field(gt=0.0, le=1.0)
    runaway_tail_updates: StrictInt = Field(gt=0)
    runaway_tail_factor: StrictFloat = Field(gt=0.0)
    ratio_change_floor: StrictFloat = Field(gt=0.0)
    guidance_rms_shift_floor: StrictFloat = Field(gt=0.0)
    beta_parameter_floor: StrictFloat = Field(gt=0.0)
    boundary_mass_ceiling: StrictFloat = Field(gt=0.0, le=1.0)
    episode_length_retention_floor: StrictFloat = Field(gt=0.0, le=1.0)
    collision_budget: StrictInt = Field(ge=0)
    heldout_relative_change_floor: StrictFloat = Field(gt=0.0)
    heldout_metrics: list[HeldoutMetricName] = Field(min_length=1)


class EffectiveUpdateStudyConfig(_StrictModel):
    version: Literal[1]
    study_name: str = Field(min_length=1)
    protocol: str = Field(min_length=1)
    arm: Literal["a1"]
    training_seed: StrictInt = Field(ge=0)
    update_count: StrictInt = Field(gt=0)
    base_overrides: list[str] = Field(min_length=1)
    grid: GridConfig
    gate: GateConfig
    selection: Literal["lowest_learning_rate_then_epochs_then_max_gradient_norm"]

    def protocol_path(self) -> Path:
        return CONFIG_ROOT / self.protocol


def load_effective_update_study(path: Path) -> EffectiveUpdateStudyConfig:
    """Load one explicit effective-update study manifest."""

    return EffectiveUpdateStudyConfig.model_validate(load_resolved_yaml_mapping(path))
