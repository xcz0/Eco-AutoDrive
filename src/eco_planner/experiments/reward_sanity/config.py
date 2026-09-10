"""Configuration for the fixed synthetic reward sanity audit."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictFloat

from eco_planner._repository import CONFIG_ROOT
from eco_planner.configuration import load_resolved_yaml_mapping

DEFAULT_CONFIG = CONFIG_ROOT / "experiments" / "reward-sanity" / "sanity.yaml"


class _StrictModel(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid", allow_inf_nan=False)


class _ParticipantConfig(_StrictModel):
    object_id: str
    kind: Literal["vehicle", "pedestrian", "bicycle"]
    position_xy_m: list[StrictFloat] = Field(min_length=2, max_length=2)
    heading_rad: StrictFloat
    velocity_xy_mps: list[StrictFloat] = Field(min_length=2, max_length=2)
    width_m: StrictFloat = Field(gt=0.0)
    length_m: StrictFloat = Field(gt=0.0)


class _StaticObjectConfig(_StrictModel):
    object_id: str
    kind: Literal["barrier", "traffic_cone", "generic"]
    position_xy_m: list[StrictFloat] = Field(min_length=2, max_length=2)
    heading_rad: StrictFloat
    width_m: StrictFloat = Field(gt=0.0)
    length_m: StrictFloat = Field(gt=0.0)


class RewardInputConfig(_StrictModel):
    previous_position_xy_m: list[StrictFloat] = Field(min_length=2, max_length=2)
    position_xy_m: list[StrictFloat] = Field(min_length=2, max_length=2)
    previous_velocity_xy_mps: list[StrictFloat] = Field(min_length=2, max_length=2)
    velocity_xy_mps: list[StrictFloat] = Field(min_length=2, max_length=2)
    previous_acceleration_xy_mps2: list[StrictFloat] = Field(min_length=2, max_length=2)
    heading_rad: StrictFloat
    yaw_rate_radps: StrictFloat
    route_progress_delta_m: StrictFloat
    route_heading_rad: StrictFloat
    speed_limit_mps: StrictFloat = Field(gt=0.0)
    ego_width_m: StrictFloat = Field(gt=0.0)
    ego_length_m: StrictFloat = Field(gt=0.0)
    participants: list[_ParticipantConfig]
    static_objects: list[_StaticObjectConfig]
    crash_vehicle: StrictBool
    crash_object: StrictBool
    crash_building: StrictBool
    crash_human: StrictBool
    crash_sidewalk: StrictBool
    out_of_road: StrictBool
    native_step_energy_ml: StrictFloat = Field(ge=0.0)
    native_episode_energy_ml: StrictFloat = Field(ge=0.0)
    timestep_s: StrictFloat = Field(gt=0.0)


class _InputOverrides(_StrictModel):
    previous_position_xy_m: list[StrictFloat] | None = Field(
        default=None, min_length=2, max_length=2
    )
    position_xy_m: list[StrictFloat] | None = Field(default=None, min_length=2, max_length=2)
    previous_velocity_xy_mps: list[StrictFloat] | None = Field(
        default=None, min_length=2, max_length=2
    )
    velocity_xy_mps: list[StrictFloat] | None = Field(default=None, min_length=2, max_length=2)
    previous_acceleration_xy_mps2: list[StrictFloat] | None = Field(
        default=None, min_length=2, max_length=2
    )
    heading_rad: StrictFloat | None = None
    yaw_rate_radps: StrictFloat | None = None
    route_progress_delta_m: StrictFloat | None = None
    route_heading_rad: StrictFloat | None = None
    speed_limit_mps: StrictFloat | None = Field(default=None, gt=0.0)
    participants: list[_ParticipantConfig] | None = None
    static_objects: list[_StaticObjectConfig] | None = None
    crash_vehicle: StrictBool | None = None
    crash_object: StrictBool | None = None
    crash_building: StrictBool | None = None
    crash_human: StrictBool | None = None
    crash_sidewalk: StrictBool | None = None
    out_of_road: StrictBool | None = None


class _NumericExpectation(_StrictModel):
    field: str
    value: StrictFloat
    absolute_tolerance: StrictFloat = Field(ge=0.0)


class _BooleanExpectation(_StrictModel):
    field: str
    value: StrictBool


class _CaseConfig(_StrictModel):
    name: str
    overrides: _InputOverrides
    numeric_expectations: list[_NumericExpectation]
    boolean_expectations: list[_BooleanExpectation]


class _ComparisonConfig(_StrictModel):
    left_case: str
    left_field: str
    relation: Literal["greater_than", "less_than"]
    right_case: str
    right_field: str
    minimum_difference: StrictFloat = Field(ge=0.0)


class SanityConfig(_StrictModel):
    version: Literal[1]
    reward_config: str
    base_input: RewardInputConfig
    cases: list[_CaseConfig] = Field(min_length=1)
    comparisons: list[_ComparisonConfig]


def load_sanity_config(path: Path) -> SanityConfig:
    return SanityConfig.model_validate(load_resolved_yaml_mapping(path))
