"""Run the fixed synthetic reward sanity audit declared in one strict config."""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Literal

from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictFloat

from eco_planner._repository import CONFIG_ROOT, REPOSITORY_ROOT
from eco_planner.artifacts import write_json
from eco_planner.configuration import load_resolved_yaml_mapping
from eco_planner.envs.domain import MetaDriveFuelProxyProvider
from eco_planner.envs.domain.metrics import TransitionMetricInput, derive_transition_metrics
from eco_planner.envs.domain.traffic import (
    StaticTrafficObjectState,
    TrafficFrame,
    TrafficParticipantState,
)
from eco_planner.rl.reward import PlannerRFTEnergyRewardConfig, score_plannerrft_energy_step

DEFAULT_CONFIG = CONFIG_ROOT / "experiments" / "reward" / "sanity.yaml"
_SCORE_FIELDS = (
    "reward_total",
    "reward_ungated",
    "reward_gate",
    "collision_score",
    "drivable_score",
    "wrong_direction_score",
    "ttc_score",
    "progress_score",
    "comfort_score",
    "speed_score",
    "energy_score",
)


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


class _RewardInputConfig(_StrictModel):
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


class _SanityConfig(_StrictModel):
    version: Literal[1]
    reward_config: str
    base_input: _RewardInputConfig
    cases: list[_CaseConfig] = Field(min_length=1)
    comparisons: list[_ComparisonConfig]


def load_sanity_config(path: Path) -> _SanityConfig:
    return _SanityConfig.model_validate(load_resolved_yaml_mapping(path))


def evaluate_sanity(config: _SanityConfig) -> dict[str, object]:
    reward_path = (REPOSITORY_ROOT / config.reward_config).resolve()
    reward_raw = load_resolved_yaml_mapping(reward_path)
    reward = PlannerRFTEnergyRewardConfig.model_validate(reward_raw)
    cases: dict[str, dict[str, object]] = {}
    checks: list[dict[str, object]] = []
    for case in config.cases:
        if case.name in cases:
            raise ValueError(f"duplicate reward sanity case {case.name!r}")
        values = config.base_input.model_dump(mode="python")
        values.update(case.overrides.model_dump(mode="python", exclude_none=True))
        metrics = derive_transition_metrics(_reward_input(values), MetaDriveFuelProxyProvider())
        audit = score_plannerrft_energy_step(reward, metrics)
        payload = asdict(audit)
        cases[case.name] = payload
        scores_valid = all(
            math.isfinite(float(payload[field])) and 0.0 <= float(payload[field]) <= 1.0
            for field in _SCORE_FIELDS
        )
        checks.append({"name": f"{case.name}:scores_finite_unit_interval", "passed": scores_valid})
        for expected in case.numeric_expectations:
            actual = _numeric_field(payload, expected.field)
            checks.append(
                {
                    "name": f"{case.name}:{expected.field}",
                    "passed": math.isclose(
                        actual, expected.value, rel_tol=0.0, abs_tol=expected.absolute_tolerance
                    ),
                    "actual": actual,
                    "expected": expected.value,
                }
            )
        for expected in case.boolean_expectations:
            actual = _boolean_field(payload, expected.field)
            checks.append(
                {
                    "name": f"{case.name}:{expected.field}",
                    "passed": actual is expected.value,
                    "actual": actual,
                    "expected": expected.value,
                }
            )
    for comparison in config.comparisons:
        left = _case_numeric(cases, comparison.left_case, comparison.left_field)
        right = _case_numeric(cases, comparison.right_case, comparison.right_field)
        difference = left - right if comparison.relation == "greater_than" else right - left
        checks.append(
            {
                "name": (
                    f"{comparison.left_case}:{comparison.left_field} "
                    f"{comparison.relation} {comparison.right_case}:{comparison.right_field}"
                ),
                "passed": difference > comparison.minimum_difference,
                "difference": difference,
                "minimum_difference": comparison.minimum_difference,
            }
        )
    passed = all(bool(item["passed"]) for item in checks)
    return {
        "status": "passed" if passed else "failed",
        "reward_profile": reward.name,
        "case_count": len(cases),
        "cases": cases,
        "checks": checks,
    }


def run_sanity(config_path: Path, output_root: Path) -> int:
    config = load_sanity_config(config_path)
    output_root.mkdir(parents=True, exist_ok=False)
    OmegaConf.save(OmegaConf.load(config_path), output_root / "sanity_manifest.yaml", resolve=True)
    reward_path = (REPOSITORY_ROOT / config.reward_config).resolve()
    OmegaConf.save(OmegaConf.load(reward_path), output_root / "resolved_reward.yaml", resolve=True)
    report = evaluate_sanity(config)
    write_json(output_root / "sanity_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


def _reward_input(values: dict[str, object]) -> TransitionMetricInput:
    parsed = _RewardInputConfig.model_validate(values)
    participants = tuple(
        TrafficParticipantState(
            object_id=item.object_id,
            kind=item.kind,
            position_xy_m=_point(item.position_xy_m),
            heading_rad=item.heading_rad,
            velocity_xy_mps=_point(item.velocity_xy_mps),
            width_m=item.width_m,
            length_m=item.length_m,
        )
        for item in parsed.participants
    )
    static_objects = tuple(
        StaticTrafficObjectState(
            object_id=item.object_id,
            kind=item.kind,
            position_xy_m=_point(item.position_xy_m),
            heading_rad=item.heading_rad,
            width_m=item.width_m,
            length_m=item.length_m,
        )
        for item in parsed.static_objects
    )
    return TransitionMetricInput(
        previous_position_xy_m=_point(parsed.previous_position_xy_m),
        position_xy_m=_point(parsed.position_xy_m),
        previous_velocity_xy_mps=_point(parsed.previous_velocity_xy_mps),
        velocity_xy_mps=_point(parsed.velocity_xy_mps),
        previous_acceleration_xy_mps2=_point(parsed.previous_acceleration_xy_mps2),
        heading_rad=parsed.heading_rad,
        yaw_rate_radps=parsed.yaw_rate_radps,
        route_progress_delta_m=parsed.route_progress_delta_m,
        route_heading_rad=parsed.route_heading_rad,
        speed_limit_mps=parsed.speed_limit_mps,
        ego_width_m=parsed.ego_width_m,
        ego_length_m=parsed.ego_length_m,
        traffic_frame=TrafficFrame(
            simulator_step=1,
            ego_center_xy_m=_point(parsed.position_xy_m),
            ego_heading_rad=parsed.heading_rad,
            ego_rear_wheelbase_m=1.0,
            participants=participants,
            static_objects=static_objects,
        ),
        target_position_xy_m=_point(parsed.position_xy_m),
        target_heading_rad=parsed.heading_rad,
        crash_vehicle=parsed.crash_vehicle,
        crash_object=parsed.crash_object,
        crash_building=parsed.crash_building,
        crash_human=parsed.crash_human,
        crash_sidewalk=parsed.crash_sidewalk,
        out_of_road=parsed.out_of_road,
        native_step_energy_ml=parsed.native_step_energy_ml,
        native_episode_energy_ml=parsed.native_episode_energy_ml,
        timestep_s=parsed.timestep_s,
    )


def _point(values: list[StrictFloat]) -> tuple[float, float]:
    return (float(values[0]), float(values[1]))


def _numeric_field(payload: dict[str, object], field: str) -> float:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"reward audit field {field!r} is not numeric")
    return float(value)


def _boolean_field(payload: dict[str, object], field: str) -> bool:
    value = payload.get(field)
    if not isinstance(value, bool):
        raise ValueError(f"reward audit field {field!r} is not boolean")
    return value


def _case_numeric(cases: dict[str, dict[str, object]], case: str, field: str) -> float:
    if case not in cases:
        raise ValueError(f"reward sanity comparison references unknown case {case!r}")
    return _numeric_field(cases[case], field)
