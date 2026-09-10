from __future__ import annotations

import math
from dataclasses import asdict

from pydantic import StrictFloat, TypeAdapter

from eco_planner._repository import REPOSITORY_ROOT
from eco_planner.configuration import load_resolved_yaml_mapping
from eco_planner.envs import (
    MetaDriveFuelProxyProvider,
    StaticTrafficObjectState,
    TrafficFrame,
    TrafficParticipantState,
    TransitionMetricInput,
    derive_transition_metrics,
)
from eco_planner.rl.reward import RewardEvaluator, RewardProfileConfig

from .config import RewardInputConfig, SanityConfig

_SCORE_FIELDS = (
    "total",
    "base_total",
    "safety_gate",
    "diagnostics.collision_score",
    "diagnostics.drivable_score",
    "diagnostics.wrong_direction_score",
    "components.ttc",
    "components.progress",
    "components.comfort",
    "components.speed",
    "components.energy",
)


def evaluate_sanity(config: SanityConfig) -> dict[str, object]:
    reward_path = (REPOSITORY_ROOT / config.reward_config).resolve()
    reward_raw = load_resolved_yaml_mapping(reward_path)
    reward = TypeAdapter(RewardProfileConfig).validate_python(reward_raw)
    evaluator = RewardEvaluator(reward)
    cases: dict[str, dict[str, object]] = {}
    checks: list[dict[str, object]] = []
    for case in config.cases:
        if case.name in cases:
            raise ValueError(f"duplicate reward sanity case {case.name!r}")
        values = config.base_input.model_dump(mode="python")
        values.update(case.overrides.model_dump(mode="python", exclude_none=True))
        metrics = derive_transition_metrics(_reward_input(values), MetaDriveFuelProxyProvider())
        result = evaluator(metrics)
        payload = asdict(result)
        cases[case.name] = payload
        scores_valid = all(
            math.isfinite(_numeric_field(payload, field))
            and 0.0 <= _numeric_field(payload, field) <= 1.0
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


def _reward_input(values: dict[str, object]) -> TransitionMetricInput:
    parsed = RewardInputConfig.model_validate(values)
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
    value = _field(payload, field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"reward audit field {field!r} is not numeric")
    return float(value)


def _boolean_field(payload: dict[str, object], field: str) -> bool:
    value = _field(payload, field)
    if not isinstance(value, bool):
        raise ValueError(f"reward audit field {field!r} is not boolean")
    return value


def _field(payload: dict[str, object], field: str) -> object:
    value: object = payload
    for part in field.split("."):
        if not isinstance(value, dict) or part not in value:
            raise ValueError(f"reward result does not contain field {field!r}")
        value = value[part]
    return value


def _case_numeric(cases: dict[str, dict[str, object]], case: str, field: str) -> float:
    if case not in cases:
        raise ValueError(f"reward sanity comparison references unknown case {case!r}")
    return _numeric_field(cases[case], field)
