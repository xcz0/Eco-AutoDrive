"""Configuration and predeclared verdict for the execution-horizon intervention."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, model_validator

from eco_planner.analysis.horizon import (
    PLANNER_RESPONSE_CHECKPOINTS_S,
    HorizonDesign,
)
from eco_planner.analysis.horizon import analyze_horizon_episodes as describe_episodes
from eco_planner.contracts import PLANNER_HORIZON

_REQUIRED_HORIZONS = (1, 2, 5, 10)


class HorizonInterventionConfig(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid", allow_inf_nan=False)
    protocol: str
    runtime_seed: StrictInt = Field(ge=0)
    noise_seeds: list[StrictInt]
    longitudinal_actions: list[StrictFloat]
    lateral_action: StrictFloat
    total_window_steps: StrictInt = Field(ge=10, le=30)
    execution_horizons: list[StrictInt]
    spearman_threshold: StrictFloat = Field(gt=0, le=1)
    noise_multiplier: StrictFloat = Field(gt=0)
    required_scenarios: StrictInt = Field(gt=0)
    overrides: list[str]

    @model_validator(mode="after")
    def validate_protocol(self) -> HorizonInterventionConfig:
        if len(self.noise_seeds) < 2 or len(set(self.noise_seeds)) != len(self.noise_seeds):
            raise ValueError("at least two distinct noise seeds are required")
        if min(self.noise_seeds) < 0:
            raise ValueError("noise seeds must be non-negative")
        if self.longitudinal_actions != [-1.0, -0.5, 0.0, 0.5, 1.0]:
            raise ValueError("horizon requires the five fixed longitudinal endpoints/interior arms")
        if self.lateral_action != 0.0:
            raise ValueError("horizon fixes lateral guidance to zero")
        if not self.execution_horizons or len(set(self.execution_horizons)) != len(
            self.execution_horizons
        ):
            raise ValueError("execution horizons must be non-empty and distinct")
        for horizon in self.execution_horizons:
            if not 1 <= horizon < PLANNER_HORIZON:
                raise ValueError("execution horizons must lie in [1, PLANNER_HORIZON)")
            if self.total_window_steps % horizon:
                raise ValueError("execution horizons must divide the total window")
        if not set(_REQUIRED_HORIZONS).issubset(self.execution_horizons):
            raise ValueError("execution horizons must include 0.1/0.2/0.5/1.0 s prefixes")
        return self


def design(config: HorizonInterventionConfig) -> HorizonDesign:
    return HorizonDesign(
        list(config.longitudinal_actions),
        list(config.noise_seeds),
        list(config.execution_horizons),
        config.total_window_steps,
    )


def _passed(row: dict[str, Any], config: HorizonInterventionConfig) -> bool:
    return bool(
        row["rho"] is not None
        and abs(row["rho"]) >= config.spearman_threshold
        and row["effect"] != 0
        and abs(row["effect"]) >= config.noise_multiplier * row["noise_scale"]
    )


def _apply(metric: dict[str, Any], config: HorizonInterventionConfig) -> str:
    rows = metric["scenarios"]
    for row in rows.values():
        row["passed"] = _passed(row, config)
    positive = sum(r["passed"] and r["effect"] > 0 for r in rows.values())
    negative = sum(r["passed"] and r["effect"] < 0 for r in rows.values())
    metric.update(
        positive_pass_count=positive,
        negative_pass_count=negative,
        passed=max(positive, negative) >= config.required_scenarios,
    )
    if positive >= config.required_scenarios and positive >= negative:
        direction = "positive"
    elif negative >= config.required_scenarios and negative > positive:
        direction = "negative"
    else:
        direction = "mixed"
    metric["direction"] = direction
    return direction


def analyze_episodes(
    episodes: list[dict[str, Any]],
    config: HorizonInterventionConfig,
    scenario_names: list[str],
) -> dict[str, Any]:
    result = describe_episodes(episodes, design(config), scenario_names)
    unsafe = result.pop("unsafe_or_incomplete_episodes")
    proxy_errors = result.pop("proxy_errors")
    directions: dict[str, str] = {}
    for horizon, data in result["horizons"].items():
        metric_directions = {}
        for metric, metrics in data["metrics"].items():
            metric_directions[metric] = _apply(metrics, config)
        data["direction"] = metric_directions["speed_mps"]
        directions[horizon] = data["direction"]
    planner_directions = {}
    for checkpoint, response in result["planner_response"].items():
        planner_directions[checkpoint] = _apply(response, config)
    first, second, third = (str(value) for value in _REQUIRED_HORIZONS[:3])
    safety_clean = not unsafe and not proxy_errors
    if not safety_clean:
        status = "safety_or_proxy_failure"
    elif (
        directions[first] == "negative"
        and directions[second] == "positive"
        and directions[third] == "positive"
    ):
        status = "strong_evidence_for_receding_horizon_mismatch"
    elif all(directions[str(value)] == "negative" for value in _REQUIRED_HORIZONS[:3]):
        status = "sign_unchanged_by_horizon"
    else:
        status = "mixed_or_inconclusive"
    short = "10" if "10" in result["horizons"] else str(max(config.execution_horizons))
    return {
        **result,
        "gate_a": {
            "status": status,
            "executed_speed_direction": directions,
            "planner_response_direction": planner_directions,
            "checkpoints_s": list(PLANNER_RESPONSE_CHECKPOINTS_S),
            "safety_clean": bool(safety_clean),
            "unsafe_or_incomplete_episodes": unsafe,
            "proxy_errors": proxy_errors,
            "energy_sensitivity_limited": bool(
                result["horizons"][short]["metrics"]["speed_mps"]["passed"]
                and not result["horizons"][short]["metrics"]["energy_ml_per_km"]["passed"]
            ),
        },
    }
