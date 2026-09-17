"""Configuration and predeclared verdict for the replanning-deferral trace."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, model_validator

from eco_planner.analysis.deferral import (
    DeferralDesign,
)
from eco_planner.analysis.deferral import (
    analyze_deferral_episodes as describe_episodes,
)


class DeferralConfig(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid", allow_inf_nan=False)
    protocol: str
    runtime_seed: StrictInt = Field(ge=0)
    noise_seeds: list[StrictInt]
    longitudinal_actions: list[StrictFloat]
    lateral_action: StrictFloat
    total_window_steps: StrictInt = Field(ge=10, le=30)
    execution_steps: StrictInt
    required_scenarios: StrictInt = Field(gt=0)
    majority_fraction: StrictFloat = Field(gt=0, le=1)
    crossing_tolerance_steps: StrictInt = Field(ge=0)
    overrides: list[str]

    @model_validator(mode="after")
    def validate_protocol(self) -> DeferralConfig:
        if len(self.noise_seeds) < 2 or len(set(self.noise_seeds)) != len(self.noise_seeds):
            raise ValueError("at least two distinct noise seeds are required")
        if min(self.noise_seeds) < 0:
            raise ValueError("noise seeds must be non-negative")
        if self.longitudinal_actions != [-1.0, -0.5, 0.0, 0.5, 1.0]:
            raise ValueError("deferral requires the five fixed longitudinal guidance arms")
        if self.lateral_action != 0.0:
            raise ValueError("deferral fixes lateral guidance to zero")
        if self.execution_steps != 1:
            raise ValueError("deferral traces the baseline 0.1 s receding-horizon execution")
        return self


def design(config: DeferralConfig) -> DeferralDesign:
    return DeferralDesign(
        list(config.longitudinal_actions),
        list(config.noise_seeds),
        config.total_window_steps,
        config.execution_steps,
    )


def analyze_episodes(
    episodes: list[dict[str, Any]],
    config: DeferralConfig,
    scenario_names: list[str],
) -> dict[str, Any]:
    result = describe_episodes(
        episodes,
        design(config),
        scenario_names,
        majority_fraction=config.majority_fraction,
        crossing_tolerance_steps=config.crossing_tolerance_steps,
    )
    unsafe = result.pop("unsafe_or_incomplete_episodes")
    proxy_errors = result.pop("proxy_errors")
    threshold = config.majority_fraction
    repeated = [name for name, row in result["scenarios"].items() if row["repeated_deferral"]]
    absent = [
        name
        for name, row in result["scenarios"].items()
        if row["first_positive_count"] >= threshold * row["cycle_count"]
        and not row["first_negative_majority"]
    ]
    safety_clean = not unsafe and not proxy_errors
    if not safety_clean:
        status = "safety_or_proxy_failure"
    elif len(repeated) >= config.required_scenarios:
        status = "repeated_deferral"
    elif len(absent) >= config.required_scenarios:
        status = "deferral_absent"
    else:
        status = "mixed_or_inconclusive"
    return {
        **result,
        "gate_c": {
            "status": status,
            "execution_steps": config.execution_steps,
            "majority_fraction": config.majority_fraction,
            "crossing_tolerance_steps": config.crossing_tolerance_steps,
            "required_scenarios": config.required_scenarios,
            "repeated_deferral_scenarios": repeated,
            "first_waypoint_positive_scenarios": absent,
            "scenario_flags": {
                name: {
                    key: row[key]
                    for key in (
                        "first_negative_majority",
                        "full_positive_majority",
                        "deferred_majority",
                        "crossing_stable",
                        "repeated_deferral",
                    )
                }
                for name, row in result["scenarios"].items()
            },
            "safety_clean": bool(safety_clean),
            "unsafe_or_incomplete_episodes": unsafe,
            "proxy_errors": proxy_errors,
        },
    }
