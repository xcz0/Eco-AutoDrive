"""Configuration and pure diagnostics for control authority."""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, model_validator

from eco_planner.analysis.guidance import InterventionDesign
from eco_planner.analysis.guidance import analyze_episodes as describe_episodes
from eco_planner.analysis.guidance import matched_statistics as describe_matched


class InterventionConfig(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid", allow_inf_nan=False)
    protocol: str
    runtime_seed: StrictInt = Field(ge=0)
    noise_seeds: list[StrictInt]
    longitudinal_actions: list[StrictFloat]
    lateral_action: StrictFloat
    window_steps: StrictInt = Field(ge=10, le=30)
    spearman_threshold: StrictFloat = Field(gt=0, le=1)
    noise_multiplier: StrictFloat = Field(gt=0)
    required_scenarios: StrictInt = Field(gt=0)
    overrides: list[str]

    @model_validator(mode="after")
    def validate_protocol(self) -> InterventionConfig:
        if len(self.noise_seeds) < 2 or len(set(self.noise_seeds)) != len(self.noise_seeds):
            raise ValueError("at least two distinct noise seeds are required")
        if min(self.noise_seeds) < 0:
            raise ValueError("noise seeds must be non-negative")
        if self.longitudinal_actions != [-1.0, -0.5, 0.0, 0.5, 1.0]:
            raise ValueError("Task D requires the five fixed longitudinal endpoints/interior arms")
        if self.lateral_action != 0.0:
            raise ValueError("Task D fixes lateral guidance to zero")
        return self


def design(config: InterventionConfig) -> InterventionDesign:
    return InterventionDesign(config.longitudinal_actions, config.noise_seeds, config.window_steps)


def _passed(row: dict[str, Any], config: InterventionConfig) -> bool:
    return bool(
        row["rho"] is not None
        and abs(row["rho"]) >= config.spearman_threshold
        and row["effect"] != 0
        and abs(row["effect"]) >= config.noise_multiplier * row["noise_scale"]
    )


def matched_statistics(values: np.ndarray, config: InterventionConfig) -> dict[str, Any]:
    row = describe_matched(values, design(config))
    return {**row, "passed": _passed(row, config)}


def analyze_episodes(
    episodes: list[dict[str, Any]],
    config: InterventionConfig,
    scenario_names: list[str],
) -> dict[str, Any]:
    result = describe_episodes(episodes, design(config), scenario_names)
    unsafe = result.pop("unsafe_or_incomplete_episodes")
    proxy_errors = result.pop("proxy_errors")
    windows = result["windows"]
    for metrics in windows.values():
        for metric in metrics.values():
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
    short = windows["short_horizon"]
    immediate = windows["immediate"]
    contradictions = []
    linked = []
    for scenario in scenario_names:
        speed = short["speed_mps"]["scenarios"][scenario]
        planner = short["planner_first_speed_mps"]["scenarios"][scenario]
        energy = short["energy_ml_per_km"]["scenarios"][scenario]
        if (
            (speed["passed"] or energy["passed"])
            and planner["passed"]
            and speed["effect"] is not None
            and speed["effect"] * planner["effect"] > 0
        ):
            linked.append(scenario)
        for metric in ("speed_mps", "energy_ml_per_km"):
            a = immediate[metric]["scenarios"][scenario]
            b = short[metric]["scenarios"][scenario]
            if a["passed"] and b["passed"] and a["effect"] * b["effect"] < 0:
                contradictions.append({"scenario": scenario, "metric": metric})
    numerical = short["speed_mps"]["passed"] or short["energy_ml_per_km"]["passed"]
    # Planner-output response is assessed independently of the configured target delta.
    chain = len(linked) >= config.required_scenarios
    passed = numerical and not unsafe and not proxy_errors and not contradictions and chain
    if passed:
        attribution = "matched_control_authority"
    elif unsafe:
        attribution = "safety_or_window_validity_failed"
    elif proxy_errors:
        attribution = "energy_proxy_inconsistency"
    elif contradictions:
        attribution = "temporal_response_requires_explanation"
    elif not short["planner_first_speed_mps"]["passed"]:
        attribution = "guidance_injection_or_frozen_planner_authority"
    elif not numerical:
        attribution = "planner_to_execution_or_noise_limited_response"
    else:
        attribution = "response_chain_unresolved"
    return {
        **result,
        "gate_d": {
            "passed": bool(passed),
            "numerical_passed": bool(numerical),
            "attribution": attribution,
            "unsafe_or_incomplete_episodes": unsafe,
            "proxy_errors": proxy_errors,
            "temporal_contradictions": contradictions,
            "linked_scenarios": linked,
            "energy_sensitivity_limited": bool(
                short["speed_mps"]["passed"] and not short["energy_ml_per_km"]["passed"]
            ),
        },
    }
