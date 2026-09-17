"""Configuration and predeclared attribution verdict for the lon/lat decomposition."""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, model_validator

from eco_planner.analysis.decomposition import (
    ARM_NAMES,
    DecompositionDesign,
)
from eco_planner.analysis.decomposition import (
    analyze_decomposition_episodes as describe_episodes,
)

_ATTRIBUTION_METRICS = ("speed_mps", "energy_ml_per_km")


class ArmAction(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid", allow_inf_nan=False)
    name: Literal["r0", "lon", "lat", "joint"]
    lateral: StrictFloat
    longitudinal: StrictFloat


class SeedArms(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid", allow_inf_nan=False)
    seed: StrictInt
    arms: list[ArmAction]


class DecompositionConfig(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid", allow_inf_nan=False)
    protocol: str
    job: str
    runtime_seed: StrictInt
    noise_seeds: list[StrictInt]
    training_seeds: list[StrictInt]
    seed_arms: list[SeedArms]
    expected_joint_direction: Literal["positive", "negative"]
    dominance_share: StrictFloat = Field(gt=0.0, le=1.0)
    required_scenarios: StrictInt = Field(gt=0)
    overrides: list[str]

    @model_validator(mode="after")
    def validate_protocol(self) -> DecompositionConfig:
        if not self.noise_seeds or len(set(self.noise_seeds)) != len(self.noise_seeds):
            raise ValueError("noise seeds must be non-empty and distinct")
        if min(self.noise_seeds) < 0 or self.runtime_seed < 0:
            raise ValueError("runtime and noise seeds must be non-negative")
        declared = [spec.seed for spec in self.seed_arms]
        if declared != self.training_seeds or len(set(declared)) != len(declared):
            raise ValueError("seed_arms must declare each training seed exactly once in order")
        for spec in self.seed_arms:
            _validate_arms(spec)
        return self


def _validate_arms(spec: SeedArms) -> None:
    names = [arm.name for arm in spec.arms]
    if names != list(ARM_NAMES):
        raise ValueError(f"seed {spec.seed} must declare arms {list(ARM_NAMES)} in order")
    arms = {arm.name: (arm.lateral, arm.longitudinal) for arm in spec.arms}
    for lateral, longitudinal in arms.values():
        if not (-1.0 <= lateral <= 1.0 and -1.0 <= longitudinal <= 1.0):
            raise ValueError("arm actions must lie in [-1, 1]")
    r0, lon, lat, joint = (arms[name] for name in ARM_NAMES)

    def close(left: float, right: float) -> bool:
        return math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)

    if not close(lon[0], r0[0]) or close(lon[1], r0[1]):
        raise ValueError("lon arm must differ from r0 only in the longitudinal dimension")
    if not close(lat[1], r0[1]) or close(lat[0], r0[0]):
        raise ValueError("lat arm must differ from r0 only in the lateral dimension")
    expected = (
        r0[0] + (lon[0] - r0[0]) + (lat[0] - r0[0]),
        r0[1] + (lon[1] - r0[1]) + (lat[1] - r0[1]),
    )
    if not close(joint[0], expected[0]) or not close(joint[1], expected[1]):
        raise ValueError("joint arm must equal r0 + (lon - r0) + (lat - r0) elementwise")


def design(config: DecompositionConfig) -> DecompositionDesign:
    return DecompositionDesign(
        list(config.noise_seeds),
        {
            spec.seed: {arm.name: (arm.lateral, arm.longitudinal) for arm in spec.arms}
            for spec in config.seed_arms
        },
    )


def _direction(summary: dict[str, Any], required: int) -> str:
    if summary["median"] is None:
        return "undefined"
    positive, negative = summary["positive_count"], summary["negative_count"]
    if positive >= required and positive >= negative:
        return "positive"
    if negative >= required and negative > positive:
        return "negative"
    return "mixed"


def _attribution(metrics: dict[str, Any], metric: str, config: DecompositionConfig) -> str:
    effects = metrics[metric]["effects"]
    if _direction(effects["joint"], config.required_scenarios) != config.expected_joint_direction:
        return "not-reproduced-state-dependent"
    joint = abs(effects["joint"]["median"])
    lon = abs(effects["lon"]["median"])
    lat = abs(effects["lat"]["median"])
    if joint <= 0.0:
        return "not-reproduced-state-dependent"
    if lon >= config.dominance_share * joint and lon >= lat:
        return "longitudinal-dominated"
    if lat >= config.dominance_share * joint and lat > lon:
        return "lateral-dominated"
    return "nonlinear-interaction"


def analyze_episodes(
    episodes: list[dict[str, Any]],
    config: DecompositionConfig,
    scenario_names: list[str],
) -> dict[str, Any]:
    result = describe_episodes(episodes, design(config), scenario_names)
    unsafe = result.pop("unsafe_or_incomplete_episodes")
    proxy_errors = result.pop("proxy_errors")
    directions: dict[str, Any] = {}
    attribution: dict[str, Any] = {}
    for seed, metrics in result["metrics"].items():
        directions[seed] = {
            metric: {
                **{
                    arm: _direction(metrics[metric]["effects"][arm], config.required_scenarios)
                    for arm in ARM_NAMES[1:]
                },
                "interaction": _direction(
                    metrics[metric]["interaction"], config.required_scenarios
                ),
            }
            for metric in _ATTRIBUTION_METRICS
        }
        attribution[seed] = {
            metric: _attribution(metrics, metric, config) for metric in _ATTRIBUTION_METRICS
        }
    overall = {}
    for metric in _ATTRIBUTION_METRICS:
        verdicts = {attribution[seed][metric] for seed in attribution}
        overall[metric] = verdicts.pop() if len(verdicts) == 1 else "mixed-across-seeds"
    return {
        **result,
        "verdict": {
            "attribution_metrics": list(_ATTRIBUTION_METRICS),
            "expected_joint_direction": config.expected_joint_direction,
            "dominance_share": config.dominance_share,
            "directions": directions,
            "per_seed": attribution,
            "overall": overall,
            "unsafe_or_incomplete_episodes": unsafe,
            "proxy_errors": proxy_errors,
        },
    }
