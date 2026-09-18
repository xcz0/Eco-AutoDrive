"""Configuration and predeclared verdict for the frozen-policy execution bridge."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, model_validator

from eco_planner.analysis.execution_bridge import (
    analyze_bridge_episodes as describe_episodes,
)
from eco_planner.analysis.execution_bridge import analyze_same_state

ArmLabel = Literal["r0", "rstress"]
_ARM_REWARD_PROFILES: dict[str, str] = {
    "r0": "plannerrft_no_energy_calibrated_v1",
    "rstress": "plannerrft_energy_band_lam64_v1",
}
_REQUIRED_HORIZONS = (1, 5)


class BridgeRun(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid", allow_inf_nan=False)
    label: str = Field(pattern=r"^[a-z0-9_-]+$")
    arm: ArmLabel
    training_seed: StrictInt = Field(ge=0)
    path: str
    reward_profile: str


class ExecutionBridgeConfig(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid", allow_inf_nan=False)
    protocol: str
    job: str
    runtime_seed: StrictInt = Field(ge=0)
    noise_seeds: list[StrictInt]
    execution_horizons: list[StrictInt]
    reference_arm: Literal["r0"]
    counterfactual_arm: Literal["rstress"]
    runs: list[BridgeRun]
    same_state_contract_steps: StrictInt = Field(ge=1)
    include_same_state_audit: bool
    required_scenarios: StrictInt = Field(gt=0)
    part_b_majority_fraction: StrictFloat = Field(gt=0, le=1)
    overrides: list[str]

    @model_validator(mode="after")
    def validate_design(self) -> ExecutionBridgeConfig:
        if not self.noise_seeds or len(set(self.noise_seeds)) != len(self.noise_seeds):
            raise ValueError("noise seeds must be non-empty and distinct")
        if min(self.noise_seeds) < 0:
            raise ValueError("noise seeds must be non-negative")
        if len(self.execution_horizons) < 2 or len(set(self.execution_horizons)) != len(
            self.execution_horizons
        ):
            raise ValueError("execution horizons must be at least two distinct prefixes")
        if any(horizon < 1 for horizon in self.execution_horizons):
            raise ValueError("execution horizons must be positive")
        if not set(_REQUIRED_HORIZONS).issubset(self.execution_horizons):
            raise ValueError("execution horizons must include the 0.1 s and 0.5 s contracts")
        labels = [run.label for run in self.runs]
        if len(labels) != len(set(labels)):
            raise ValueError("run labels must be unique")
        keys = [(run.arm, run.training_seed) for run in self.runs]
        if len(keys) != len(set(keys)):
            raise ValueError("run arm/seed pairs must be unique")
        for run in self.runs:
            if run.reward_profile != _ARM_REWARD_PROFILES[run.arm]:
                raise ValueError("run reward profile does not match its arm")
        seeds = sorted({run.training_seed for run in self.runs})
        for seed in seeds:
            if {run.arm for run in self.runs if run.training_seed == seed} != {"r0", "rstress"}:
                raise ValueError("each training seed requires exactly one r0 and one rstress run")
        return self


def arm_runs(config: ExecutionBridgeConfig, training_seed: int) -> tuple[BridgeRun, BridgeRun]:
    """Return ``(reference, counterfactual)`` runs for one training seed."""
    by_arm = {run.arm: run for run in config.runs if run.training_seed == training_seed}
    return by_arm[config.reference_arm], by_arm[config.counterfactual_arm]


def _direction(summary: dict[str, Any], required: int) -> str:
    if summary["median"] is None:
        return "undefined"
    positive, negative = summary["positive_count"], summary["negative_count"]
    if positive >= required and positive >= negative:
        return "positive"
    if negative >= required and negative > positive:
        return "negative"
    return "mixed"


def _part_a_status(
    metrics: dict[str, Any],
    config: ExecutionBridgeConfig,
    first_horizon: int,
    last_horizon: int,
) -> tuple[str, dict[str, Any]]:
    reference = config.reference_arm
    counterfactual = config.counterfactual_arm
    directions: dict[str, Any] = {}
    detail: dict[str, Any] = {}
    for seed in metrics:
        seed_speed: dict[str, str] = {}
        seed_energy: dict[str, str] = {}
        for horizon in map(str, config.execution_horizons):
            speed = metrics[seed][horizon]["speed_mps"]["effects"][counterfactual]
            energy = metrics[seed][horizon]["energy_ml_per_km"]["effects"][counterfactual]
            seed_speed[horizon] = _direction(speed, config.required_scenarios)
            seed_energy[horizon] = _direction(energy, config.required_scenarios)
        directions[seed] = {"speed_mps": seed_speed, "energy_ml_per_km": seed_energy}
        detail[seed] = {
            "speed_median": {
                horizon: metrics[seed][horizon]["speed_mps"]["effects"][counterfactual]["median"]
                for horizon in seed_speed
            },
            "reference_arm": reference,
            "counterfactual_arm": counterfactual,
        }
    first, last = str(first_horizon), str(last_horizon)

    def crosses(seed: str) -> bool:
        return (
            directions[seed]["speed_mps"][first] == "negative"
            and directions[seed]["speed_mps"][last] == "positive"
        )

    def amplifies(seed: str) -> bool:
        first_dir = directions[seed]["speed_mps"][first]
        if first_dir != directions[seed]["speed_mps"][last] or first_dir not in {
            "positive",
            "negative",
        }:
            return False
        first_median = detail[seed]["speed_median"][first]
        last_median = detail[seed]["speed_median"][last]
        return (
            first_median is not None
            and last_median is not None
            and abs(last_median) >= abs(first_median)
        )

    seeds = list(directions)
    if seeds and all(crosses(seed) for seed in seeds):
        status = "crossover_confirmed"
    elif seeds and all(amplifies(seed) for seed in seeds):
        status = "amplifies_not_reverses"
    else:
        status = "no_material_effect"
    return status, {"status": status, "directions": directions, "detail": detail}


def _part_b_status(
    same_state: dict[str, Any] | None, config: ExecutionBridgeConfig
) -> dict[str, Any]:
    if same_state is None:
        return {"status": "not_run", "seeds": {}}
    checks: dict[str, Any] = {}
    for seed, row in same_state["seeds"].items():
        threshold = config.part_b_majority_fraction * row["context_count"]
        delta_lon_positive = row["delta_g_lon"]["positive_count"] >= threshold
        first = row["delta_forward"]["0.1"]["median"]
        middle = (
            row["delta_forward"]["0.2"]["median"] > 0.0
            and row["delta_forward"]["0.5"]["median"] > 0.0
        )
        checks[seed] = {
            "delta_g_lon_positive_majority": bool(delta_lon_positive),
            "first_waypoint_effect_median_m": first,
            "middle_effect_positive": bool(middle),
        }
    bridge = bool(checks) and all(
        row["delta_g_lon_positive_majority"]
        and row["first_waypoint_effect_median_m"] is not None
        and row["first_waypoint_effect_median_m"] <= 0.0
        and row["middle_effect_positive"]
        for row in checks.values()
    )
    differs = bool(checks) and all(
        row["delta_g_lon_positive_majority"]
        and row["first_waypoint_effect_median_m"] is not None
        and row["first_waypoint_effect_median_m"] > 0.0
        for row in checks.values()
    )
    if bridge:
        status = "local_temporal_bridge"
    elif differs:
        status = "local_response_differs"
    else:
        status = "mixed"
    return {"status": status, "seeds": checks}


def analyze_episodes(
    episodes: list[dict[str, Any]],
    same_state_contexts: list[dict[str, Any]],
    config: ExecutionBridgeConfig,
    scenario_names: list[str],
) -> dict[str, Any]:
    result = describe_episodes(
        episodes,
        execution_horizons=list(config.execution_horizons),
        noise_seeds=list(config.noise_seeds),
        reference_arm=config.reference_arm,
        counterfactual_arm=config.counterfactual_arm,
        scenario_names=scenario_names,
    )
    unsafe = result.pop("unsafe_or_incomplete_episodes")
    proxy_errors = result.pop("proxy_errors")
    same_state = analyze_same_state(same_state_contexts) if same_state_contexts else None
    first_horizon, last_horizon = min(_REQUIRED_HORIZONS), max(_REQUIRED_HORIZONS)
    part_a, part_a_detail = _part_a_status(result["metrics"], config, first_horizon, last_horizon)
    part_b = _part_b_status(same_state, config)
    safety_clean = not unsafe and not proxy_errors
    if part_a == "crossover_confirmed":
        verdict = "execution_contract_causal_crossover_confirmed"
    elif part_a == "amplifies_not_reverses":
        verdict = "execution_contract_amplifies_but_does_not_reverse"
    elif part_b["status"] == "local_response_differs":
        verdict = "learned_policy_local_response_differs_from_manual_full_range_sweep"
    else:
        verdict = "no_material_execution_contract_effect_proceed_to_training_budget"
    return {
        **result,
        "same_state": same_state,
        "gate": {
            "verdict": verdict,
            "part_a": part_a_detail,
            "part_b": part_b,
            "first_horizon": first_horizon,
            "last_horizon": last_horizon,
            "required_scenarios": config.required_scenarios,
            "safety_clean": bool(safety_clean),
            "unsafe_or_incomplete_episodes": unsafe,
            "proxy_errors": proxy_errors,
        },
    }
