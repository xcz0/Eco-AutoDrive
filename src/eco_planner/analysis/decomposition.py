"""Matched lon/lat guidance-arm decomposition; no simulator or model is needed to recompute."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from eco_planner.envs.domain import STOPPED_SPEED_THRESHOLD_MPS

from .horizon import PLANNER_RESPONSE_CHECKPOINTS_S
from .io import read_json

ARM_NAMES = ("r0", "lon", "lat", "joint")
FULL_METRICS = (
    "speed_mps",
    "energy_ml_per_km",
    "distance_m",
    "progress_m",
    "route_completion",
    "stopped_fraction",
)
PREFIX_METRICS = ("prefix_speed_mps",)
FIRST_WAYPOINT_METRICS = ("first_waypoint_distance_m", "first_waypoint_speed_mps")
PREFIX_STEPS = 20


@dataclass(frozen=True)
class DecompositionDesign:
    noise_seeds: list[int]
    arms: dict[int, dict[str, tuple[float, float]]]

    @property
    def seeds(self) -> list[int]:
        return sorted(self.arms)


def episode_metrics(episode: dict[str, Any]) -> dict[str, Any] | None:
    """Aggregate one variable-length episode; returns None when no step executed."""
    steps = episode["steps"]
    if not steps:
        return None
    distance = float(sum(step["distance_m"] for step in steps))
    fuel = float(sum(step["fuel_ml"] for step in steps))
    speeds = np.asarray([step["speed_mps"] for step in steps], dtype=np.float64)
    prefix = speeds[:PREFIX_STEPS]
    return {
        "step_count": len(steps),
        "duration_s": float(sum(step["dt_s"] for step in steps)),
        "speed_mps": float(speeds.mean()),
        "energy_ml_per_km": fuel * 1000 / distance if distance > 0.0 else None,
        "distance_m": distance,
        "progress_m": float(sum(step["progress_m"] for step in steps)),
        "route_completion": float(steps[-1]["route_completion"]),
        "stopped_fraction": float(np.mean(speeds < STOPPED_SPEED_THRESHOLD_MPS)),
        "prefix_speed_mps": float(prefix.mean()),
        "first_waypoint_distance_m": float(steps[0]["distance_m"]),
        "first_waypoint_speed_mps": float(steps[0]["speed_mps"]),
        "collision": any(step["collision"] for step in steps),
        "out_of_road": any(step["out_of_road"] for step in steps),
        "arrive_dest": bool(steps[-1]["arrive_dest"]),
        "max_step": bool(steps[-1]["max_step"]),
        "terminated": any(step["terminated"] for step in steps),
        "truncated": any(step["truncated"] for step in steps),
    }


def first_plan_response(episode: dict[str, Any], checkpoint: float) -> float | None:
    cycles = episode.get("planner_cycles") or []
    if not cycles:
        return None
    expected = [round(value * 10) for value in PLANNER_RESPONSE_CHECKPOINTS_S]
    if cycles[0].get("checkpoint_steps") != expected:
        raise ValueError("planner response checkpoints do not match the analysis contract")
    index = PLANNER_RESPONSE_CHECKPOINTS_S.index(checkpoint)
    return float(cycles[0]["forward_displacement_m"][index])


def _summarize(effects: dict[str, float]) -> dict[str, Any]:
    values = np.asarray(list(effects.values()), dtype=np.float64)
    if not np.isfinite(values).all():
        return {
            "median": None,
            "positive_count": 0,
            "negative_count": 0,
            "scenarios": effects,
        }
    return {
        "median": float(np.median(values)),
        "positive_count": int(np.sum(values > 0)),
        "negative_count": int(np.sum(values < 0)),
        "scenarios": effects,
    }


def _arm_scenario_values(
    metric: str,
    seed: int,
    design: DecompositionDesign,
    indexed: dict[tuple[Any, ...], dict[str, Any]],
    scenario_names: list[str],
) -> dict[str, dict[str, float]]:
    arm_values = {arm: {} for arm in ARM_NAMES}
    for scenario in scenario_names:
        for arm in ARM_NAMES:
            repeats = []
            for noise in design.noise_seeds:
                record = indexed[seed, scenario, arm, noise]["metrics"]
                value = None if record is None else record[metric]
                repeats.append(np.nan if value is None else value)
            arm_values[arm][scenario] = float(np.mean(repeats))
    return arm_values


def _effects(arm_values: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    return {
        arm: {
            scenario: arm_values[arm][scenario] - arm_values["r0"][scenario]
            for scenario in arm_values["r0"]
        }
        for arm in ARM_NAMES[1:]
    }


def analyze_decomposition_episodes(
    episodes: list[dict[str, Any]],
    design: DecompositionDesign,
    scenario_names: list[str],
) -> dict[str, Any]:
    seeds = design.seeds
    expected = len(scenario_names) * len(design.noise_seeds) * len(ARM_NAMES) * len(seeds)
    by_key = {(e["training_seed"], e["scenario"], e["arm"], e["noise_seed"]): e for e in episodes}
    if len(by_key) != len(episodes):
        raise ValueError("duplicate matched episode identity")
    expected_keys = {
        (seed, scenario, arm, noise)
        for seed in seeds
        for scenario in scenario_names
        for arm in ARM_NAMES
        for noise in design.noise_seeds
    }
    if set(by_key) != expected_keys:
        raise ValueError("episode matrix is incomplete or contains unexpected identities")
    summaries = []
    unsafe = []
    proxy_errors = []
    for episode in episodes:
        metrics = episode_metrics(episode)
        if metrics is not None and (metrics["collision"] or metrics["out_of_road"]):
            unsafe.append(episode["id"])
        for index, step in enumerate(episode["steps"]):
            expected_fuel = 32.5 * np.exp(0.036 * step["speed_mps"]) * step["distance_m"] / 1000
            if not np.isclose(step["fuel_ml"], expected_fuel, rtol=1e-10, atol=1e-12):
                proxy_errors.append({"episode": episode["id"], "step": index})
        summaries.append(
            {
                **{
                    key: episode[key]
                    for key in (
                        "id",
                        "training_seed",
                        "scenario",
                        "map",
                        "map_seed",
                        "arm",
                        "noise_seed",
                        "g_lat",
                        "g_lon",
                    )
                },
                "status": episode["status"],
                "metrics": metrics,
            }
        )
    indexed = {
        (row["training_seed"], row["scenario"], row["arm"], row["noise_seed"]): row
        for row in summaries
    }
    metric_names = (*FULL_METRICS, *PREFIX_METRICS, *FIRST_WAYPOINT_METRICS)
    metrics_out: dict[str, Any] = {}
    for seed in seeds:
        per_metric: dict[str, Any] = {}
        for metric in metric_names:
            arm_values = _arm_scenario_values(metric, seed, design, indexed, scenario_names)
            effects = _effects(arm_values)
            interaction = {
                scenario: effects["joint"][scenario]
                - effects["lon"][scenario]
                - effects["lat"][scenario]
                for scenario in scenario_names
            }
            per_metric[metric] = {
                "arm_values": arm_values,
                "arm_medians": {
                    arm: float(np.median(list(arm_values[arm].values()))) for arm in ARM_NAMES
                },
                "effects": {arm: _summarize(effects[arm]) for arm in ARM_NAMES[1:]},
                "interaction": _summarize(interaction),
            }
        metrics_out[str(seed)] = per_metric
    planner_response: dict[str, Any] = {}
    for checkpoint in PLANNER_RESPONSE_CHECKPOINTS_S:
        per_checkpoint: dict[str, Any] = {}
        for seed in seeds:
            effects = {arm: {} for arm in ARM_NAMES[1:]}
            for scenario in scenario_names:
                values = {}
                for arm in ARM_NAMES:
                    repeats = []
                    for noise in design.noise_seeds:
                        episode = by_key[seed, scenario, arm, noise]
                        response = first_plan_response(episode, checkpoint)
                        repeats.append(np.nan if response is None else response)
                    values[arm] = float(np.mean(repeats))
                for arm in ARM_NAMES[1:]:
                    effects[arm][scenario] = values[arm] - values["r0"]
            per_checkpoint[str(seed)] = {arm: _summarize(effects[arm]) for arm in ARM_NAMES[1:]}
        planner_response[str(checkpoint)] = per_checkpoint
    return {
        "episode_count": len(episodes),
        "expected_episode_count": expected,
        "transition_count": sum(len(e["steps"]) for e in episodes),
        "episodes": summaries,
        "metrics": metrics_out,
        "planner_response": planner_response,
        "unsafe_or_incomplete_episodes": unsafe,
        "proxy_errors": proxy_errors,
    }


def recompute(source: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Describe saved episodes and attach the recorded verdict unchanged."""
    config = read_json(source / "intervention_config.json")
    episodes = read_json(source / "episodes.json")["episodes"]
    scenarios = read_json(source / "scenarios.json")["scenarios"]
    decisions = read_json(source / "decisions.json")
    design = DecompositionDesign(
        config["noise_seeds"],
        {
            spec["seed"]: {
                arm["name"]: (arm["lateral"], arm["longitudinal"]) for arm in spec["arms"]
            }
            for spec in config["seed_arms"]
        },
    )
    result = analyze_decomposition_episodes(episodes, design, [s["name"] for s in scenarios])
    result.pop("unsafe_or_incomplete_episodes")
    result.pop("proxy_errors")
    result["verdict"] = decisions["verdict"]
    return result, episodes
