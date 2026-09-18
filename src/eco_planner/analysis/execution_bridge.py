"""Frozen-policy execution-contract bridge statistics.

Pure offline recomputation from persisted episodes and same-state contexts; no
simulator or model is needed. Part A compares the matched ``Rstress - R0``
closed-loop effect across execution prefixes; Part B compares the same-state
planner response to the two frozen policies' local guidance actions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .decomposition import episode_metrics
from .horizon import PLANNER_RESPONSE_CHECKPOINTS_S
from .io import read_json

PART_A_METRICS = (
    "speed_mps",
    "energy_ml_per_km",
    "route_completion",
    "distance_m",
    "progress_m",
    "stopped_fraction",
    "step_count",
    "prefix_speed_mps",
)
_SAFETY_FLAGS = ("collision", "out_of_road", "arrive_dest", "max_step", "terminated", "truncated")
_CHECKPOINT_INDICES = [round(value * 10) - 1 for value in PLANNER_RESPONSE_CHECKPOINTS_S]


def _summarize(effects: dict[str, float]) -> dict[str, Any]:
    values = np.asarray(list(effects.values()), dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        return {
            "median": None,
            "positive_count": 0,
            "negative_count": 0,
            "scenarios": effects,
        }
    return {
        "median": float(np.median(values)),
        "positive_count": int(np.sum(values > 0.0)),
        "negative_count": int(np.sum(values < 0.0)),
        "scenarios": effects,
    }


def _first_plan_response(episode: dict[str, Any], checkpoint: float) -> float | None:
    cycles = episode.get("planner_cycles") or []
    if not cycles:
        return None
    expected = [round(value * 10) for value in PLANNER_RESPONSE_CHECKPOINTS_S]
    if cycles[0].get("checkpoint_steps") != expected:
        raise ValueError("planner response checkpoints do not match the analysis contract")
    return float(
        cycles[0]["forward_displacement_m"][PLANNER_RESPONSE_CHECKPOINTS_S.index(checkpoint)]
    )


def analyze_bridge_episodes(
    episodes: list[dict[str, Any]],
    *,
    execution_horizons: list[int],
    noise_seeds: list[int],
    reference_arm: str,
    counterfactual_arm: str,
    scenario_names: list[str],
) -> dict[str, Any]:
    """Describe matched frozen-policy closed-loop effects across execution prefixes."""
    horizons = list(execution_horizons)
    noises = list(noise_seeds)
    arms = (reference_arm, counterfactual_arm)
    seeds = sorted({int(e["training_seed"]) for e in episodes})
    expected = len(scenario_names) * len(horizons) * len(arms) * len(noises) * len(seeds)
    by_key = {
        (e["training_seed"], e["scenario"], e["arm"], e["execution_horizon"], e["noise_seed"]): e
        for e in episodes
    }
    if len(by_key) != len(episodes):
        raise ValueError("duplicate matched episode identity")
    expected_keys = {
        (seed, scenario, arm, horizon, noise)
        for seed in seeds
        for scenario in scenario_names
        for arm in arms
        for horizon in horizons
        for noise in noises
    }
    if set(by_key) != expected_keys:
        raise ValueError("episode matrix is incomplete or contains unexpected identities")

    summaries = []
    unsafe: list[str] = []
    proxy_errors: list[dict[str, Any]] = []
    for episode in episodes:
        metrics = episode_metrics(episode)
        if metrics is None or metrics["collision"] or metrics["out_of_road"]:
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
                        "execution_horizon",
                    )
                },
                "status": episode["status"],
                "metrics": metrics,
            }
        )
    indexed = {
        (
            row["training_seed"],
            row["scenario"],
            row["arm"],
            row["execution_horizon"],
            row["noise_seed"],
        ): row
        for row in summaries
    }
    metrics_out: dict[str, Any] = {}
    safety: dict[str, Any] = {}
    for seed in seeds:
        seeds_metrics: dict[str, Any] = {}
        seeds_safety: dict[str, Any] = {}
        for horizon in horizons:
            per_metric: dict[str, Any] = {}
            for metric in PART_A_METRICS:
                arm_values: dict[str, dict[str, float]] = {arm: {} for arm in arms}
                for scenario in scenario_names:
                    for arm in arms:
                        repeats = []
                        for noise in noises:
                            record = indexed[seed, scenario, arm, horizon, noise]["metrics"]
                            value = None if record is None else record[metric]
                            repeats.append(np.nan if value is None else value)
                        arm_values[arm][scenario] = float(np.mean(repeats))
                effects: dict[str, float] = {
                    scenario: arm_values[counterfactual_arm][scenario]
                    - arm_values[reference_arm][scenario]
                    for scenario in scenario_names
                }
                per_metric[metric] = {
                    "arm_values": arm_values,
                    "effects": {counterfactual_arm: _summarize(effects)},
                }
            seeds_metrics[str(horizon)] = per_metric
            seeds_safety[str(horizon)] = {
                arm: {
                    flag: int(
                        sum(
                            bool(indexed[seed, scenario, arm, horizon, noise]["metrics"][flag])
                            for scenario in scenario_names
                            for noise in noises
                            if indexed[seed, scenario, arm, horizon, noise]["metrics"] is not None
                        )
                    )
                    for flag in _SAFETY_FLAGS
                }
                for arm in arms
            }
        metrics_out[str(seed)] = seeds_metrics
        safety[str(seed)] = seeds_safety

    planner_response: dict[str, Any] = {}
    for checkpoint in PLANNER_RESPONSE_CHECKPOINTS_S:
        per_seed: dict[str, Any] = {}
        for seed in seeds:
            effects: dict[str, float] = {}
            for scenario in scenario_names:
                values: dict[str, float] = {}
                for arm in arms:
                    reference = _first_plan_response(
                        by_key[seed, scenario, arm, horizons[0], noises[0]], checkpoint
                    )
                    for horizon in horizons[1:]:
                        for noise in noises:
                            candidate = _first_plan_response(
                                by_key[seed, scenario, arm, horizon, noise], checkpoint
                            )
                            if (reference is None) != (candidate is None) or (
                                reference is not None
                                and candidate is not None
                                and not np.isclose(reference, candidate, rtol=0.0, atol=1e-6)
                            ):
                                raise ValueError(
                                    "first-plan policy response is not matched across "
                                    "execution prefixes"
                                )
                    repeats = [
                        _first_plan_response(
                            by_key[seed, scenario, arm, horizons[0], noise], checkpoint
                        )
                        for noise in noises
                    ]
                    values[arm] = float(
                        np.mean([np.nan if value is None else value for value in repeats])
                    )
                effects[scenario] = values[counterfactual_arm] - values[reference_arm]
            per_seed[str(seed)] = {counterfactual_arm: _summarize(effects)}
        planner_response[str(checkpoint)] = per_seed
    return {
        "episode_count": len(episodes),
        "expected_episode_count": expected,
        "transition_count": sum(len(e["steps"]) for e in episodes),
        "episodes": summaries,
        "metrics": metrics_out,
        "safety": safety,
        "planner_response": planner_response,
        "unsafe_or_incomplete_episodes": unsafe,
        "proxy_errors": proxy_errors,
    }


def zero_crossing_step(effect: np.ndarray) -> int | None:
    """First zero-based waypoint index with a strictly positive matched effect."""
    positive = np.flatnonzero(effect > 0.0)
    return int(positive[0]) if positive.size else None


def analyze_same_state(contexts: list[dict[str, Any]]) -> dict[str, Any]:
    """Describe same-state local policy -> planner temporal responses."""
    if not contexts:
        raise ValueError("same-state audit requires at least one context")
    seeds = sorted({int(context["training_seed"]) for context in contexts})
    per_seed: dict[str, Any] = {}
    for seed in seeds:
        rows = [context for context in contexts if int(context["training_seed"]) == seed]
        delta_lat = np.asarray(
            [c["g_lat_counterfactual"] - c["g_lat_reference"] for c in rows], dtype=np.float64
        )
        delta_lon = np.asarray(
            [c["g_lon_counterfactual"] - c["g_lon_reference"] for c in rows], dtype=np.float64
        )
        forward_delta = np.asarray(
            [
                np.asarray(c["forward_counterfactual_m"], dtype=np.float64)
                - np.asarray(c["forward_reference_m"], dtype=np.float64)
                for c in rows
            ]
        )
        lateral_delta = np.asarray(
            [
                np.asarray(c["lateral_counterfactual_m"], dtype=np.float64)
                - np.asarray(c["lateral_reference_m"], dtype=np.float64)
                for c in rows
            ]
        )
        crossings = [zero_crossing_step(forward_delta[index]) for index in range(len(rows))]
        defined = [value for value in crossings if value is not None]
        per_seed[str(seed)] = {
            "context_count": len(rows),
            "delta_g_lon": _summarize({str(i): float(value) for i, value in enumerate(delta_lon)}),
            "delta_g_lat": _summarize({str(i): float(value) for i, value in enumerate(delta_lat)}),
            "delta_forward": {
                str(checkpoint): {
                    "median": float(np.median(forward_delta[:, index])),
                    "positive_count": int(np.sum(forward_delta[:, index] > 0.0)),
                    "negative_count": int(np.sum(forward_delta[:, index] < 0.0)),
                }
                for checkpoint, index in zip(
                    PLANNER_RESPONSE_CHECKPOINTS_S, _CHECKPOINT_INDICES, strict=True
                )
            },
            "delta_lateral": {
                str(checkpoint): float(np.median(lateral_delta[:, position]))
                for position, checkpoint in enumerate(PLANNER_RESPONSE_CHECKPOINTS_S)
            },
            "crossing_median_step": float(np.median(defined)) if defined else None,
            "crossing_defined_count": len(defined),
        }
    return {"context_count": len(contexts), "seeds": per_seed}


def recompute(source: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Describe saved artifacts and attach the recorded verdict unchanged."""
    config = read_json(source / "intervention_config.json")
    episodes = read_json(source / "episodes.json")["episodes"]
    scenarios = read_json(source / "scenarios.json")["scenarios"]
    decisions = read_json(source / "decisions.json")
    same_state_path = source / "same_state.json"
    contexts = read_json(same_state_path)["contexts"] if same_state_path.is_file() else []
    result = analyze_bridge_episodes(
        episodes,
        execution_horizons=config["execution_horizons"],
        noise_seeds=config["noise_seeds"],
        reference_arm=config["reference_arm"],
        counterfactual_arm=config["counterfactual_arm"],
        scenario_names=[s["name"] for s in scenarios],
    )
    result.pop("unsafe_or_incomplete_episodes")
    result.pop("proxy_errors")
    result["same_state"] = analyze_same_state(contexts) if contexts else None
    result["gate"] = decisions["gate"]
    return result, episodes
