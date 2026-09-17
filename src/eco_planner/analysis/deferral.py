"""Matched replanning-deferral statistics; no simulator or model is needed to recompute."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from eco_planner.contracts import PLANNER_HORIZON

from .io import read_json


@dataclass(frozen=True)
class DeferralDesign:
    longitudinal_actions: list[float]
    noise_seeds: list[int]
    total_window_steps: int
    execution_steps: int


def waypoint_response(episode: dict[str, Any]) -> np.ndarray | None:
    """Return the per-cycle full-horizon forward displacement, shape ``(cycles, horizon)``."""
    cycles = episode.get("planner_cycles") or []
    if not cycles:
        return None
    rows = [cycle.get("waypoint_forward_displacement_m") for cycle in cycles]
    if any(row is None for row in rows):
        raise ValueError("deferral requires per-waypoint planner response")
    if {len(row) for row in rows} != {PLANNER_HORIZON}:
        raise ValueError("planner waypoint response does not match PLANNER_HORIZON")
    return np.asarray(rows, dtype=np.float64)


def zero_crossing_step(effect: np.ndarray) -> int | None:
    """First zero-based waypoint index with a strictly positive matched effect."""
    positive = np.flatnonzero(effect > 0.0)
    return int(positive[0]) if positive.size else None


def _matched_curves(
    indexed: dict[tuple[Any, ...], np.ndarray],
    scenario: str,
    actions: list[float],
    noises: list[int],
) -> np.ndarray:
    """Median-over-repeats matched endpoint difference, shape ``(cycles, horizon)``."""
    plus = indexed[scenario, actions[-1], noises[0]]
    minus = indexed[scenario, actions[0], noises[0]]
    cycles = min(plus.shape[0], minus.shape[0])
    for action in actions:
        for noise in noises:
            cycles = min(cycles, indexed[scenario, action, noise].shape[0])
    curves = np.empty((cycles, PLANNER_HORIZON), dtype=np.float64)
    for cycle in range(cycles):
        repeats = np.asarray(
            [
                indexed[scenario, actions[-1], noise][cycle]
                - indexed[scenario, actions[0], noise][cycle]
                for noise in noises
            ],
            dtype=np.float64,
        )
        curves[cycle] = np.median(repeats, axis=0)
    return curves


def _scenario_summary(
    curves: np.ndarray, majority_fraction: float, tolerance: int
) -> dict[str, Any]:
    cycles = curves.shape[0]
    rows = []
    for cycle in range(cycles):
        crossing = zero_crossing_step(curves[cycle])
        rows.append(
            {
                "cycle": cycle,
                "first_waypoint_effect_m": float(curves[cycle, 0]),
                "full_horizon_effect_m": float(curves[cycle, -1]),
                "crossing_step": crossing,
            }
        )
    first = np.asarray([row["first_waypoint_effect_m"] for row in rows])
    full = np.asarray([row["full_horizon_effect_m"] for row in rows])
    crossings = [row["crossing_step"] for row in rows]
    defined = [value for value in crossings if value is not None]
    threshold = majority_fraction * cycles
    deferred = sum(
        row["first_waypoint_effect_m"] <= 0.0
        and row["full_horizon_effect_m"] > 0.0
        and (row["crossing_step"] is None or row["crossing_step"] >= 1)
        for row in rows
    )
    first_negative = int(np.sum(first < 0.0))
    first_positive = int(np.sum(first > 0.0))
    full_positive = int(np.sum(full > 0.0))
    defined_majority = len(defined) >= threshold
    spread = int(max(defined) - min(defined)) if len(defined) >= 2 else None
    stable = bool(defined_majority and spread is not None and spread <= tolerance)
    repeated = bool(
        first_negative >= threshold
        and full_positive >= threshold
        and deferred >= threshold
        and stable
    )
    return {
        "cycle_count": cycles,
        "cycles": rows,
        "median_first_waypoint_effect_m": float(np.median(first)),
        "median_full_horizon_effect_m": float(np.median(full)),
        "first_negative_count": first_negative,
        "first_positive_count": first_positive,
        "first_zero_count": int(np.sum(first == 0.0)),
        "full_positive_count": full_positive,
        "full_negative_count": int(np.sum(full < 0.0)),
        "deferred_count": deferred,
        "crossing_defined_count": len(defined),
        "crossing_steps": defined,
        "crossing_median_step": float(np.median(defined)) if defined else None,
        "crossing_spread_steps": spread,
        "first_negative_majority": bool(first_negative >= threshold),
        "full_positive_majority": bool(full_positive >= threshold),
        "deferred_majority": bool(deferred >= threshold),
        "crossing_stable": stable,
        "repeated_deferral": repeated,
    }


def analyze_deferral_episodes(
    episodes: list[dict[str, Any]],
    design: DeferralDesign,
    scenario_names: list[str],
    *,
    majority_fraction: float,
    crossing_tolerance_steps: int,
) -> dict[str, Any]:
    actions = list(design.longitudinal_actions)
    noises = list(design.noise_seeds)
    expected = len(scenario_names) * len(actions) * len(noises)
    by_key = {(e["scenario"], e["g_lon"], e["noise_seed"]): e for e in episodes}
    if len(by_key) != len(episodes):
        raise ValueError("duplicate matched episode identity")
    expected_keys = {
        (scenario, action, noise)
        for scenario in scenario_names
        for action in actions
        for noise in noises
    }
    if set(by_key) != expected_keys:
        raise ValueError("episode matrix is incomplete or contains unexpected identities")
    summaries = []
    unsafe = []
    proxy_errors = []
    indexed: dict[tuple[Any, ...], np.ndarray] = {}
    for episode in episodes:
        steps = episode["steps"]
        valid = (
            len(steps) == design.total_window_steps
            and design.execution_steps == 1
            and episode["status"] == "window_complete"
            and all(
                not step["collision"]
                and not step["out_of_road"]
                and not step["terminated"]
                and not step["truncated"]
                for step in steps
            )
        )
        if not valid:
            unsafe.append(episode["id"])
        for index, step in enumerate(steps):
            expected_fuel = 32.5 * np.exp(0.036 * step["speed_mps"]) * step["distance_m"] / 1000
            if not np.isclose(step["fuel_ml"], expected_fuel, rtol=1e-10, atol=1e-12):
                proxy_errors.append({"episode": episode["id"], "step": index})
        response = waypoint_response(episode)
        if response is not None:
            indexed[episode["scenario"], episode["g_lon"], episode["noise_seed"]] = response
        summaries.append(
            {
                **{
                    key: episode[key]
                    for key in (
                        "id",
                        "scenario",
                        "map",
                        "map_seed",
                        "noise_seed",
                        "g_lon",
                        "execution_horizon",
                    )
                },
                "status": episode["status"],
                "cycle_count": 0 if response is None else int(response.shape[0]),
            }
        )
    scenarios: dict[str, Any] = {}
    aggregate_curves = []
    for scenario in scenario_names:
        curves = _matched_curves(indexed, scenario, actions, noises)
        scenarios[scenario] = _scenario_summary(curves, majority_fraction, crossing_tolerance_steps)
        aggregate_curves.append(curves)
    cycle_count = min((curves.shape[0] for curves in aggregate_curves), default=0)
    median_curves = (
        np.median(np.asarray([curves[:cycle_count] for curves in aggregate_curves]), axis=0)
        if aggregate_curves and cycle_count
        else np.empty((0, PLANNER_HORIZON))
    )
    crossings_by_cycle = [zero_crossing_step(median_curves[cycle]) for cycle in range(cycle_count)]
    return {
        "episode_count": len(episodes),
        "expected_episode_count": expected,
        "transition_count": sum(len(e["steps"]) for e in episodes),
        "episodes": summaries,
        "scenarios": scenarios,
        "aggregate": {
            "cycle_count": cycle_count,
            "median_effect_by_cycle": median_curves.tolist(),
            "first_waypoint_effect_by_cycle": median_curves[:, 0].tolist(),
            "full_horizon_effect_by_cycle": median_curves[:, -1].tolist(),
            "crossing_step_by_cycle": crossings_by_cycle,
        },
        "unsafe_or_incomplete_episodes": unsafe,
        "proxy_errors": proxy_errors,
    }


def recompute(source: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Describe saved episodes and attach the recorded deferral gate unchanged."""
    config = read_json(source / "intervention_config.json")
    episodes = read_json(source / "episodes.json")["episodes"]
    scenarios = read_json(source / "scenarios.json")["scenarios"]
    decisions = read_json(source / "decisions.json")
    design = DeferralDesign(
        config["longitudinal_actions"],
        config["noise_seeds"],
        config["total_window_steps"],
        config["execution_steps"],
    )
    result = analyze_deferral_episodes(
        episodes,
        design,
        [s["name"] for s in scenarios],
        majority_fraction=config["majority_fraction"],
        crossing_tolerance_steps=config["crossing_tolerance_steps"],
    )
    result.pop("unsafe_or_incomplete_episodes")
    result.pop("proxy_errors")
    result["gate_c"] = decisions["gate_c"]
    return result, episodes
