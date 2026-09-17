"""Matched execution-horizon statistics; no simulator or model is needed to recompute."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .guidance import InterventionDesign, aggregate, matched_statistics
from .io import read_json

PLANNER_RESPONSE_CHECKPOINTS_S = (0.1, 0.2, 0.5, 1.0, 2.0, 4.0, 8.0)
WINDOW_METRICS = (
    "speed_mps",
    "endpoint_speed_mps",
    "energy_ml_per_km",
    "distance_m",
    "progress_m",
    "position_error_mean_m",
)


@dataclass(frozen=True)
class HorizonDesign:
    longitudinal_actions: list[float]
    noise_seeds: list[int]
    execution_horizons: list[int]
    total_window_steps: int


def window_aggregate(steps: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate one executed window and add planner-to-execution tracking error."""
    summary = aggregate(steps)
    errors = [s["position_error_m"] for s in steps]
    return {
        **summary,
        "position_error_mean_m": float(np.mean(errors)),
        "position_error_max_m": float(np.max(errors)),
    }


def _matched(values: np.ndarray, design: HorizonDesign) -> dict[str, Any]:
    return matched_statistics(
        values,
        InterventionDesign(
            design.longitudinal_actions, design.noise_seeds, design.total_window_steps
        ),
    )


def _first_plan_response(episode: dict[str, Any], checkpoint: float) -> float | None:
    cycles = episode.get("planner_cycles") or []
    if not cycles:
        return None
    expected = [round(value * 10) for value in PLANNER_RESPONSE_CHECKPOINTS_S]
    recorded = cycles[0].get("checkpoint_steps")
    if recorded != expected:
        raise ValueError("planner response checkpoints do not match the analysis contract")
    index = PLANNER_RESPONSE_CHECKPOINTS_S.index(checkpoint)
    return float(cycles[0]["forward_displacement_m"][index])


def analyze_horizon_episodes(
    episodes: list[dict[str, Any]],
    config: HorizonDesign,
    scenario_names: list[str],
) -> dict[str, Any]:
    horizons = list(config.execution_horizons)
    actions = list(config.longitudinal_actions)
    noises = list(config.noise_seeds)
    expected = len(scenario_names) * len(horizons) * len(actions) * len(noises)
    by_key = {
        (e["scenario"], e["execution_horizon"], e["g_lon"], e["noise_seed"]): e for e in episodes
    }
    if len(by_key) != len(episodes):
        raise ValueError("duplicate matched episode identity")
    expected_keys = {
        (scenario, horizon, action, noise)
        for scenario in scenario_names
        for horizon in horizons
        for action in actions
        for noise in noises
    }
    if set(by_key) != expected_keys:
        raise ValueError("episode matrix is incomplete or contains unexpected identities")
    summaries = []
    unsafe = []
    proxy_errors = []
    for e in episodes:
        steps = e["steps"]
        valid = (
            len(steps) == config.total_window_steps
            and e["status"] == "window_complete"
            and all(
                not s["collision"]
                and not s["out_of_road"]
                and not s["terminated"]
                and not s["truncated"]
                for s in steps
            )
        )
        if not valid:
            unsafe.append(e["id"])
        # Independent formula check is diagnostic only; the measured stream owns all totals.
        for i, s in enumerate(steps):
            expected_fuel = 32.5 * np.exp(0.036 * s["speed_mps"]) * s["distance_m"] / 1000
            if not np.isclose(s["fuel_ml"], expected_fuel, rtol=1e-10, atol=1e-12):
                proxy_errors.append({"episode": e["id"], "step": i})
        summaries.append(
            {
                **{
                    k: e[k]
                    for k in (
                        "id",
                        "scenario",
                        "map",
                        "map_seed",
                        "noise_seed",
                        "g_lon",
                        "execution_horizon",
                    )
                },
                "safe_complete": valid,
                "window": window_aggregate(steps) if steps else None,
            }
        )
    indexed = {
        (e["scenario"], e["execution_horizon"], e["g_lon"], e["noise_seed"]): e for e in summaries
    }
    horizon_stats: dict[str, Any] = {}
    for horizon in horizons:
        per_metric = {}
        for metric in WINDOW_METRICS:
            rows = {}
            for scenario in scenario_names:
                values = []
                for action in actions:
                    repeats = []
                    for noise in noises:
                        record = indexed[scenario, horizon, action, noise]["window"]
                        value = None if record is None else record[metric]
                        repeats.append(np.nan if value is None else value)
                    values.append(repeats)
                rows[scenario] = _matched(np.asarray(values), config)
            per_metric[metric] = {"scenarios": rows}
        horizon_stats[str(horizon)] = {"metrics": per_metric}
    planner_response: dict[str, Any] = {}
    for checkpoint in PLANNER_RESPONSE_CHECKPOINTS_S:
        rows = {}
        for scenario in scenario_names:
            values = []
            for action in actions:
                repeats = []
                for noise in noises:
                    reference = _first_plan_response(
                        by_key[scenario, horizons[0], action, noise], checkpoint
                    )
                    for horizon in horizons[1:]:
                        candidate = _first_plan_response(
                            by_key[scenario, horizon, action, noise], checkpoint
                        )
                        if (reference is None) != (candidate is None) or (
                            reference is not None
                            and candidate is not None
                            and not np.isclose(reference, candidate, rtol=0.0, atol=1e-6)
                        ):
                            raise ValueError("first-plan response is not matched across horizons")
                    repeats.append(np.nan if reference is None else reference)
                values.append(repeats)
            rows[scenario] = _matched(np.asarray(values), config)
        planner_response[f"{checkpoint}"] = {"scenarios": rows}
    return {
        "episode_count": len(episodes),
        "expected_episode_count": expected,
        "transition_count": sum(len(e["steps"]) for e in episodes),
        "episodes": summaries,
        "horizons": horizon_stats,
        "planner_response": planner_response,
        "unsafe_or_incomplete_episodes": unsafe,
        "proxy_errors": proxy_errors,
    }


def recompute(source: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Describe saved episodes and attach recorded experimental decisions unchanged."""
    config = read_json(source / "intervention_config.json")
    episodes = read_json(source / "episodes.json")["episodes"]
    scenarios = read_json(source / "scenarios.json")["scenarios"]
    decisions = read_json(source / "decisions.json")
    design = HorizonDesign(
        config["longitudinal_actions"],
        config["noise_seeds"],
        config["execution_horizons"],
        config["total_window_steps"],
    )
    result = analyze_horizon_episodes(episodes, design, [s["name"] for s in scenarios])
    result.pop("unsafe_or_incomplete_episodes")
    result.pop("proxy_errors")
    result["gate_a"] = decisions["gate_a"]
    for horizon, recorded in decisions["horizons"].items():
        horizon_data = result["horizons"][horizon]
        horizon_data["direction"] = recorded["direction"]
        for metric, metrics in horizon_data["metrics"].items():
            source_metric = recorded["metrics"][metric]
            for scenario, row in metrics["scenarios"].items():
                row["passed"] = source_metric["scenarios"][scenario]["passed"]
            for key in ("passed", "positive_pass_count", "negative_pass_count", "direction"):
                metrics[key] = source_metric[key]
    for checkpoint, recorded in decisions["planner_response"].items():
        response = result["planner_response"][checkpoint]
        for scenario, row in response["scenarios"].items():
            row["passed"] = recorded["scenarios"][scenario]["passed"]
        for key in ("passed", "positive_pass_count", "negative_pass_count", "direction"):
            response[key] = recorded[key]
    return result, episodes
