"""Predeclared matched statistics; no simulator or model is needed for recomputation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from scipy.stats import spearmanr


@dataclass(frozen=True)
class InterventionDesign:
    longitudinal_actions: list[float]
    noise_seeds: list[int]
    window_steps: int


def aggregate(steps: list[dict[str, Any]]) -> dict[str, Any]:
    if not steps:
        raise ValueError("cannot aggregate an empty execution window")
    distance = sum(s["distance_m"] for s in steps)
    fuel = sum(s["fuel_ml"] for s in steps)
    return {
        "speed_mps": float(np.mean([s["speed_mps"] for s in steps])),
        "endpoint_speed_mps": steps[-1]["speed_mps"],
        "energy_ml_per_km": fuel * 1000 / distance if distance > 0 else None,
        "fuel_ml": fuel,
        "distance_m": distance,
        "energy_score": float(np.mean([s["energy_score"] for s in steps])),
        "progress_m": sum(s["progress_m"] for s in steps),
        "duration_s": sum(s["dt_s"] for s in steps),
        "planner_first_speed_mps": float(np.mean([s["planner_first_speed_mps"] for s in steps])),
        "planner_mean_speed_mps": float(np.mean([s["planner_mean_speed_mps"] for s in steps])),
        "reference_first_speed_mps": float(
            np.mean([s["reference_first_speed_mps"] for s in steps])
        ),
        "target_first_delta_mps": float(np.mean([s["target_first_delta_mps"] for s in steps])),
    }


def matched_statistics(values: np.ndarray, config: InterventionDesign) -> dict[str, Any]:
    """Rows are ordered guidance arms; columns are paired noise repeats."""
    if values.shape != (len(config.longitudinal_actions), len(config.noise_seeds)):
        raise ValueError("matched matrix does not match the declared arms and repeats")
    if not np.isfinite(values).all():
        return {"reason": "undefined_metric", "rho": None, "effect": None}
    means = values.mean(axis=1)
    rho = (
        None
        if np.ptp(means) == 0
        else float(cast(Any, spearmanr(config.longitudinal_actions, means))[0])
    )
    differences = values[-1] - values[0]
    effect = float(differences.mean())
    noise = float(np.sqrt(values.var(axis=1, ddof=1).mean()))
    return {
        "values": values.tolist(),
        "arm_means": means.tolist(),
        "rho": rho,
        "effect": effect,
        "paired_endpoint_differences": differences.tolist(),
        "noise_scale": noise,
    }


def analyze_episodes(
    episodes: list[dict[str, Any]],
    config: InterventionDesign,
    scenario_names: list[str],
) -> dict[str, Any]:
    expected = len(scenario_names) * len(config.noise_seeds) * len(config.longitudinal_actions)
    by_key = {(e["scenario"], e["g_lon"], e["noise_seed"]): e for e in episodes}
    if len(by_key) != len(episodes):
        raise ValueError("duplicate matched episode identity")
    expected_keys = {
        (s, a, n)
        for s in scenario_names
        for a in config.longitudinal_actions
        for n in config.noise_seeds
    }
    if set(by_key) != expected_keys:
        raise ValueError("episode matrix is incomplete or contains unexpected identities")
    summaries = []
    unsafe = []
    proxy_errors = []
    for e in episodes:
        steps = e["steps"]
        valid = (
            len(steps) == config.window_steps
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
                **{k: e[k] for k in ("id", "scenario", "map", "map_seed", "noise_seed", "g_lon")},
                "safe_complete": valid,
                "immediate": aggregate(steps[:1]) if steps else None,
                "short_horizon": aggregate(steps) if steps else None,
            }
        )
    indexed = {(e["scenario"], e["g_lon"], e["noise_seed"]): e for e in summaries}
    windows: dict[str, Any] = {}
    for window in ("immediate", "short_horizon"):
        metrics = {}
        for metric in (
            "speed_mps",
            "energy_ml_per_km",
            "planner_first_speed_mps",
            "planner_mean_speed_mps",
            "target_first_delta_mps",
        ):
            rows = {}
            for scenario in scenario_names:
                values = []
                for action in config.longitudinal_actions:
                    repeats = []
                    for noise in config.noise_seeds:
                        record = indexed[scenario, action, noise][window]
                        value = None if record is None else record[metric]
                        repeats.append(np.nan if value is None else value)
                    values.append(repeats)
                rows[scenario] = matched_statistics(np.asarray(values), config)
            metrics[metric] = {"scenarios": rows}
        windows[window] = metrics
    return {
        "episode_count": len(episodes),
        "expected_episode_count": expected,
        "transition_count": sum(len(e["steps"]) for e in episodes),
        "episodes": summaries,
        "windows": windows,
        "unsafe_or_incomplete_episodes": unsafe,
        "proxy_errors": proxy_errors,
    }
