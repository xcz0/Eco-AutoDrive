"""Bootstrap statistics for validated evaluation episodes."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import numpy as np

from eco_planner.evaluation.artifacts.models import EpisodeSummary

BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 0


def build_matrix_statistics(episodes: Sequence[EpisodeSummary]) -> dict[str, Any]:
    """Build per-scenario statistics from already validated episode summaries."""

    grouped: defaultdict[tuple[str, float], list[EpisodeSummary]] = defaultdict(list)
    for episode in episodes:
        if episode.status == "completed":
            grouped[(episode.scenario.name, episode.traffic_density)].append(episode)
    statistics: dict[str, Any] = {}
    for (scenario, density), group in sorted(grouped.items()):
        ml_per_km = [episode.energy.ml_per_km for episode in group]
        if any(value is None for value in ml_per_km):
            raise ValueError(
                "matrix cannot bootstrap energy_ml_per_km for a completed zero-distance episode"
            )
        statistics[f"{scenario}/density_{density:.2f}"] = {
            "episode_count": len(group),
            "metrics": {
                "simulated_seconds": bootstrap(
                    np.asarray([episode.simulated_seconds for episode in group], dtype=np.float64)
                ),
                "distance_m": bootstrap(
                    np.asarray([episode.distance_m for episode in group], dtype=np.float64)
                ),
                "energy_total_ml": bootstrap(
                    np.asarray([episode.energy.total_ml for episode in group], dtype=np.float64)
                ),
                "energy_ml_per_km": bootstrap(np.asarray(ml_per_km, dtype=np.float64)),
                "route_completion": bootstrap(
                    np.asarray([episode.route_completion for episode in group], dtype=np.float64)
                ),
                "mean_speed_mps": bootstrap(
                    np.asarray([episode.speed_mps.mean for episode in group], dtype=np.float64)
                ),
                "total_reward": bootstrap(
                    np.asarray([episode.total_reward for episode in group], dtype=np.float64)
                ),
            },
            "arrive_rate": float(np.mean([episode.arrive_dest for episode in group])),
            "collision_rate": float(
                np.mean(
                    [
                        episode.crash_vehicle
                        or episode.crash_object
                        or episode.crash_building
                        or episode.crash_human
                        for episode in group
                    ]
                )
            ),
            "out_of_road_rate": float(np.mean([episode.out_of_road for episode in group])),
        }
    return statistics


def bootstrap(values: np.ndarray) -> dict[str, float | list[float]]:
    """Return fixed-seed mean, median, and bootstrap interval for one metric."""

    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError("bootstrap values must be a non-empty finite vector")
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    indices = generator.integers(0, values.size, size=(BOOTSTRAP_SAMPLES, values.size))
    means = values[indices].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "mean_bootstrap_95_percentile_interval": [
            float(np.percentile(means, 2.5)),
            float(np.percentile(means, 97.5)),
        ],
    }
