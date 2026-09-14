from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

_BETA_STATISTICS = ("beta_mean", "concentration", "variance")
_BETA_AGGREGATES = ("mean", "min", "max")


def beta_statistics(alpha: float, beta: float) -> tuple[float, float, float]:
    """Affine-Beta mean, concentration, and variance from one (alpha, beta) pair."""

    concentration = alpha + beta
    mean = 2.0 * alpha / concentration - 1.0
    variance = 4.0 * alpha * beta / (concentration * concentration * (concentration + 1.0))
    return mean, concentration, variance


def beta_probe_statistics(probe: Mapping[str, Any]) -> dict[str, Any]:
    """Aggregate one fixed-context probe into per-dimension Beta statistics.

    The probe carries per-context ``alpha``/``beta`` pairs for the two guidance
    dimensions; each pair is reduced to the affine-Beta mean, concentration,
    and variance, then aggregated over contexts per dimension.
    """

    alpha_rows = _probe_rows(probe, "alpha")
    beta_rows = _probe_rows(probe, "beta")
    statistics: dict[str, list[list[float]]] = {name: [] for name in _BETA_STATISTICS}
    for alpha_row, beta_row in zip(alpha_rows, beta_rows, strict=True):
        per_dimension = list(
            zip(
                *[
                    beta_statistics(alpha, beta)
                    for alpha, beta in zip(alpha_row, beta_row, strict=True)
                ],
                strict=True,
            )
        )
        for name, values in zip(_BETA_STATISTICS, per_dimension, strict=True):
            statistics[name].append([float(value) for value in values])
    aggregated: dict[str, dict[str, list[float]]] = {}
    for name in _BETA_STATISTICS:
        aggregated[name] = {
            aggregate: [
                _aggregate([row[dimension] for row in statistics[name]], aggregate)
                for dimension in range(len(statistics[name][0]))
            ]
            for aggregate in _BETA_AGGREGATES
        }
    return {"context_count": len(alpha_rows), **aggregated}


def paired_beta_deltas(reference: Mapping[str, Any], stress: Mapping[str, Any]) -> dict[str, Any]:
    """Stress-minus-reference per-context Beta-stat deltas aggregated per dimension."""

    reference_alpha = _probe_rows(reference, "alpha")
    reference_beta = _probe_rows(reference, "beta")
    stress_alpha = _probe_rows(stress, "alpha")
    stress_beta = _probe_rows(stress, "beta")
    deltas: dict[str, list[list[float]]] = {name: [] for name in _BETA_STATISTICS}
    for reference_alpha_row, reference_beta_row, stress_alpha_row, stress_beta_row in zip(
        reference_alpha, reference_beta, stress_alpha, stress_beta, strict=True
    ):
        reference_statistics = [
            beta_statistics(alpha, beta)
            for alpha, beta in zip(reference_alpha_row, reference_beta_row, strict=True)
        ]
        stress_statistics = [
            beta_statistics(alpha, beta)
            for alpha, beta in zip(stress_alpha_row, stress_beta_row, strict=True)
        ]
        for index, name in enumerate(_BETA_STATISTICS):
            deltas[name].append(
                [
                    stress_statistics[dimension][index] - reference_statistics[dimension][index]
                    for dimension in range(len(reference_statistics))
                ]
            )
    return {
        name: {
            "mean_delta_per_dimension": [
                sum(row[dimension] for row in deltas[name]) / len(deltas[name])
                for dimension in range(len(deltas[name][0]))
            ],
            "rms": math.sqrt(
                sum(value * value for row in deltas[name] for value in row)
                / (len(deltas[name]) * len(deltas[name][0]))
            ),
        }
        for name in _BETA_STATISTICS
    }


def _probe_rows(probe: Mapping[str, Any], field: str) -> list[list[float]]:
    rows = probe[field]
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"probe summary has no {field} values")
    return [[float(value) for value in row] for row in rows]


def _aggregate(values: list[float], name: str) -> float:
    if name == "mean":
        return sum(values) / len(values)
    if name == "min":
        return min(values)
    return max(values)


def heldout_metric_values(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate one held-out evaluation summary's per-episode metrics."""

    if not episodes:
        raise ValueError("held-out evaluation requires at least one episode")
    speeds = [episode["metrics"]["speed_mps"]["mean"] for episode in episodes]
    distances = [episode["metrics"]["distance_m"] for episode in episodes]
    completions = [episode["metrics"]["route_completion"] for episode in episodes]
    energy_totals = [episode["metrics"]["energy"]["total_ml"] for episode in episodes]
    energy_intensities = [
        episode["metrics"]["energy"]["ml_per_km"]
        for episode in episodes
        if episode["metrics"]["energy"]["ml_per_km"] is not None
    ]
    return {
        "mean_speed_mps": sum(speeds) / len(speeds),
        "distance_m": sum(distances) / len(distances),
        "route_completion": sum(completions) / len(completions),
        "energy_total_ml": sum(energy_totals) / len(energy_totals),
        "energy_ml_per_km": (
            sum(energy_intensities) / len(energy_intensities) if energy_intensities else None
        ),
        "arrive_dest_fraction": sum(1 for episode in episodes if episode["metrics"]["arrive_dest"])
        / len(episodes),
        "collision_count": sum(1 for episode in episodes if episode["metrics"]["collision"]),
        "out_of_road_count": sum(1 for episode in episodes if episode["metrics"]["out_of_road"]),
    }
