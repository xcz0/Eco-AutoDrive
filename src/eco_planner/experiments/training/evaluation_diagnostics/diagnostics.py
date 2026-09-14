"""Offline deterministic-vs-stochastic and Beta-distribution diagnostics."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from eco_planner.experiments.training.effective_update.diagnostics import beta_statistics
from eco_planner.experiments.training.objective_positive_control.diagnostics import (
    heldout_values,
    paired_heldout_deltas,
)

ARMS = ("r0", "rstress")
POLICIES = ("initial", "r0", "rstress")
COMPARISON_METRICS = ("mean_speed_mps", "energy_ml_per_km", "route_completion")
_BETA_STATISTICS = ("beta_mean", "concentration", "variance")
_BETA_AGGREGATES = ("mean", "min", "max")


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


def stochastic_metric_summary(
    values_by_action_seed: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    """Per-metric values, mean, and range across the fixed policy-action seeds."""

    if not values_by_action_seed:
        raise ValueError("stochastic summary requires at least one policy-action seed")
    first = next(iter(values_by_action_seed.values()))
    metrics = sorted(first)
    for values in values_by_action_seed.values():
        if sorted(values) != metrics:
            raise ValueError("stochastic values must share one metric set")
    summary: dict[str, Any] = {}
    for metric in metrics:
        values = {
            str(action_seed): values_by_action_seed[action_seed][metric]
            for action_seed in values_by_action_seed
        }
        if any(value is None for value in values.values()):
            summary[metric] = {
                "by_action_seed": values,
                "mean": None,
                "min": None,
                "max": None,
            }
            continue
        numeric = list(values.values())
        summary[metric] = {
            "by_action_seed": values,
            "mean": sum(numeric) / len(numeric),
            "min": min(numeric),
            "max": max(numeric),
        }
    return summary


def stochastic_mean(values_by_action_seed: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    """Mean held-out values across the fixed policy-action seeds."""

    if not values_by_action_seed:
        raise ValueError("stochastic mean requires at least one policy-action seed")
    first = next(iter(values_by_action_seed.values()))
    metrics = sorted(first)
    for values in values_by_action_seed.values():
        if sorted(values) != metrics:
            raise ValueError("stochastic values must share one metric set")
    return {
        metric: (
            None
            if any(values[metric] is None for values in values_by_action_seed.values())
            else sum(values[metric] for values in values_by_action_seed.values())
            / len(values_by_action_seed)
        )
        for metric in metrics
    }


def deterministic_vs_stochastic_shift(
    deterministic: Mapping[str, Any], stochastic: Mapping[str, Any]
) -> dict[str, Any]:
    """Per-metric stochastic-minus-deterministic shift and relative effect."""

    shift: dict[str, Any] = {}
    for metric in COMPARISON_METRICS:
        deterministic_value = deterministic[metric]
        stochastic_value = stochastic[metric]
        if deterministic_value is None or stochastic_value is None or deterministic_value == 0.0:
            shift[metric] = {"delta": None, "relative": None}
            continue
        delta = stochastic_value - deterministic_value
        shift[metric] = {
            "delta": delta,
            "relative": abs(delta) / abs(deterministic_value),
        }
    return shift


def build_diagnostics_summary(
    records: Mapping[int, Mapping[str, Mapping[str, Any]]],
    beta_records: Mapping[int, Mapping[str, Mapping[str, Any]]],
    action_seeds: list[int],
) -> dict[str, Any]:
    """Assemble the Task H comparison from per-seed evaluation and probe records.

    ``records`` maps each training seed to ``{policy: {"deterministic": ...,
    "stochastic": {action_seed: ...}}}`` aggregated held-out values, and
    ``beta_records`` maps each training seed to ``{arm: {"probe_before": ...,
    "probe_after": ...}}`` fixed-context probe summaries. The result is a
    descriptive diagnostic artifact; it is not a gate verdict.
    """

    policies = {
        str(seed): {
            policy: {
                "checkpoint": records[seed][policy]["checkpoint"],
                "deterministic": records[seed][policy]["deterministic"],
                "stochastic": stochastic_metric_summary(records[seed][policy]["stochastic"]),
            }
            for policy in POLICIES
        }
        for seed in sorted(records)
    }
    shifts = {
        str(seed): {
            policy: deterministic_vs_stochastic_shift(
                records[seed][policy]["deterministic"],
                stochastic_mean(records[seed][policy]["stochastic"]),
            )
            for policy in POLICIES
        }
        for seed in sorted(records)
    }
    paired_deterministic = {
        str(seed): paired_heldout_deltas(
            records[seed]["r0"]["deterministic"],
            records[seed]["rstress"]["deterministic"],
            COMPARISON_METRICS,
        )
        for seed in sorted(records)
    }
    paired_stochastic_mean = {
        str(seed): paired_heldout_deltas(
            stochastic_mean(records[seed]["r0"]["stochastic"]),
            stochastic_mean(records[seed]["rstress"]["stochastic"]),
            COMPARISON_METRICS,
        )
        for seed in sorted(records)
    }
    paired_stochastic_by_action_seed = {
        str(seed): {
            str(action_seed): paired_heldout_deltas(
                records[seed]["r0"]["stochastic"][action_seed],
                records[seed]["rstress"]["stochastic"][action_seed],
                COMPARISON_METRICS,
            )
            for action_seed in action_seeds
        }
        for seed in sorted(records)
    }
    direction_matches = {
        str(seed): {
            metric: _same_direction(
                paired_deterministic[str(seed)][metric]["delta"],
                paired_stochastic_mean[str(seed)][metric]["delta"],
            )
            for metric in COMPARISON_METRICS
        }
        for seed in sorted(records)
    }
    beta_distribution = {
        str(seed): {
            **{
                arm: {
                    "initial": beta_probe_statistics(beta_records[seed][arm]["probe_before"]),
                    "final": beta_probe_statistics(beta_records[seed][arm]["probe_after"]),
                    "final_minus_initial": paired_beta_deltas(
                        beta_records[seed][arm]["probe_before"],
                        beta_records[seed][arm]["probe_after"],
                    ),
                }
                for arm in ARMS
            },
            "paired_rstress_minus_r0": paired_beta_deltas(
                beta_records[seed]["r0"]["probe_after"],
                beta_records[seed]["rstress"]["probe_after"],
            ),
        }
        for seed in sorted(beta_records)
    }
    return {
        "status": "completed",
        "policy_action_seeds": list(action_seeds),
        "training_seeds": sorted(records),
        "policies": policies,
        "stochastic_vs_deterministic": shifts,
        "paired_rstress_minus_r0": {
            "deterministic": paired_deterministic,
            "stochastic_mean": paired_stochastic_mean,
            "stochastic_by_action_seed": paired_stochastic_by_action_seed,
            "direction_matches_deterministic": direction_matches,
        },
        "beta_distribution": beta_distribution,
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


def _same_direction(left: float | None, right: float | None) -> bool | None:
    if left is None or right is None or left == 0.0 or right == 0.0:
        return None
    return left * right > 0.0


__all__ = [
    "ARMS",
    "COMPARISON_METRICS",
    "POLICIES",
    "beta_probe_statistics",
    "build_diagnostics_summary",
    "deterministic_vs_stochastic_shift",
    "heldout_values",
    "paired_beta_deltas",
    "stochastic_mean",
    "stochastic_metric_summary",
]
