"""Offline pairing analysis and Gate G evaluation for the Task G study."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from eco_planner.experiments.training.effective_update.diagnostics import (
    heldout_metric_values as _effective_update_heldout_metric_values,
)

from .config import GateGConfig

ProbeField = list[list[float]]


def heldout_values(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate one held-out evaluation with the Task G collapse indicators.

    Extends the Task F aggregation with the per-episode ``stopped_fraction``
    and ``simulated_seconds`` means used by Gate G condition 5 to rule out
    parking / episode-length collapse as the explanation of a separation.
    """

    if not episodes:
        raise ValueError("held-out evaluation requires at least one episode")
    values = dict(_effective_update_heldout_metric_values(episodes))
    values["stopped_fraction"] = sum(
        episode["metrics"]["stopped_fraction"] for episode in episodes
    ) / len(episodes)
    values["simulated_seconds"] = sum(
        episode["metrics"]["simulated_seconds"] for episode in episodes
    ) / len(episodes)
    return values


def load_probe_field(run_dir: Path, probe: str, field: str) -> ProbeField:
    """Read one probe field (16 contexts x 2 dims) from a training summary."""

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    values = summary[probe][field]
    if not isinstance(values, list) or not values:
        raise ValueError(f"run {run_dir} has no {probe}.{field} probe values")
    return [[float(value) for value in row] for row in values]


def probe_rms(values: list[float]) -> float:
    """Root-mean-square of a flattened probe field or delta."""

    if not values:
        raise ValueError("probe rms requires at least one value")
    return math.sqrt(sum(value * value for value in values) / len(values))


def probe_cosine(left: list[float], right: list[float]) -> float:
    """Cosine similarity between two flattened probe deltas."""

    if len(left) != len(right) or not left:
        raise ValueError("probe cosine requires two non-empty equal-length vectors")
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    norm_left = math.sqrt(sum(a * a for a in left))
    norm_right = math.sqrt(sum(b * b for b in right))
    if norm_left == 0.0 or norm_right == 0.0:
        return 0.0
    return dot / (norm_left * norm_right)


def probe_sign_agreement(left: list[float], right: list[float]) -> float:
    """Fraction of elements where two probe deltas share a strict sign."""

    if len(left) != len(right) or not left:
        raise ValueError("sign agreement requires two non-empty equal-length vectors")
    agreeing = sum(1 for a, b in zip(left, right, strict=True) if a * b > 0.0)
    return agreeing / len(left)


def flatten_probe_delta(r0: ProbeField, rstress: ProbeField) -> list[float]:
    """Element-wise rstress-minus-r0 delta of one probe field, flattened."""

    if len(r0) != len(rstress) or not r0:
        raise ValueError("paired probe fields must share one non-empty context set")
    return [
        rstress[context][dimension] - r0[context][dimension]
        for context in range(len(r0))
        for dimension in range(len(r0[0]))
    ]


def paired_heldout_deltas(
    r0: dict[str, Any], rstress: dict[str, Any], metrics: Sequence[str]
) -> dict[str, dict[str, Any]]:
    """Per-metric rstress-minus-r0 paired deltas and relative effects."""

    paired: dict[str, dict[str, Any]] = {}
    for name in metrics:
        r0_value = r0[name]
        rstress_value = rstress[name]
        if r0_value is None or rstress_value is None or r0_value == 0.0:
            paired[name] = {"delta": None, "relative": None}
            continue
        delta = rstress_value - r0_value
        paired[name] = {
            "delta": delta,
            "relative": abs(delta) / abs(r0_value),
        }
    return paired


def evaluate_gate_g(
    records: dict[int, dict[str, dict[str, Any]]], gate: GateGConfig
) -> dict[str, Any]:
    """Evaluate the five Gate G conditions over the paired per-seed records.

    ``records`` maps each training seed to ``{"r0": ..., "rstress": ...}`` run
    records carrying the extracted training metrics, the probe fields, and the
    aggregated held-out values. Conditions 2 and 3 are operationally the same
    judgement under the two-seed design: a matched separation is only counted
    when the paired relative effect exceeds the evaluation-noise floor on every
    seed and the delta direction reproduces across seeds, so both keys share
    that judgement input.
    """

    seeds = sorted(records)
    if len(seeds) < 2:
        raise ValueError("Gate G requires at least two training seeds")
    probe_deltas = {
        seed: flatten_probe_delta(
            records[seed]["r0"]["probe_after_guidance_mean"],
            records[seed]["rstress"]["probe_after_guidance_mean"],
        )
        for seed in seeds
    }
    probe_rms_by_seed = {seed: probe_rms(delta) for seed, delta in probe_deltas.items()}
    probe_cosines = [
        probe_cosine(probe_deltas[seeds[index]], probe_deltas[seeds[index + 1]])
        for index in range(len(seeds) - 1)
    ]
    probe_sign_agreements = [
        probe_sign_agreement(probe_deltas[seeds[index]], probe_deltas[seeds[index + 1]])
        for index in range(len(seeds) - 1)
    ]
    c1 = all(value >= gate.guidance_rms_floor for value in probe_rms_by_seed.values()) and all(
        value > 0.0 for value in probe_cosines
    )

    heldout_paired = {
        seed: paired_heldout_deltas(
            records[seed]["r0"]["heldout"],
            records[seed]["rstress"]["heldout"],
            gate.separation_metrics,
        )
        for seed in seeds
    }
    separated_metrics: dict[str, dict[str, Any]] = {}
    for metric in gate.separation_metrics:
        deltas = [heldout_paired[seed][metric]["delta"] for seed in seeds]
        relatives = [heldout_paired[seed][metric]["relative"] for seed in seeds]
        beyond_noise = all(
            relative is not None and relative >= gate.paired_relative_effect_floor
            for relative in relatives
        )
        same_direction = all(delta is not None and delta * deltas[0] > 0.0 for delta in deltas[1:])
        separated_metrics[metric] = {
            "deltas": deltas,
            "relatives": relatives,
            "beyond_noise": beyond_noise,
            "same_direction": same_direction,
            "separated": beyond_noise and same_direction,
        }
    c2 = any(entry["separated"] for entry in separated_metrics.values())
    c3 = c2

    energy_deltas = [
        records[seed]["rstress"]["heldout"]["energy_ml_per_km"]
        - records[seed]["r0"]["heldout"]["energy_ml_per_km"]
        for seed in seeds
    ]
    speed_deltas = [
        records[seed]["rstress"]["heldout"]["mean_speed_mps"]
        - records[seed]["r0"]["heldout"]["mean_speed_mps"]
        for seed in seeds
    ]
    c4 = all(delta < 0.0 for delta in energy_deltas) or all(delta < 0.0 for delta in speed_deltas)

    collapse_checks: dict[int, dict[str, Any]] = {}
    for seed in seeds:
        r0_heldout = records[seed]["r0"]["heldout"]
        rstress_heldout = records[seed]["rstress"]["heldout"]
        collapse_checks[seed] = {
            "collision_budget": (
                rstress_heldout["collision_count"] + rstress_heldout["out_of_road_count"]
                <= gate.collision_budget
            ),
            "arrive_dest_retained": (
                rstress_heldout["arrive_dest_fraction"]
                >= r0_heldout["arrive_dest_fraction"] - gate.arrive_dest_drop_ceiling
            ),
            "no_parking": (
                rstress_heldout["stopped_fraction"]
                <= r0_heldout["stopped_fraction"] + gate.stopped_fraction_increase_ceiling
            ),
            "route_completion_retained": (
                rstress_heldout["route_completion"]
                >= r0_heldout["route_completion"] - gate.route_completion_drop_ceiling
            ),
        }
    c5 = all(all(checks.values()) for checks in collapse_checks.values())

    conditions = {
        "c1_guidance_distribution_separation": c1,
        "c2_closed_loop_separation": c2,
        "c3_paired_effect_reproduced": c3,
        "c4_direction_matches_objective": c4,
        "c5_no_collapse_explanation": c5,
    }
    return {
        "conditions": conditions,
        "passed": all(conditions.values()),
        "failure_reasons": [name for name, passed in conditions.items() if not passed],
        "probe_paired": {
            "rms_by_seed": probe_rms_by_seed,
            "adjacent_seed_cosines": probe_cosines,
            "adjacent_seed_sign_agreements": probe_sign_agreements,
        },
        "heldout_paired": heldout_paired,
        "separated_metrics": separated_metrics,
        "objective_direction": {
            "energy_ml_per_km_deltas": energy_deltas,
            "mean_speed_mps_deltas": speed_deltas,
        },
        "collapse_checks": collapse_checks,
    }
