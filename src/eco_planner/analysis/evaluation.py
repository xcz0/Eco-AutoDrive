"""Matched episode comparisons from existing typed evaluation artifacts."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from eco_planner.evaluation.artifacts import JobSummary, load_job_summary
from eco_planner.rl.artifacts import TrainingRunSummary

from .io import read_json
from .statistics import statistics


def episode_records(summary: JobSummary) -> list[dict[str, Any]]:
    rows = []
    for episode in summary.episodes:
        row: dict[str, Any] = {
            "key": [
                episode.scenario.name,
                episode.scenario.map_sequence,
                episode.scenario.seed,
                episode.noise_seed,
                episode.evaluation_mode,
                episode.traffic_density,
            ],
            "status": episode.status,
        }
        if episode.status == "completed":
            row.update(
                energy_ml=episode.metrics.energy.total_ml,
                route_completion=episode.metrics.route_completion,
                collision=episode.metrics.collision,
                out_of_road=episode.metrics.out_of_road,
            )
        else:
            row.update(
                energy_ml=None,
                route_completion=None,
                failure=episode.failure.model_dump(mode="json"),
            )
        rows.append(row)
    return rows


def paired(reference: JobSummary, comparison: JobSummary) -> dict[str, Any]:
    if (
        reference.workload != comparison.workload
        or reference.sampler != comparison.sampler
        or reference.runtime.seed != comparison.runtime.seed
    ):
        raise ValueError("paired evaluation workloads or samplers differ")
    left, right = episode_records(reference), episode_records(comparison)
    lm, rm = {tuple(r["key"]): r for r in left}, {tuple(r["key"]): r for r in right}
    if len(lm) != len(left) or len(rm) != len(right) or lm.keys() != rm.keys():
        raise ValueError("duplicate or missing scenario/map/noise-seed pairs")
    rows = []
    for key, reference_row in lm.items():
        r = rm[key]
        complete = reference_row["status"] == r["status"] == "completed"
        rows.append(
            {
                "key": list(key),
                "reference": reference_row,
                "comparison": r,
                "available": complete,
                **{
                    m + "_delta": r[m] - reference_row[m] if complete else None
                    for m in ("energy_ml", "route_completion")
                },
            }
        )
    valid = [row for row in rows if row["available"]]
    return {
        "difference_direction": "comparison - reference",
        "pairs": rows,
        "pair_count": len(rows),
        "available_pair_count": len(valid),
        "unavailable_pair_count": len(rows) - len(valid),
        "statistics": {
            m: statistics(np.asarray([r[m + "_delta"] for r in valid]), [0.0, 0.25, 0.5, 0.75, 1.0])
            if valid
            else None
            for m in ("energy_ml", "route_completion")
        },
    }


def energy_sweep(source: Path) -> dict[str, Any]:
    matrix = read_json(source / "matrix_summary.json")
    groups: dict[str, dict[str, tuple[dict, JobSummary | None]]] = {}
    for record in matrix["runs"]:
        job, guidance = record["job"], record["guidance"]
        if guidance in groups.setdefault(job, {}):
            raise ValueError(f"duplicate energy job/guidance: {job}/{guidance}")
        summary = (
            None
            if record["status"] == "launcher_failure"
            else load_job_summary(source / job / guidance / "summary.json")
        )
        groups[job][guidance] = record, summary
    comparisons = {}
    for job, runs in groups.items():
        if "baseline" not in runs:
            raise ValueError(f"energy job {job} has no baseline")
        baseline = runs["baseline"][1]
        for guidance, (record, summary) in runs.items():
            if guidance == "baseline":
                continue
            comparisons[f"{job}/{guidance}"] = (
                paired(baseline, summary)
                if baseline is not None and summary is not None
                else {"unavailable": "launcher failure", "run": record}
            )
    return {"runs": matrix["runs"], "comparisons": comparisons}


@dataclass(frozen=True)
class ScalarComparisonRun:
    arm: str
    checkpoint_label: str
    training: TrainingRunSummary
    evaluation: JobSummary


@dataclass(frozen=True)
class ScalarComparison:
    baseline: JobSummary
    runs: tuple[ScalarComparisonRun, ...]


def scalar_reward(comparison: ScalarComparison) -> dict[str, Any]:
    baseline = comparison.baseline
    runs = []
    for run in comparison.runs:
        training, summary = run.training, run.evaluation
        runs.append(
            {
                "arm": run.arm,
                "training_seed": training.training_seed,
                "checkpoint_label": run.checkpoint_label,
                "comparison": paired(baseline, summary),
                "training_curve": {
                    "update": [u.update_index for u in training.updates],
                    "reward": [u.total_reward for u in training.updates],
                },
            }
        )
    aggregates = {}
    for arm, label in sorted({(r["arm"], r["checkpoint_label"]) for r in runs}):
        items = [r for r in runs if r["arm"] == arm and r["checkpoint_label"] == label]
        aggregates[f"{arm}/{label}"] = {
            "training_seeds": [r["training_seed"] for r in items],
            "aggregation_unit": "training_seed_mean_of_available_matched_episodes",
            "metrics": {
                m: statistics(
                    np.asarray([r["comparison"]["statistics"][m]["mean"] for r in items]),
                    [0.0, 0.5, 1.0],
                )
                if all(r["comparison"]["statistics"][m] is not None for r in items)
                else None
                for m in ("energy_ml", "route_completion")
            },
        }
    return {
        "runs": runs,
        "seed_aggregates": aggregates,
        "interpretation": "Descriptive matched comparisons; update0 is diagnostic only.",
    }
