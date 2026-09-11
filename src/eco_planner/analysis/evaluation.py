"""Matched episode comparisons from existing typed evaluation artifacts."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from eco_planner.evaluation.artifacts import JobSummary, load_job_summary
from eco_planner.rl.artifacts import TrainingRunSummary

from .io import read_json
from .statistics import ScenarioBootstrapConfig, scenario_effect, statistics


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
                wrong_direction=episode.metrics.wrong_direction,
                arrive_dest=episode.metrics.arrive_dest,
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
    for key in sorted(lm):
        reference_row = lm[key]
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
        "available_rate": len(valid) / len(rows) if rows else None,
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
    bootstrap: ScenarioBootstrapConfig
    training_seeds: tuple[int, ...]


def arm_outcomes(summary: JobSummary) -> dict:
    rows = episode_records(summary)
    valid = [row for row in rows if row["status"] == "completed"]
    count, available = len(rows), len(valid)
    return {
        "completion": {
            "episode_count": count,
            "completed_count": available,
            "failed_count": count - available,
            "completed_rate": available / count if count else None,
            "arrive_dest_count": sum(row["arrive_dest"] for row in valid),
            "arrive_dest_rate": sum(row["arrive_dest"] for row in valid) / available
            if available
            else None,
            "arrival_denominator": available,
            "route_completion_mean": float(np.mean([row["route_completion"] for row in valid]))
            if available
            else None,
        },
        "safety": {
            "denominator": available,
            "unavailable_count": count - available,
            **{
                metric: {
                    "count": sum(row[metric] for row in valid),
                    "rate": sum(row[metric] for row in valid) / available if available else None,
                }
                for metric in ("collision", "out_of_road", "wrong_direction")
            },
        },
        "failures": [row for row in rows if row["status"] != "completed"],
    }


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
                "outcomes": arm_outcomes(summary),
                "comparison": paired(baseline, summary),
                "training_curve": {
                    "update": [u.update_index for u in training.updates],
                    "reward": [u.total_reward for u in training.updates],
                },
            }
        )
    indexed = {
        (r.arm, r.training.training_seed, r.checkpoint_label): r.evaluation for r in comparison.runs
    }
    contrasts = {}
    for name, reference_arm, comparison_arm in (
        ("a2-a1", "a1", "a2"),
        ("a1-a0", "a0", "a1"),
        ("a2-a0", "a0", "a2"),
    ):
        checkpoints = {}
        for label in sorted({"final", *(r.checkpoint_label for r in comparison.runs)}):
            effects = []
            for seed in comparison.training_seeds:
                reference = (
                    baseline if reference_arm == "a0" else indexed.get((reference_arm, seed, label))
                )
                target = indexed.get((comparison_arm, seed, label))
                pairs = (
                    paired(reference, target)
                    if reference is not None and target is not None
                    else None
                )
                delta = (
                    np.asarray([r["energy_ml_delta"] for r in pairs["pairs"] if r["available"]])
                    if pairs is not None
                    else np.asarray([])
                )
                effect = scenario_effect(delta, comparison.bootstrap)
                if pairs is None:
                    effect["unavailable_reason"] = "missing evaluation for this arm/seed/checkpoint"
                effects.append({"training_seed": seed, **effect, "comparison": pairs})
            entry: dict[str, Any] = {"effects": effects}
            if label == "final":
                estimates = [e["estimate"] for e in effects if e["estimate"] is not None]
                entry["direction_counts"] = {
                    "lower": sum(e < 0 for e in estimates),
                    "zero": sum(e == 0 for e in estimates),
                    "higher": sum(e > 0 for e in estimates),
                    "unavailable": len(effects) - len(estimates),
                    "total": len(effects),
                }
                entry["partial"] = len(estimates) != len(effects)
            checkpoints[label] = entry
        contrasts[name] = checkpoints
    return {
        "baseline": arm_outcomes(baseline),
        "runs": runs,
        "contrasts": contrasts,
        "bootstrap": {
            **comparison.bootstrap.model_dump(),
            "method": "percentile",
            "unit": "matched_scenario_delta",
        },
        "interpretation": (
            "Completed means normally ended with metrics, including collision/out-of-road."
            " Energy is comparison - reference on jointly-completed matched episodes; negative is"
            " lower MetaDrive fuel proxy (mL). Interpret energy after completion, arrival/progress"
            " and safety. Scenario bootstrap CIs condition on each trained policy and available"
            " scenarios; they are "
            "not uncertainty across training seeds. Initial/update0 is diagnostic only."
        ),
    }
