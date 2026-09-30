"""Matched episode comparisons from existing typed evaluation artifacts."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from eco_planner.evaluation.artifacts import JobSummary, load_job_summary
from eco_planner.rl.artifacts import TrainingRunSummary

from .io import read_json
from .statistics import (
    DistributionStatistics,
    ScenarioBootstrapConfig,
    scenario_effect,
    statistics,
)

_PAIRED_METRICS = (
    "energy_ml",
    "route_completion",
    "mean_speed_mps",
    "distance_m",
    "stopped_fraction",
    "energy_ml_per_km",
    "energy_distance_m",
)


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
                mean_speed_mps=episode.metrics.speed_mps.mean,
                speed_min_mps=episode.metrics.speed_mps.minimum,
                speed_max_mps=episode.metrics.speed_mps.maximum,
                distance_m=episode.metrics.distance_m,
                stopped_fraction=episode.metrics.stopped_fraction,
                energy_ml_per_km=episode.metrics.energy.ml_per_km,
                energy_distance_m=episode.metrics.energy.distance_m,
                terminated=episode.terminated,
                truncated=episode.truncated,
                terminal_reason=episode.terminal_reason,
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
                    m + "_delta": r[m] - reference_row[m]
                    if complete and r[m] is not None and reference_row[m] is not None
                    else None
                    for m in _PAIRED_METRICS
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
        "statistics": {m: _delta_statistics(valid, m) for m in _PAIRED_METRICS},
    }


def _delta_statistics(valid: list[dict[str, Any]], metric: str) -> DistributionStatistics | None:
    values = [row[metric + "_delta"] for row in valid if row[metric + "_delta"] is not None]
    if not values:
        return None
    return statistics(np.asarray(values), [0.0, 0.25, 0.5, 0.75, 1.0])


def paired_metric_effects(
    pairs: dict[str, Any] | None, config: ScenarioBootstrapConfig
) -> dict[str, Any] | None:
    """Per-metric paired scenario effects from one ``paired`` result."""

    if pairs is None:
        return None
    return {
        metric: scenario_effect(
            np.asarray(
                [
                    row[metric + "_delta"]
                    for row in pairs["pairs"]
                    if row["available"] and row[metric + "_delta"] is not None
                ]
            ),
            config,
        )
        for metric in _PAIRED_METRICS
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
class PolicyComparisonRun:
    arm: str
    checkpoint_label: str
    training: TrainingRunSummary
    evaluation: JobSummary


@dataclass(frozen=True)
class PolicyComparison:
    baseline: JobSummary | None
    runs: tuple[PolicyComparisonRun, ...]
    bootstrap: ScenarioBootstrapConfig
    training_seeds: tuple[int, ...]
    contrasts: tuple[tuple[str, str], ...]
    baseline_arm: str | None
    source_directories: tuple[Path, ...] = ()


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
        "behavior": _behavior_outcomes(valid),
        "energy": _energy_outcomes(valid),
        "termination": _termination_outcomes(valid),
        "failures": [row for row in rows if row["status"] != "completed"],
    }


def _behavior_outcomes(valid: list[dict[str, Any]]) -> dict[str, Any]:
    def mean(values: list[float]) -> float | None:
        return sum(values) / len(values) if values else None

    return {
        "mean_speed_mps": mean([row["mean_speed_mps"] for row in valid]),
        "speed_min_mps": min((row["speed_min_mps"] for row in valid), default=None),
        "speed_max_mps": max((row["speed_max_mps"] for row in valid), default=None),
        "distance_m": mean([row["distance_m"] for row in valid]),
        "stopped_fraction": mean([row["stopped_fraction"] for row in valid]),
    }


def _energy_outcomes(valid: list[dict[str, Any]]) -> dict[str, Any]:
    intensities = [row["energy_ml_per_km"] for row in valid if row["energy_ml_per_km"] is not None]

    def mean(values: list[float]) -> float | None:
        return sum(values) / len(values) if values else None

    return {
        "total_ml": mean([row["energy_ml"] for row in valid]),
        "distance_m": mean([row["energy_distance_m"] for row in valid]),
        "ml_per_km": mean(intensities),
    }


def _termination_outcomes(valid: list[dict[str, Any]]) -> dict[str, Any]:
    reasons: dict[str, int] = {}
    for row in valid:
        reasons[row["terminal_reason"]] = reasons.get(row["terminal_reason"], 0) + 1
    return {
        "terminated_count": sum(row["terminated"] for row in valid),
        "truncated_count": sum(row["truncated"] for row in valid),
        "terminal_reasons": reasons,
    }


def scalar_reward(comparison: PolicyComparison) -> dict[str, Any]:
    from .training import beta_probe_statistics, paired_beta_deltas

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
                "comparison": paired(baseline, summary) if baseline is not None else None,
                "training_curve": {
                    "update": [u.update_index for u in training.updates],
                    "reward": [u.total_reward for u in training.updates],
                },
                "policy_probe": {
                    "before": beta_probe_statistics(training.probe_before.model_dump(mode="json")),
                    "after": beta_probe_statistics(training.probe_after.model_dump(mode="json")),
                    "change": paired_beta_deltas(
                        training.probe_before.model_dump(mode="json"),
                        training.probe_after.model_dump(mode="json"),
                    ),
                },
            }
        )
    indexed = {
        (r.arm, r.training.training_seed, r.checkpoint_label): r.evaluation for r in comparison.runs
    }
    contrasts = {}
    for reference_arm, comparison_arm in comparison.contrasts:
        name = f"{comparison_arm}-{reference_arm}"
        checkpoints = {}
        for label in sorted({"final", *(r.checkpoint_label for r in comparison.runs)}):
            effects = []
            for seed in comparison.training_seeds:
                reference = (
                    baseline
                    if reference_arm == comparison.baseline_arm
                    else indexed.get((reference_arm, seed, label))
                )
                target = (
                    baseline
                    if comparison_arm == comparison.baseline_arm
                    else indexed.get((comparison_arm, seed, label))
                )
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
                effect: dict[str, Any] = dict(scenario_effect(delta, comparison.bootstrap))
                if pairs is None:
                    effect["unavailable_reason"] = "missing evaluation for this arm/seed/checkpoint"
                effect["metrics"] = paired_metric_effects(pairs, comparison.bootstrap)
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
    guidance = _guidance_contrasts(comparison)
    return {
        "baseline": arm_outcomes(baseline) if baseline is not None else None,
        "baseline_arm": comparison.baseline_arm,
        "runs": runs,
        "contrasts": contrasts,
        "guidance": guidance,
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


def _guidance_contrasts(comparison: PolicyComparison) -> dict[str, Any]:
    """Paired per-context guidance Beta shifts for each declared contrast.

    The fixed-context probe captures each trained policy's guidance Beta per
    scenario; the paired deltas isolate the comparison arm's guidance
    distribution shift relative to the reference arm, per training seed. The
    ``before`` delta is the matched-initial guard and the ``after`` delta is the
    post-training separation consumed by transfer/positive-control gates.
    """

    from .training import paired_beta_deltas

    training_by_key: dict[tuple[str, int], TrainingRunSummary] = {}
    for run in comparison.runs:
        training_by_key.setdefault((run.arm, run.training.training_seed), run.training)
    guidance: dict[str, Any] = {}
    for reference_arm, comparison_arm in comparison.contrasts:
        name = f"{comparison_arm}-{reference_arm}"
        effects = []
        for seed in comparison.training_seeds:
            reference = training_by_key.get((reference_arm, seed))
            target = training_by_key.get((comparison_arm, seed))
            if reference is None or target is None:
                effects.append({"training_seed": seed, "available": False})
                continue
            effects.append(
                {
                    "training_seed": seed,
                    "available": True,
                    "before": paired_beta_deltas(
                        reference.probe_before.model_dump(mode="json"),
                        target.probe_before.model_dump(mode="json"),
                    ),
                    "after": paired_beta_deltas(
                        reference.probe_after.model_dump(mode="json"),
                        target.probe_after.model_dump(mode="json"),
                    ),
                }
            )
        guidance[name] = {"effects": effects}
    return guidance
