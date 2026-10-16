"""Matched episode comparisons from existing typed evaluation artifacts."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from eco_planner.evaluation.artifacts import JobSummary
from eco_planner.rl.artifacts.summaries import TrainingRunSummary
from eco_planner.statistics import (
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


def paired_robustness(pairs: dict[str, Any] | None) -> dict[str, Any] | None:
    """Scenario-family split and leave-one-scenario-out sign stability of paired deltas.

    Plain statistics without resampling: per map family (pair key index 1) and,
    for each metric, the mean with each single scenario excluded, so that
    single-family or single-scenario dominance stays visible.
    """

    if pairs is None:
        return None
    rows = [row for row in pairs["pairs"] if row["available"]]
    families = sorted({row["key"][1] for row in rows})
    result: dict[str, Any] = {"families": {}, "leave_one_out": {}}
    for metric in _PAIRED_METRICS:
        field = metric + "_delta"
        family_effects: dict[str, Any] = {}
        for family in families:
            values = [
                row[field] for row in rows if row["key"][1] == family and row[field] is not None
            ]
            family_effects[family] = (
                {"mean": float(np.mean(values)), "count": len(values)} if values else None
            )
        result["families"][metric] = family_effects
        deltas = [row[field] for row in rows if row[field] is not None]
        if len(deltas) < 2:
            result["leave_one_out"][metric] = None
            continue
        total = float(np.sum(deltas))
        full_mean = total / len(deltas)
        loo_means = [(total - value) / (len(deltas) - 1) for value in deltas]
        result["leave_one_out"][metric] = {
            "sample_count": len(deltas),
            "full_mean": full_mean,
            "loo_mean_min": float(np.min(loo_means)),
            "loo_mean_max": float(np.max(loo_means)),
            "sign_change_count": int(
                sum(np.sign(mean) != np.sign(full_mean) for mean in loo_means)
            ),
        }
    return result


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
    from eco_planner.experiments.training.statistics import (
        beta_probe_statistics,
        paired_beta_deltas,
    )

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
                effect["robustness"] = paired_robustness(pairs)
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


def analyze(
    source: Path, output: Path, *, figures: bool = True, scalar_comparison: PolicyComparison
) -> dict:
    from eco_planner.reporting.artifacts import separate_output

    source, output = separate_output(source, output, sources=scalar_comparison.source_directories)
    return publish(source, output, figures=figures, scalar_comparison=scalar_comparison)


def publish(
    source: Path, output: Path, *, figures: bool = True, scalar_comparison: PolicyComparison
) -> dict:
    from eco_planner.experiments.comparison.report import write_report
    from eco_planner.reporting.artifacts import write_analysis

    data = scalar_reward(scalar_comparison)
    output.mkdir(parents=True, exist_ok=True)
    files = []
    if figures:
        from eco_planner.experiments.comparison.report import plot
        from eco_planner.reporting.plots import plt

        with plt.style.context("default"):
            files = plot(data, output)
    payload = write_analysis(output, data, files, experiment="scalar-reward", source=source)
    write_report(source, output, data, files)
    return {"status": "completed", "output_dir": str(output), **payload}


def publish_scalar_run(source: Path, *, training: bool, figures: bool = True) -> None:
    """Single-run presentation; cross-arm comparisons require explicit input grouping."""
    from .analysis import episode_records

    if training:
        from eco_planner.rl.artifacts.summaries import TrainingRunSummary

        summary = TrainingRunSummary.model_validate_json(
            (source / "summary.json").read_text(encoding="utf-8")
        )
        data = summary.model_dump(mode="json")
    else:
        from eco_planner.evaluation.artifacts.io import load_job_summary

        rows = episode_records(load_job_summary(source / "summary.json"))
        data = {"episodes": rows}
    files = []
    if figures:
        from eco_planner.reporting.plots import plt

        from .report import scalar_run_figures

        with plt.style.context("default"):
            files = scalar_run_figures(data, source, training=training)
    from eco_planner.reporting.artifacts import write_analysis

    from .report import write_report

    write_analysis(source, data, files)
    write_report(source, source, data, files)


def _guidance_contrasts(comparison: PolicyComparison) -> dict[str, Any]:
    """Paired per-context guidance Beta shifts for each declared contrast.

    The fixed-context probe captures each trained policy's guidance Beta per
    scenario; the paired deltas isolate the comparison arm's guidance
    distribution shift relative to the reference arm, per training seed. The
    ``before`` delta is the matched-initial guard and the ``after`` delta is the
    post-training separation consumed by transfer/positive-control gates.
    """

    from eco_planner.experiments.training.statistics import paired_beta_deltas

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
