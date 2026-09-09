"""Matched episode comparisons from existing typed evaluation artifacts."""

from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field

from eco_planner.evaluation.artifacts.io import load_job_summary
from eco_planner.evaluation.artifacts.models import JobSummary
from eco_planner.rl.artifacts.summaries import TrainingRunSummary

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


class ComparisonRun(BaseModel):
    model_config = ConfigDict(extra="forbid")
    arm: str
    training_summary: Path
    checkpoint_label: str
    evaluation_dir: Path


class ComparisonConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    protocol: Path
    a0_evaluation_dir: Path
    runs: list[ComparisonRun] = Field(min_length=1)


def scalar_reward(config_path: Path) -> dict[str, Any]:
    from eco_planner.experiments.scalar_reward.config import load_scalar_reward_protocol

    config = ComparisonConfig.model_validate(OmegaConf.to_container(OmegaConf.load(config_path)))
    root = config_path.parent
    protocol = load_scalar_reward_protocol(root / config.protocol)
    baseline = load_job_summary(root / config.a0_evaluation_dir / "summary.json")
    expected = protocol.held_out_pairs()
    if {(e.scenario.map_sequence, e.scenario.seed) for e in baseline.episodes} != expected:
        raise ValueError("A0 does not cover the protocol held-out pool")
    if baseline.policy_checkpoint is not None or baseline.runtime.seed != protocol.evaluation.seed:
        raise ValueError("A0 checkpoint/seed differs from protocol")
    if (
        baseline.workload.evaluated_horizon_steps != protocol.evaluation.horizon_steps
        or baseline.sampler.name != protocol.evaluation.sampler
        or baseline.sampler.ddim_stochasticity != 0
    ):
        raise ValueError("A0 horizon/sampler differs from protocol")
    runs = []
    seen = set()
    for run in config.runs:
        if run.arm not in ("a1", "a2"):
            raise ValueError("comparison arm must be a1 or a2")
        training = TrainingRunSummary.model_validate_json(
            (root / run.training_summary).read_text(encoding="utf-8")
        )
        key = (run.arm, training.training_seed, run.checkpoint_label)
        if key in seen:
            raise ValueError(f"duplicate scalar-reward run: {key}")
        seen.add(key)
        if training.training_seed not in protocol.training.seeds:
            raise ValueError("training seed absent from protocol")
        if training.reward_profile != getattr(protocol.arms, run.arm).reward_profile:
            raise ValueError("training reward differs from declared arm")
        summary = load_job_summary(root / run.evaluation_dir / "summary.json")
        checkpoint = summary.policy_checkpoint
        if checkpoint is None or checkpoint.label != run.checkpoint_label:
            raise ValueError("evaluation checkpoint label differs from comparison config")
        expected_hash = (
            training.initial_policy_hash
            if run.checkpoint_label == "initial"
            else (training.final_policy_hash if run.checkpoint_label == "final" else None)
        )
        if expected_hash is None or checkpoint.policy_hash != expected_hash:
            raise ValueError(
                "evaluation checkpoint is not the declared training initial/final state"
            )
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
