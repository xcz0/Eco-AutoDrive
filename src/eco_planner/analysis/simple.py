"""Descriptive analysis of persisted sanity, replay and execution evidence."""

from pathlib import Path
from typing import Any

import numpy as np

from eco_planner.rl.artifacts.summaries import TrainingRunSummary

from .io import read_json
from .statistics import measurement, statistics


def reproducibility(source: Path) -> dict[str, Any]:
    acceptance = read_json(source / "training_report.json")
    runs = []
    seen = set()
    for path in sorted(source.glob("seed-*-replay-*/summary.json")):
        s = TrainingRunSummary.model_validate_json(path.read_text(encoding="utf-8"))
        key = (s.training_seed, s.replay_id)
        if key in seen:
            raise ValueError(f"duplicate seed/replay: {key}")
        seen.add(key)
        values = [u.total_reward for u in s.updates]
        runs.append(
            {
                "training_seed": s.training_seed,
                "replay_id": s.replay_id,
                "updates": [u.update_index for u in s.updates],
                "rewards": values,
                "reward_statistics": statistics(np.asarray(values), [0.0, 0.5, 1.0]),
            }
        )
    expected = {(r["training_seed"], r["replay_id"]) for r in acceptance["runs"]}
    if seen != expected:
        raise ValueError("source runs differ from recorded reproducibility acceptance")
    return {
        "recorded_acceptance": acceptance,
        "runs": runs,
        "interpretation": "Descriptive replay curves; original acceptance is not re-executed.",
    }


def execution(source: Path) -> dict[str, Any]:
    report = read_json(source)
    modes = {}
    for name, entry in report["evaluation_modes"].items():
        elapsed = [job["metadata"]["elapsed_seconds"] for job in entry["jobs"]]
        modes[name] = {
            "outer_wall_s": entry["outer_wall_s"],
            "job_elapsed_s": elapsed,
            "statistics": statistics(np.asarray(elapsed), [0.0, 0.5, 1.0]),
        }
    return {"recorded_comparison": report, "modes": modes}


def mode_report(jobs: list[dict[str, Any]], outer_wall_s: float) -> dict[str, Any]:
    elapsed = [job["metadata"]["elapsed_seconds"] for job in jobs]
    return {
        "job_count": len(jobs),
        "outer_wall_s": measurement([outer_wall_s]),
        "job_elapsed_s": measurement(elapsed),
        "summed_job_elapsed_s": measurement([sum(elapsed)]),
        "jobs": jobs,
    }
