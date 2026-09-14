"""Descriptive analysis of persisted sanity, replay and execution evidence."""

from pathlib import Path
from typing import Any

import numpy as np

from .io import read_json
from .statistics import measurement, statistics


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
