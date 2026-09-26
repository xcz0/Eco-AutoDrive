"""Descriptive analysis of persisted sanity, replay and execution evidence."""

from pathlib import Path
from typing import Any

import numpy as np

from eco_planner.artifacts import read_json
from eco_planner.statistics import measurement, statistics


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


def analyze(
    source: Path, output: Path, *, figures: bool = True, source_file: Path | None = None
) -> dict:
    from eco_planner.reporting.artifacts import separate_output

    source, output = separate_output(source, output)
    return publish(source, output, figures=figures, source_file=source_file)


def publish(
    source: Path, output: Path, *, figures: bool = True, source_file: Path | None = None
) -> dict:
    from eco_planner.benchmarking.execution_report import write_report
    from eco_planner.reporting.artifacts import write_analysis

    data = execution(source_file or source / "evaluation_modes.json")
    output.mkdir(parents=True, exist_ok=True)
    files = []
    if figures:
        from eco_planner.benchmarking.execution_report import plot
        from eco_planner.reporting.plots import plt

        with plt.style.context("default"):
            files = plot(data, output)
    payload = write_analysis(output, data, files, experiment="execution-backend", source=source)
    write_report(source, output, data, files)
    return {"status": "completed", "output_dir": str(output), **payload}
