from pathlib import Path
from typing import Any

from eco_planner.artifacts import read_json
from eco_planner.evaluation.artifacts import JobSummary, load_job_summary
from eco_planner.experiments.comparison.analysis import paired


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


def analyze(source: Path, output: Path, *, figures: bool = True) -> dict:
    from eco_planner.reporting.artifacts import separate_output

    source, output = separate_output(source, output)
    return publish(source, output, figures=figures)


def publish(source: Path, output: Path, *, figures: bool = True) -> dict:
    from eco_planner.experiments.guidance.sweep_report import write_report
    from eco_planner.reporting.artifacts import write_analysis

    data = energy_sweep(source)
    output.mkdir(parents=True, exist_ok=True)
    files = []
    if figures:
        from eco_planner.experiments.guidance.sweep_report import plot
        from eco_planner.reporting.plots import plt

        with plt.style.context("default"):
            files = plot(data, output)
    payload = write_analysis(output, data, files, experiment="energy-sweep", source=source)
    write_report(source, output, data, files)
    return {"status": "completed", "output_dir": str(output), **payload}
