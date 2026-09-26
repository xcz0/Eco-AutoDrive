from pathlib import Path

from eco_planner.artifacts import read_json


def training(source: Path) -> dict:
    # All Torch/checkpoint measurements are persisted by diagnose/grid; reporting stays lightweight.
    summary = read_json(source / "summary.json")
    if summary["kind"] not in ("training-grid", "training-diagnostics", "training-evaluation"):
        raise ValueError("source is not a current training workflow result")
    return summary


def analyze(source: Path, output: Path, *, figures: bool = True) -> dict:
    from eco_planner.reporting.artifacts import separate_output

    source, output = separate_output(source, output)
    return publish(source, output, figures=figures)


def publish(source: Path, output: Path, *, figures: bool = True) -> dict:
    from eco_planner.experiments.training.report import write_report
    from eco_planner.reporting.artifacts import write_analysis

    data = training(source)
    output.mkdir(parents=True, exist_ok=True)
    files = []
    if figures:
        from eco_planner.experiments.training.report import plot
        from eco_planner.reporting.plots import plt

        with plt.style.context("default"):
            files = plot(data, output)
    payload = write_analysis(output, data, files, experiment="training", source=source)
    write_report(source, output, data, files)
    return {"status": "completed", "output_dir": str(output), **payload}
