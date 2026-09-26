from pathlib import Path

from eco_planner.artifacts import read_json


def analyze(source: Path, output: Path, *, figures: bool = True) -> dict:
    from eco_planner.reporting.artifacts import separate_output

    source, output = separate_output(source, output)
    return publish(source, output, figures=figures)


def publish(source: Path, output: Path, *, figures: bool = True) -> dict:
    from eco_planner.experiments.reward.sanity_report import write_report
    from eco_planner.reporting.artifacts import write_analysis

    data = read_json(source / "sanity_report.json")
    output.mkdir(parents=True, exist_ok=True)
    files = []
    if figures:
        from eco_planner.experiments.reward.sanity_report import plot
        from eco_planner.reporting.plots import plt

        with plt.style.context("default"):
            files = plot(data, output)
    payload = write_analysis(output, data, files, experiment="reward-sanity", source=source)
    write_report(source, output, data, files)
    return {"status": "completed", "output_dir": str(output), **payload}
