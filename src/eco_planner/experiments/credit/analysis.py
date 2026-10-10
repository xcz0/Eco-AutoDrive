from pathlib import Path

from eco_planner.experiments.fixed_batch import fixed


def analyze(source: Path, output: Path, *, figures: bool = True) -> dict:
    from eco_planner.reporting.artifacts import separate_output

    source, output = separate_output(source, output)
    return publish(source, output, figures=figures)


def publish(source: Path, output: Path, *, figures: bool = True) -> dict:
    from eco_planner.experiments.credit.report import write_report
    from eco_planner.reporting.artifacts import write_analysis

    data = fixed(source)
    output.mkdir(parents=True, exist_ok=True)
    files = []
    if figures:
        from eco_planner.experiments.credit.report import plot
        from eco_planner.reporting.plots import plt

        with plt.style.context("default"):
            files = plot(data, output)
    payload = write_analysis(output, data, files, experiment="credit", source=source)
    write_report(source, output, data, files)
    return {"status": "completed", "output_dir": str(output), **payload}
