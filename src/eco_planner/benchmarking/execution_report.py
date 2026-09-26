from pathlib import Path


def plot(data: dict, output: Path) -> list[str]:
    from eco_planner.reporting.plots import curves

    files = curves(
        output,
        "job-elapsed",
        {
            k: (list(range(len(v["job_elapsed_s"]))), v["job_elapsed_s"])
            for (k, v) in data["modes"].items()
        },
        "job index",
        "elapsed seconds",
        scatter=True,
    )
    files += curves(
        output,
        "outer-wall",
        {
            "wall time": (
                list(data["modes"]),
                [v["outer_wall_s"]["median"] for v in data["modes"].values()],
            )
        },
        "execution mode",
        "wall seconds",
    )
    return files


def write_report(source: Path, output: Path, data: dict, files: list[str]) -> None:
    from eco_planner.reporting.markdown import write_report as write_markdown

    write_markdown(
        "execution-backend",
        source,
        output,
        data,
        files,
        evidence={
            "modes": data["modes"],
            "recorded_provenance": data["recorded_comparison"].get("provenance"),
        },
    )
