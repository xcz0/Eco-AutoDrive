from pathlib import Path


def plot(data: dict, output: Path) -> list[str]:
    from eco_planner.reporting.plots import heatmap

    cases = data["cases"]
    keys = list(next(iter(cases.values()))["components"])
    return heatmap(
        output,
        "reward-components",
        [[v["components"][k] for k in keys] for v in cases.values()],
        list(cases),
        keys,
        "Reward components",
        limits=(0, 1),
    )


def write_report(source: Path, output: Path, data: dict, files: list[str]) -> None:
    from eco_planner.reporting.markdown import write_report as write_markdown

    write_markdown(
        "reward-sanity",
        source,
        output,
        data,
        files,
        description="Reward: synthetic checks establish component correctness, not learning.",
    )
