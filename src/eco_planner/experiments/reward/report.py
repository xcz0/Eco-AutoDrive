from pathlib import Path


def plot(data: dict, output: Path) -> list[str]:
    from eco_planner.reporting.plots import curves, heatmap

    files = curves(
        output,
        "distributions",
        {
            name: (
                [float(q) for q in value["all"]["quantiles"]],
                list(value["all"]["quantiles"].values()),
            )
            for (name, value) in data["distributions"].items()
            if name.endswith("__reward_total") or name.endswith("__reward")
        },
        "quantile",
        "reward (dimensionless)",
    )
    names = list(data["distributions"])
    files += heatmap(
        output,
        "component-means",
        [[data["distributions"][name]["all"]["mean"]] for name in names],
        names,
        ["mean"],
        "Reward component means (dimensionless)",
    )
    return files


def write_report(source: Path, output: Path, data: dict, files: list[str]) -> None:
    from eco_planner.reporting.markdown import write_report as write_markdown

    write_markdown(
        "reward",
        source,
        output,
        data,
        files,
        description="Reward distributions and calibration from one persisted batch.",
    )
