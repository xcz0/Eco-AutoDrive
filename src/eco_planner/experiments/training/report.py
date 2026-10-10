from pathlib import Path


def plot(data: dict, output: Path) -> list[str]:
    from eco_planner.reporting.plots import curves

    runs = data.get("runs", data.get("arms", []))
    series = {}
    for index, record in enumerate(runs):
        metrics = record.get("metrics")
        if metrics is not None:
            values = metrics["post_update_kl"]
            series[str(record.get("label", record.get("source", index)))] = (
                list(range(len(values))),
                values,
            )
    if series:
        return curves(output, "post-update-kl", series, "PPO update", "KL (dimensionless)")
    if data["kind"] == "training-evaluation":
        return curves(
            output,
            "completion",
            {
                str(index): (
                    ["deterministic", *record["stochastic"]],
                    [
                        record["deterministic"]["completion"]["completed_rate"],
                        *[
                            v["outcomes"]["completion"]["completed_rate"]
                            for v in record["stochastic"].values()
                        ],
                    ],
                )
                for (index, record) in enumerate(runs)
            },
            "policy action condition",
            "completed episode fraction",
        )
    return []


def write_report(source: Path, output: Path, data: dict, files: list[str]) -> None:
    from eco_planner.reporting.markdown import write_report as write_markdown

    write_markdown(
        "training",
        source,
        output,
        data,
        files,
        description="Policy update measurements and matched evaluation; all seeds retained.",
    )
