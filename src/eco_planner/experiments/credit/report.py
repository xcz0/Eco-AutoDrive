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
    groups = data["recorded"]["gradient_groups"]
    pairs = data["pairs"]
    labels = [
        f"{p['comparison']}-{p['reference']} / {p['credit_form']} / {p['advantage_form']}"
        for p in pairs
    ]
    for metric in ("cosine", "one_minus_cosine", "norm_ratio_j_over_i"):
        values = []
        for pair in pairs:
            row = []
            for group in groups:
                stats = pair["gradients"][group]
                value = stats["cosine"] if metric == "one_minus_cosine" else stats[metric]
                row.append(
                    1 - value if metric == "one_minus_cosine" and value is not None else value
                )
            values.append(row)
        files += heatmap(output, "gradient-" + metric, values, labels, groups, metric)
    return files


def write_report(source: Path, output: Path, data: dict, files: list[str]) -> None:
    from eco_planner.reporting.markdown import write_report as write_markdown

    write_markdown(
        "credit",
        source,
        output,
        data,
        files,
        description="Reward to credit assignment to actor gradient; no optimizer updates.",
    )
