"""Small shared static figure functions, with explicit unavailable cells."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save(figure: Any, output: Path, name: str) -> list[str]:
    directory = output / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    files = []
    try:
        with matplotlib.rc_context({"svg.hashsalt": "eco-planner", "font.family": "DejaVu Sans"}):
            for extension in ("svg", "png"):
                relative = f"figures/{name}.{extension}"
                figure.savefig(
                    output / relative,
                    dpi=150,
                    bbox_inches="tight",
                    metadata={"Date": None} if extension == "svg" else {},
                )
                files.append(relative)
    finally:
        plt.close(figure)
    return files


def heatmap(
    output: Path,
    name: str,
    values: Sequence[Sequence[float | None]],
    rows: Sequence[str],
    columns: Sequence[str],
    title: str,
    *,
    limits: tuple[float, float] | None = None,
    number_format: str = ".3g",
    axis_label: str = "",
) -> list[str]:
    data = np.asarray(values, dtype=float)
    fig, ax = plt.subplots(figsize=(max(5, len(columns) * 1.15), max(3, len(rows) * 0.5 + 1)))
    cmap = matplotlib.colormaps["viridis"].copy()
    cmap.set_bad("#dddddd")
    im = ax.imshow(
        np.ma.masked_invalid(data),
        aspect="auto",
        cmap=cmap,
        vmin=limits[0] if limits else None,
        vmax=limits[1] if limits else None,
    )
    ax.set_xticks(range(len(columns)), columns, rotation=35, ha="right")
    ax.set_yticks(range(len(rows)), rows)
    ax.set_title(title)
    if axis_label:
        ax.set(xlabel=axis_label + " j", ylabel=axis_label + " i")
    for i in range(len(rows)):
        for j in range(len(columns)):
            v = data[i, j]
            ax.text(
                j,
                i,
                "undefined" if not np.isfinite(v) else format(v, number_format),
                ha="center",
                va="center",
                fontsize=8,
                color="black",
                bbox={"facecolor": "white", "alpha": 0.65, "edgecolor": "none"},
            )
    fig.colorbar(im, ax=ax)
    return save(fig, output, name)


def curves(
    output: Path,
    name: str,
    series: dict[str, tuple[Sequence, Sequence]],
    xlabel: str,
    ylabel: str,
    *,
    scatter: bool = False,
) -> list[str]:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, (x, y) in series.items():
        if scatter:
            ax.scatter(x, y, label=label, s=22)
        else:
            ax.plot(x, y, marker="o", markersize=3, label=label)
    ax.set(xlabel=xlabel, ylabel=ylabel, title=name.replace("-", " "))
    ax.grid(alpha=0.2)
    ax.legend(loc="best", fontsize=8)
    return save(fig, output, name)


def fixed_figures(data: dict, output: Path, prefix: str = "") -> list[str]:
    files = []
    arms = data["arms"]
    labels = [str(a.get("label", a.get("lambda"))) for a in arms]
    ablation = "credit_forms" in data
    for metric in ("reward", "raw_advantage", "center_advantage", "normalized_advantage"):
        if ablation and metric != "reward":
            continue
        if metric not in arms[0]:
            continue
        x = [a["lambda"] for a in arms] if "lambda" in arms[0] else labels
        files += curves(
            output,
            prefix + metric.replace("_", "-") + "-response",
            {
                "mean": (x, [a[metric]["mean"] for a in arms]),
                "std": (x, [a[metric]["std"] for a in arms]),
            },
            "lambda" if "lambda" in arms[0] else "arm",
            metric + " (score units)",
        )
    if not ablation:
        for metric in ("cosine", "normalized_advantage_rmse", "sign_flip_fraction"):
            matrix: list[list[float | None]] = [[None] * len(arms) for _ in arms]
            for i, arm in enumerate(arms):
                norms = arm["gradient_norms"]
                norm = norms["z"]["actor_head"] if "z" in norms else norms["actor_head"]
                matrix[i][i] = (1.0 if norm else None) if metric == "cosine" else 0.0
            for pair in data["pairs"]:
                i = labels.index(str(pair.get("arm_i", pair.get("lambda_i"))))
                j = labels.index(str(pair.get("arm_j", pair.get("lambda_j"))))
                v = (
                    pair["gradients"]["actor_head"]["cosine"]
                    if metric == "cosine"
                    else pair[metric]
                )
                matrix[i][j] = matrix[j][i] = v
            files += heatmap(
                output,
                prefix + metric.replace("_", "-"),
                matrix,
                labels,
                labels,
                "Actor head cosine" if metric == "cosine" else metric,
                limits=(-1, 1) if metric == "cosine" else None,
                number_format=".8g" if metric == "cosine" else ".3g",
                axis_label="lambda" if "lambda" in arms[0] else "arm",
            )
            if metric == "cosine":
                files += heatmap(
                    output,
                    prefix + "one-minus-cosine",
                    [[None if v is None else 1 - v for v in row] for row in matrix],
                    labels,
                    labels,
                    "1 - actor head cosine (dimensionless)",
                    axis_label="lambda" if "lambda" in arms[0] else "arm",
                )
    forms = list(FORMS) if "advantage_forms" in data else [""]
    credits = data.get("credit_forms", [""])
    for credit in credits:
        for form in forms:
            entries = [a["credit_forms"][credit] if credit else a for a in arms]
            norms = [e["gradient_norms"][form] if form else e["gradient_norms"] for e in entries]
            groups = list(norms[0])
            files += heatmap(
                output,
                prefix + f"gradient-profile-{credit}-{form}".strip("-"),
                [[n[g] for g in groups] for n in norms],
                labels,
                groups,
                f"Gradient norm {credit} {form}",
            )
    if ablation:
        for form in FORMS:
            entries = [p if form == "z" else p["advantage_forms"][form] for p in data["pairs"]]
            for metric in ("cosine", "rmse", "sign_flip_fraction"):
                values = [
                    [
                        e["gradients"]["actor_head"]["cosine"]
                        if metric == "cosine"
                        else e["normalized_advantage_rmse" if form == "z" else "advantage_rmse"]
                        if metric == "rmse"
                        else e[metric]
                    ]
                    for e in entries
                ]
                files += heatmap(
                    output,
                    prefix + f"credit-{form}-{metric}",
                    values,
                    list(credits),
                    [metric],
                    f"Endpoint {form}: {metric}",
                )
    if "endpoint_forms" in data:
        endpoint = next(
            p for p in data["pairs"] if p["arm_i"] == "r0" and p["arm_j"] == "energy_only"
        )
        entries = [data["endpoint_forms"][f] for f in ("raw", "center")] + [endpoint]
        for metric in ("cosine", "rmse", "sign_flip_fraction"):
            y = [
                e["gradients"]["actor_head"]["cosine"]
                if metric == "cosine"
                else e.get("advantage_rmse", e.get("normalized_advantage_rmse"))
                if metric == "rmse"
                else e[metric]
                for e in entries
            ]
            files += curves(
                output,
                prefix + f"endpoint-{metric}",
                {metric: (["raw", "center", "normalized"], y)},
                "advantage form",
                metric,
            )
    return files


FORMS = ("raw", "center", "z")
