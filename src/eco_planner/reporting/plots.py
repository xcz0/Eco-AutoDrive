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
