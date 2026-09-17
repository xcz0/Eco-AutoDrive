"""Critic-attribution presentation and gradient/credit trajectory plots."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

_COMPARISON_GROUPS = ("actor_head", "lateral", "longitudinal")


def _finite(values: list[float | None]) -> list[float]:
    return [float("nan") if value is None else value for value in values]


def _series(checkpoints: list[dict], metric: str, group: str | None = None) -> dict:
    output: dict[str, tuple[list, list]] = {}
    for run in dict.fromkeys(row["run"] for row in checkpoints):
        rows = sorted(
            (row for row in checkpoints if row["run"] == run), key=lambda row: row["update_index"]
        )
        updates = [row["update_index"] for row in rows]
        if metric == "sign_flip":
            values = [row["advantage"]["sign_flip_fraction"] for row in rows]
        elif metric == "spearman":
            values = _finite([row["advantage"]["spearman"] for row in rows])
        elif metric == "gradient" and group is not None:
            values = _finite([row["gradients"][group]["cosine"] for row in rows])
        elif metric == "explained_variance":
            values = [row["critic"]["explained_variance"] for row in rows]
        elif metric == "value_loss":
            values = [row["critic"]["value_loss"] for row in rows]
        else:
            raise ValueError(f"unknown critic-attribution metric: {metric}")
        output[run] = (updates, values)
    return output


def plot(result: dict[str, Any], output: Path) -> list[str]:
    from .plots import curves

    checkpoints = result["checkpoints"]
    comparisons = list(dict.fromkeys(row["comparison"] for row in checkpoints))
    files = curves(
        output, "advantage-sign-flip", _series(checkpoints, "sign_flip"), "PPO update", "fraction"
    )
    files += curves(
        output, "advantage-spearman", _series(checkpoints, "spearman"), "PPO update", "Spearman"
    )
    for comparison in comparisons:
        rows = [row for row in checkpoints if row["comparison"] == comparison]
        suffix = "" if len(comparisons) == 1 else f"-{comparison}"
        for group in _COMPARISON_GROUPS:
            files += curves(
                output,
                f"actor-gradient-cosine-{group}{suffix}",
                _series(rows, "gradient", group),
                "PPO update",
                "cosine",
            )
    files += curves(
        output,
        "critic-explained-variance",
        _series(checkpoints, "explained_variance"),
        "PPO update",
        "explained variance",
    )
    files += curves(
        output,
        "critic-value-loss",
        _series(checkpoints, "value_loss"),
        "PPO update",
        "value loss",
    )
    cross_arm = result.get("cross_arm", [])
    for group in _COMPARISON_GROUPS:
        series: dict[str, tuple[Sequence, Sequence]] = {}
        for seed in sorted({row["training_seed"] for row in cross_arm}):
            rows = sorted(
                (
                    row
                    for row in cross_arm
                    if row["training_seed"] == seed and row["group"] == group
                ),
                key=lambda row: row["update_index"],
            )
            if rows:
                series[f"seed-{seed}"] = (
                    [row["update_index"] for row in rows],
                    _finite([row["cosine"] for row in rows]),
                )
        if series:
            files += curves(
                output,
                f"cross-arm-gradient-cosine-{group}",
                series,
                "PPO update",
                "cosine",
            )
    return files


def write_report(result: dict[str, Any], output: Path, files: list[str]) -> None:
    gate = result["gate"]
    checkpoints = result["checkpoints"]
    lines = [
        "# Training critic attribution",
        "",
        "Backward-only offline attribution over saved training runs: for every saved PPO "
        "update, compare the actor gradient produced by the current standard GAE against a "
        "critic-free reward-only GAE under the same batch, episode boundaries and reward.",
        "The analysis layer recomputes the advantage comparisons from the persisted arrays "
        "and re-attaches the recorded gradient measurements unchanged.",
        "",
        f"Verdict: **{gate['verdict']}**",
        "",
        f"Runs: {len(result['runs'])}; advantage form: `{result['advantage_form']}`; "
        f"checkpoints: {len(result['checkpoints'])}; update indices: "
        f"{len(result['update_indices'])} (0–{max(result['update_indices'])}).",
        "",
        "## Materiality gate",
        "",
        "| Comparison | passed | checkpoints | failures | undefined checks |",
        "|---|---|---:|---:|---:|",
    ]
    for label, entry in gate["comparisons"].items():
        lines.append(
            f"| {label} | {entry['passed']} | {entry['checkpoint_count']} | "
            f"{len(entry['failures'])} | {len(entry['excluded_undefined'])} |"
        )
    lines += [
        "",
        "Per-run extrema over the checkpoint trajectory (standard GAE vs critic-free):",
        "",
        "| Run | updates | min actor-head cos | min lateral cos | min longitudinal cos | "
        "max advantage sign-flip |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for run in dict.fromkeys(row["run"] for row in checkpoints):
        rows = [row for row in checkpoints if row["run"] == run]
        lines.append(
            f"| {run} | {len(rows)} | {_minimum(rows, 'actor_head'):.6g} | "
            f"{_minimum(rows, 'lateral'):.6g} | {_minimum(rows, 'longitudinal'):.6g} | "
            f"{max(row['advantage']['sign_flip_fraction'] for row in rows):.6g} |"
        )
    critic = [row["critic"]["explained_variance"] for row in checkpoints]
    lines += [
        "",
        f"Critic explained variance along the trajectory: min {min(critic):.6g}, "
        f"max {max(critic):.6g}.",
        "",
        "## Recorded gate",
        "",
        "```json",
        json.dumps(gate, indent=2),
        "```",
        "",
    ]
    for name in files:
        if name.endswith(".png"):
            lines += [
                f"![{Path(name).stem}](<{name}>)",
                "",
                f"[SVG](<{name[:-4]}.svg>) · [PNG](<{name}>)",
                "",
            ]
    lines += [
        "",
        "Offline backward-only actor-gradient evidence on saved rollouts; it does not "
        "establish learned behavior, real-vehicle energy, or a new training protocol.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _minimum(rows: list[dict], group: str) -> float:
    values = [row["gradients"][group]["cosine"] for row in rows]
    defined = [value for value in values if value is not None]
    return float(min(defined)) if defined else float("nan")
