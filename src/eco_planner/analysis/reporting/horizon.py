"""Execution-horizon presentation and response plots from prepared evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_report(result: dict[str, Any], output: Path, files: list[str]) -> None:
    gate = result["gate_a"]
    lines = [
        "# Guidance execution-horizon intervention",
        "",
        "Behavior link: follow matched imposed guidance through one frozen planner to "
        "executed motion while only the number of waypoints executed per plan changes. "
        "This is a causal intervention, not a proposed execution baseline.",
        "",
        f"Gate A: **{gate['status']}**",
        "",
        f"Episodes: {result['episode_count']}; transitions: {result['transition_count']}.",
        "",
        "Executed-speed direction per horizon (0.1 s prefix first):",
        "",
        "```json",
        json.dumps(gate["executed_speed_direction"], indent=2),
        "```",
        "",
        "First-plan forward-displacement direction per checkpoint (s):",
        "",
        "```json",
        json.dumps(gate["planner_response_direction"], indent=2),
        "```",
        "",
        "| Horizon | Metric | Direction | Positive | Negative | Pass |",
        "|---:|---|---|---:|---:|---|",
    ]
    for horizon, data in result["horizons"].items():
        for metric, metrics in data["metrics"].items():
            lines.append(
                f"| {horizon} | {metric} | {metrics['direction']} | "
                f"{metrics['positive_pass_count']} | {metrics['negative_pass_count']} | "
                f"{metrics['passed']} |"
            )
    lines += [
        "",
        "## Executed response detail",
        "",
        "| Horizon | Metric | Scenario | Spearman | Endpoint effect | Noise | Pass |",
        "|---:|---|---|---:|---:|---:|---|",
    ]
    for horizon, data in result["horizons"].items():
        for metric in ("speed_mps", "energy_ml_per_km"):
            metrics = data["metrics"][metric]
            for scenario, row in metrics["scenarios"].items():
                lines.append(
                    f"| {horizon} | {metric} | {scenario} | {row['rho']} | "
                    f"{row['effect']} | {row.get('noise_scale')} | {row['passed']} |"
                )
    lines += [
        "",
        "## First-plan predicted response",
        "",
        "| Checkpoint (s) | Scenario | Spearman | Endpoint effect | Noise | Pass |",
        "|---:|---|---:|---:|---:|---|",
    ]
    for checkpoint, response in result["planner_response"].items():
        for scenario, row in response["scenarios"].items():
            lines.append(
                f"| {checkpoint} | {scenario} | {row['rho']} | {row['effect']} | "
                f"{row.get('noise_scale')} | {row['passed']} |"
            )
    lines += [
        "",
        "## Safety and attribution",
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
        "This is a kinematic, no-traffic, 2 s proxy-energy intervention over one frozen "
        "planner; it does not establish learned behavior, physical trackability, or a "
        "recommended execution horizon.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _effect(row: dict[str, Any]) -> float:
    value = row.get("effect")
    return float("nan") if value is None else float(value)


def _noise(row: dict[str, Any]) -> float:
    value = row.get("noise_scale")
    return float("nan") if value is None else float(value)


def _grid(names: list[str]) -> tuple[Any, list[Any]]:
    from .plots import plt

    count = len(names)
    columns = min(4, count)
    rows = (count + columns - 1) // columns
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(3.75 * columns, 2.75 * rows),
        constrained_layout=True,
        squeeze=False,
    )
    return fig, list(axes.flat)


def plot(result: dict[str, Any], episodes: list[dict[str, Any]], output: Path) -> list[str]:
    from .plots import save

    horizons = sorted(int(value) for value in result["horizons"])
    names = list(result["horizons"][str(horizons[0])]["metrics"]["speed_mps"]["scenarios"])
    files = []

    fig, axes = _grid(names)
    for ax, name in zip(axes, names, strict=False):
        effects, noises = [], []
        for horizon in horizons:
            row = result["horizons"][str(horizon)]["metrics"]["speed_mps"]["scenarios"][name]
            effects.append(_effect(row))
            noises.append(_noise(row))
        ax.errorbar(horizons, effects, yerr=noises, fmt="ko-", capsize=3)
        ax.axhline(0.0, color="grey", linewidth=0.8)
        ax.set_title(name)
        ax.set_xscale("log")
        ax.set_xticks(horizons)
        ax.set_xticklabels([str(value) for value in horizons])
        ax.set_xlabel("Executed waypoints per plan")
        ax.set_ylabel("Executed speed endpoint effect")
    files += save(fig, output, "effect-by-horizon")

    fig, axes = _grid(names)
    for ax, name in zip(axes, names, strict=False):
        for horizon in horizons:
            row = result["horizons"][str(horizon)]["metrics"]["speed_mps"]["scenarios"][name]
            if "arm_means" in row:
                ax.plot([-1, -0.5, 0, 0.5, 1], row["arm_means"], "o-", label=str(horizon))
        ax.set_title(name)
        ax.set_xlabel("Longitudinal guidance")
        ax.set_ylabel("Mean executed speed (m/s)")
    axes[len(names) - 1].legend(title="waypoints", fontsize=7)
    files += save(fig, output, "horizon-speed-response")

    fig, axes = _grid(names)
    checkpoints = list(result["planner_response"])
    times = [float(value) for value in checkpoints]
    for ax, name in zip(axes, names, strict=False):
        effects = [_effect(result["planner_response"][cp]["scenarios"][name]) for cp in checkpoints]
        ax.plot(times, effects, "ko-")
        ax.axhline(0.0, color="grey", linewidth=0.8)
        ax.set_title(name)
        ax.set_xlabel("Predicted time (s)")
        ax.set_ylabel("Forward displacement effect (m)")
    files += save(fig, output, "planner-response")
    return files
