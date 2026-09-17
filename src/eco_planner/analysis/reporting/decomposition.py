"""Lon/lat decomposition presentation and attribution plots from prepared evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ARM_NAMES = ("r0", "lon", "lat", "joint")


def write_report(result: dict[str, Any], output: Path, files: list[str]) -> None:
    verdict = result["verdict"]
    lines = [
        "# Guidance lon/lat component decomposition",
        "",
        "Behavior link: impose four matched constant guidance arms through one frozen planner "
        "on the held-out scenarios; only the lateral/longitudinal components change. This is a "
        "causal intervention, not a proposed guidance or execution baseline.",
        "",
        "Attribution by metric (E-040 expected joint direction: "
        f"**{verdict['expected_joint_direction']}**, dominance share "
        f"{verdict['dominance_share']}):",
        "",
        "```json",
        json.dumps(verdict["overall"], indent=2),
        "```",
        "",
        f"Episodes: {result['episode_count']}; transitions: {result['transition_count']}.",
        "",
        "| Seed | Metric | r0 | lon | lat | joint | lon effect | lat effect | joint effect | "
        "interaction | verdict |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for seed, metrics in result["metrics"].items():
        for metric in verdict["attribution_metrics"]:
            data = metrics[metric]
            medians = data["arm_medians"]
            effects = data["effects"]
            lines.append(
                f"| {seed} | {metric} | {medians['r0']:.6g} | {medians['lon']:.6g} | "
                f"{medians['lat']:.6g} | {medians['joint']:.6g} | "
                f"{effects['lon']['median']:.6g} | {effects['lat']['median']:.6g} | "
                f"{effects['joint']['median']:.6g} | {data['interaction']['median']:.6g} | "
                f"{verdict['per_seed'][seed][metric]} |"
            )
    lines += [
        "",
        "Effect directions use the predeclared scenario majority gate; the interaction is "
        "`joint - lon - lat` per scenario.",
        "",
        "## Per-seed directions",
        "",
        "```json",
        json.dumps(verdict["directions"], indent=2),
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
        "The joint arm injects the Rstress constant Beta mean, not the Rstress policy; failure "
        "to reproduce E-040 is itself the state-dependence evidence. This is a kinematic, "
        "no-traffic, held-out intervention over one frozen planner.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _effect(row: dict[str, Any]) -> float:
    value = row.get("median")
    return float("nan") if value is None else float(value)


def plot(result: dict[str, Any], episodes: list[dict[str, Any]], output: Path) -> list[str]:
    from .plots import plt, save

    verdict = result["verdict"]
    metrics = verdict["attribution_metrics"]
    seeds = list(result["metrics"])
    files = []
    width = 0.25
    fig, axes = plt.subplots(
        1, len(seeds), figsize=(5.0 * len(seeds), 3.2), constrained_layout=True, squeeze=False
    )
    offsets = {"lon": -width, "lat": 0.0, "joint": width}
    for ax, seed in zip(axes.flat, seeds, strict=True):
        positions = list(range(len(metrics)))
        for arm in ("lon", "lat", "joint"):
            values = [
                _effect(result["metrics"][seed][metric]["effects"][arm]) for metric in metrics
            ]
            ax.bar(
                [position + offsets[arm] for position in positions],
                values,
                width=width,
                label=arm,
            )
        ax.axhline(0.0, color="grey", linewidth=0.8)
        ax.set_xticks(positions, metrics, rotation=20, ha="right")
        ax.set_title(f"seed {seed}")
        ax.set_ylabel("Median paired effect vs r0")
    axes.flat[-1].legend(fontsize=8)
    files += save(fig, output, "effect-by-metric")

    fig, axes = plt.subplots(
        1, len(seeds), figsize=(5.0 * len(seeds), 3.2), constrained_layout=True, squeeze=False
    )
    checkpoints = list(result["planner_response"])
    times = [float(value) for value in checkpoints]
    for ax, seed in zip(axes.flat, seeds, strict=True):
        for arm in ("lon", "lat", "joint"):
            values = [
                _effect(result["planner_response"][checkpoint][seed][arm])
                for checkpoint in checkpoints
            ]
            ax.plot(times, values, marker="o", markersize=3, label=arm)
        ax.axhline(0.0, color="grey", linewidth=0.8)
        ax.set_title(f"seed {seed}")
        ax.set_xlabel("Predicted time (s)")
        ax.set_ylabel("First-plan forward displacement effect (m)")
    axes.flat[-1].legend(fontsize=8)
    files += save(fig, output, "planner-response")

    fig, axes = plt.subplots(
        1, len(seeds), figsize=(5.0 * len(seeds), 3.2), constrained_layout=True, squeeze=False
    )
    for ax, seed in zip(axes.flat, seeds, strict=True):
        scenarios = result["metrics"][seed]["speed_mps"]["interaction"]["scenarios"]
        values = [_effect({"median": value}) for value in scenarios.values()]
        ax.scatter(range(len(values)), values, s=20)
        ax.axhline(0.0, color="grey", linewidth=0.8)
        ax.set_title(f"seed {seed}")
        ax.set_xlabel("Scenario index")
        ax.set_ylabel("Speed interaction effect (m/s)")
    files += save(fig, output, "speed-interaction-scenarios")
    return files
