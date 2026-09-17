"""Replanning-deferral presentation and temporal-consistency plots."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..horizon import PLANNER_RESPONSE_CHECKPOINTS_S

_CHECKPOINT_INDICES = [round(value * 10) - 1 for value in PLANNER_RESPONSE_CHECKPOINTS_S]


def write_report(result: dict[str, Any], output: Path, files: list[str]) -> None:
    gate = result["gate_c"]
    lines = [
        "# Guidance replanning-deferral trace",
        "",
        "Behavior link: follow matched imposed guidance through one frozen planner under the "
        "baseline 0.1 s receding-horizon execution and compare, across consecutive replanning "
        "cycles, the matched endpoint difference at every predicted waypoint. This is a causal "
        "diagnostic, not a proposed execution baseline.",
        "",
        f"Gate C: **{gate['status']}**",
        "",
        f"Episodes: {result['episode_count']}; transitions: {result['transition_count']}; "
        f"scenarios flagged `repeated_deferral`: "
        f"{len(gate['repeated_deferral_scenarios'])}/{len(result['scenarios'])}.",
        "",
        "Per-scenario deferral evidence (matched `+1 - (-1)` forward displacement):",
        "",
        "| Scenario | cycles | median first (m) | median full (m) | first<0 | full>0 | deferred | "
        "crossing median (step) | crossing spread | repeated |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name, row in result["scenarios"].items():
        crossing = row["crossing_median_step"]
        spread = row["crossing_spread_steps"]
        lines.append(
            f"| {name} | {row['cycle_count']} | {row['median_first_waypoint_effect_m']:.4g} | "
            f"{row['median_full_horizon_effect_m']:.4g} | {row['first_negative_count']} | "
            f"{row['full_positive_count']} | {row['deferred_count']} | "
            f"{'undefined' if crossing is None else f'{crossing:g}'} | "
            f"{'undefined' if spread is None else spread} | {row['repeated_deferral']} |"
        )
    lines += [
        "",
        "Median matched effect across scenarios at the fixed checkpoints "
        "(0.1/0.2/0.5/1/2/4/8 s), one row per replanning cycle:",
        "",
        "| Cycle | " + " | ".join(f"{value}s" for value in PLANNER_RESPONSE_CHECKPOINTS_S) + " |",
        "|---:|" + "---:|" * len(PLANNER_RESPONSE_CHECKPOINTS_S),
    ]
    for cycle, curve in enumerate(result["aggregate"]["median_effect_by_cycle"]):
        values = " | ".join(f"{curve[index]:.4g}" for index in _CHECKPOINT_INDICES)
        lines.append(f"| {cycle} | {values} |")
    lines += [
        "",
        "## Gate",
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
        "This is a kinematic, no-traffic, frozen-planner proxy trace; it does not establish "
        "learned behavior, a recommended execution horizon, or real-vehicle energy.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot(result: dict[str, Any], episodes: list[dict[str, Any]], output: Path) -> list[str]:
    from .plots import curves, heatmap, plt, save

    checkpoints = list(PLANNER_RESPONSE_CHECKPOINTS_S)
    median_curves = result["aggregate"]["median_effect_by_cycle"]
    rows = [str(cycle) for cycle in range(len(median_curves))]
    values = [[curve[index] for index in _CHECKPOINT_INDICES] for curve in median_curves]
    files = heatmap(
        output,
        "effect-by-cycle-and-time",
        values,
        rows,
        [f"{value}" for value in checkpoints],
        "Median matched forward-displacement effect (m)",
        axis_label="",
    )

    cycles = list(range(len(result["aggregate"]["first_waypoint_effect_by_cycle"])))
    files += curves(
        output,
        "first-vs-full-by-cycle",
        {
            "first waypoint (0.1 s)": (
                cycles,
                result["aggregate"]["first_waypoint_effect_by_cycle"],
            ),
            "full horizon (8 s)": (cycles, result["aggregate"]["full_horizon_effect_by_cycle"]),
        },
        "Replanning cycle",
        "Median matched effect (m)",
    )

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, row in result["scenarios"].items():
        defined = [
            (cycle["cycle"], cycle["crossing_step"])
            for cycle in row["cycles"]
            if cycle["crossing_step"] is not None
        ]
        if defined:
            xs, ys = zip(*defined, strict=True)
            ax.plot(xs, ys, marker="o", markersize=3, label=name)
    ax.axhline(1, color="grey", linewidth=0.8, linestyle="--")
    ax.set(
        xlabel="Replanning cycle",
        ylabel="Zero-crossing waypoint step",
        title="zero crossing per cycle (dashed = executed prefix)",
    )
    ax.grid(alpha=0.2)
    ax.legend(loc="best", fontsize=7)
    files += save(fig, output, "crossing-step-by-cycle")
    return files
