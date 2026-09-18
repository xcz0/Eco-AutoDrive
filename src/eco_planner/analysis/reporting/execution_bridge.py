"""Frozen-policy execution-contract bridge presentation and crossover plots."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..horizon import PLANNER_RESPONSE_CHECKPOINTS_S

_CHECKPOINT_INDICES = [round(value * 10) - 1 for value in PLANNER_RESPONSE_CHECKPOINTS_S]


def write_report(result: dict[str, Any], output: Path, files: list[str]) -> None:
    gate = result["gate"]
    part_a = gate["part_a"]
    part_b = gate["part_b"]
    directions = part_a["directions"]
    horizons = sorted(next(iter(directions.values()))["speed_mps"], key=int)
    detail = next(iter(part_a["detail"].values()))
    lines = [
        "# Frozen-policy execution-contract bridge",
        "",
        "Behavior link: replay the E-040 frozen R0/Rstress final checkpoints in one matched "
        "closed loop across explicit execution prefixes, then audit both policies' local "
        "guidance actions on identical held-out contexts. This is a diagnostic causal "
        "intervention, not a proposed execution contract.",
        "",
        f"Verdict: **{gate['verdict']}**",
        "",
        f"Episodes: {result['episode_count']}; transitions: {result['transition_count']}; "
        f"same-state contexts: "
        f"{0 if result['same_state'] is None else result['same_state']['context_count']}.",
        "",
        "## Part A — closed-loop crossover",
        "",
        f"`{detail['counterfactual_arm']} - {detail['reference_arm']}` per-scenario direction "
        "by execution prefix:",
        "",
        "| Seed | Metric | " + " | ".join(f"k={horizon}" for horizon in horizons) + " |",
        "|---|---|" + "---|" * len(horizons),
    ]
    for seed, seed_directions in directions.items():
        for metric, horizon_directions in seed_directions.items():
            cells = " | ".join(horizon_directions[horizon] for horizon in horizons)
            lines.append(f"| {seed} | {metric} | {cells} |")
    lines += [
        "",
        "First-plan planner response is checked to match across execution prefixes before the "
        "closed-loop effects are read; safety, termination, and tracker errors are recorded per "
        "prefix in the raw `episodes.json`.",
        "",
        "## Part B — same-state local bridge",
        "",
        "```json",
        json.dumps(part_b, indent=2),
        "```",
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
        "This is a kinematic, no-traffic, BF16, frozen-policy proxy study; it does not establish "
        "real-vehicle behavior, a recommended execution contract, or a new baseline.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _median(effect: dict[str, Any]) -> float:
    value = effect.get("median")
    return float("nan") if value is None else float(value)


def plot(result: dict[str, Any], episodes: list[dict[str, Any]], output: Path) -> list[str]:
    from .plots import curves

    gate = result["gate"]
    counterfactual = next(iter(gate["part_a"]["detail"].values()))["counterfactual_arm"]
    horizons = sorted({h for seed in result["metrics"] for h in result["metrics"][seed]}, key=int)
    times = [int(horizon) for horizon in horizons]
    files = []
    for metric in ("speed_mps", "energy_ml_per_km"):
        series = {}
        for seed in result["metrics"]:
            values = [
                _median(result["metrics"][seed][horizon][metric]["effects"][counterfactual])
                for horizon in horizons
            ]
            series[f"seed {seed}"] = (times, values)
        files += curves(
            output,
            f"part-a-{metric.replace('_', '-')}",
            series,
            "Execution prefix (waypoints / plan)",
            f"Median paired effect ({metric})",
        )
    if result["same_state"] is not None:
        checkpoints = list(PLANNER_RESPONSE_CHECKPOINTS_S)
        x = [float(value) for value in checkpoints]
        series = {}
        for seed, row in result["same_state"]["seeds"].items():
            delta = row["delta_forward"]
            series[f"seed {seed}"] = (
                x,
                [delta[str(checkpoint)]["median"] for checkpoint in checkpoints],
            )
        files += curves(
            output,
            "part-b-forward-displacement",
            series,
            "Predicted time (s)",
            "Same-state forward-displacement effect (m)",
        )
    return files
