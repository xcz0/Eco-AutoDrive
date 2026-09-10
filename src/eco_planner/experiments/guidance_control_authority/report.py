"""Offline Task D recomputation and response plots."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from eco_planner.analysis.io import write_json

from .config import InterventionConfig
from .diagnostics import analyze_episodes


def publish(source: Path, output: Path, *, figures: bool = True) -> dict[str, Any]:
    config = InterventionConfig.model_validate_json(
        (source / "intervention_config.json").read_text(encoding="utf-8")
    )
    episodes = json.loads((source / "episodes.json").read_text(encoding="utf-8"))["episodes"]
    scenarios = json.loads((source / "scenarios.json").read_text(encoding="utf-8"))["scenarios"]
    result = analyze_episodes(episodes, config, [s["name"] for s in scenarios])
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "summary.json", {"status": "completed", **result})
    files = plot(result, episodes, output) if figures else []
    lines = [
        "# Task D: Guidance control authority",
        "",
        f"Gate D: **{'PASSED' if result['gate_d']['passed'] else 'FAILED'}**",
        "",
        f"Attribution: `{result['gate_d']['attribution']}`",
        "",
        f"Episodes: {result['episode_count']}; transitions: {result['transition_count']}.",
        "",
        "Noise = sqrt(mean of within-arm sample variances over paired repeats)).",
        "Each metric needs at least 9/16 scenarios with the same effect direction.",
        "",
        "| Window | Metric | Scenario | Spearman | Endpoint effect | Noise | Pass |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for window, metrics in result["windows"].items():
        for metric, data in metrics.items():
            for scenario, row in data["scenarios"].items():
                lines.append(
                    f"| {window} | {metric} | {scenario} | {row['rho']} | "
                    f"{row['effect']} | {row.get('noise_scale')} | {row['passed']} |"
                )
    lines += [
        "",
        "## Safety and attribution",
        "",
        "```json",
        json.dumps(result["gate_d"], indent=2),
        "```",
        "",
    ]
    lines += [f"![{name}]({name})" for name in files]
    lines += [
        "",
        "This is a kinematic, no-traffic, 2 s proxy-energy intervention study; "
        "it does not establish learned behavior or physical vehicle trackability.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(
        output / "analysis.json",
        {
            "experiment": "guidance-control-authority",
            "source": str(source.resolve()),
            "evidence": result,
            "figures": files,
        },
    )
    return {"status": "completed", "output_dir": str(output), "gate_d": result["gate_d"]}


def plot(result: dict[str, Any], episodes: list[dict[str, Any]], output: Path) -> list[str]:
    import matplotlib.pyplot as plt

    files = []
    for metric, label in (
        ("speed_mps", "Mean executed speed (m/s)"),
        ("energy_ml_per_km", "Fuel proxy intensity (mL/km)"),
    ):
        fig, axes = plt.subplots(4, 4, figsize=(15, 11), constrained_layout=True)
        rows = result["windows"]["short_horizon"][metric]["scenarios"]
        for ax, (name, row) in zip(axes.flat, rows.items(), strict=True):
            if "values" in row:
                ax.plot([-1, -0.5, 0, 0.5, 1], row["values"], alpha=0.35)
                ax.plot([-1, -0.5, 0, 0.5, 1], row["arm_means"], "ko-")
            ax.set_title(name)
            ax.set_xlabel("Longitudinal guidance")
            ax.set_ylabel(label)
        filename = f"response-{metric}.png"
        fig.savefig(output / filename, dpi=140)
        plt.close(fig)
        files.append(filename)
    fig, axes = plt.subplots(4, 4, figsize=(15, 11), constrained_layout=True)
    names = list(result["windows"]["short_horizon"]["speed_mps"]["scenarios"])
    colors = dict(
        zip([-1, -0.5, 0, 0.5, 1], plt.get_cmap("coolwarm")([0, 0.25, 0.5, 0.75, 1]), strict=True)
    )
    for ax, name in zip(axes.flat, names, strict=True):
        for episode in episodes:
            if episode["scenario"] != name:
                continue
            steps = episode["steps"]
            ax.plot(
                [sum(s["dt_s"] for s in steps[: i + 1]) for i in range(len(steps))],
                [s["speed_mps"] for s in steps],
                color=colors[episode["g_lon"]],
                alpha=0.6,
            )
        ax.set_title(name)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Executed speed (m/s)")
    fig.suptitle("Guidance -1 (blue) to +1 (red); three paired noise repeats")
    filename = "speed-trajectories.png"
    fig.savefig(output / filename, dpi=140)
    plt.close(fig)
    return [*files, filename]
