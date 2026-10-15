"""Presentation of the fuel-proxy structural proof and speed-attribution diagnostic."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _fmt(value: Any, digits: int = 5) -> str:
    if value is None:
        return "undefined"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def _ci(effect: dict[str, Any]) -> str:
    if effect.get("ci95") is None:
        return "n/a"
    low, high = effect["ci95"]
    return f"[{low:.5g}, {high:.5g}]"


def plot(result: dict[str, Any], output: Path) -> list[str]:
    from eco_planner.reporting.plots import heatmap, plt, save

    if result["status"] != "completed":
        return []
    label = result["config"]["checkpoint_labels"][0]
    primary = result["attribution"][label]["primary"]
    seeds = sorted(primary["per_seed"])
    files: list[str] = []
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(seeds), dtype=np.float64)
    level = [primary["per_seed"][seed]["effects"]["speed_level_term"]["estimate"] for seed in seeds]
    shape = [primary["per_seed"][seed]["effects"]["speed_shape_term"]["estimate"] for seed in seeds]
    total = [primary["per_seed"][seed]["effects"]["ln_intensity"]["estimate"] for seed in seeds]
    width = 0.25
    ax.bar(x - width, level, width, label="speed level 0.036·Δμ_d")
    ax.bar(x, shape, width, label="speed-profile shape Δln shape")
    ax.bar(x + width, total, width, label="total Δln I")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x, seeds)
    ax.set_xlabel("training seed")
    ax.set_ylabel("paired mean Δln I")
    ax.set_title(
        f"Primary contrast {primary['reference_arm']}→{primary['comparison_arm']}"
        f" (label {label}): exact Δln I decomposition"
    )
    ax.legend()
    files.extend(save(fig, output, "primary-ln-intensity-decomposition"))

    scenarios = sorted(
        {p["scenario"] for seed in seeds for p in primary["per_seed"][seed]["pairs"]}
    )
    values: list[list[float | None]] = [[None] * len(seeds) for _ in scenarios]
    for column, seed in enumerate(seeds):
        by_scenario = {
            p["scenario"]: p for p in primary["per_seed"][seed]["pairs"] if p["available"]
        }
        for row, scenario in enumerate(scenarios):
            pair = by_scenario.get(scenario)
            values[row][column] = None if pair is None else pair["deltas"]["ln_intensity"]
    files.extend(
        heatmap(
            output,
            "primary-scenario-ln-intensity-delta",
            values,
            scenarios,
            seeds,
            f"Per-scenario Δln I ({primary['reference_arm']}→{primary['comparison_arm']})",
            number_format=".3g",
        )
    )
    return files


def _contrast_section(name: str, contrast: dict[str, Any], lines: list[str]) -> None:
    title_arm = (
        f"{contrast['reference_arm']} → {contrast['comparison_arm']}"
        if contrast["reference_has_training_seed"]
        else f"{contrast['reference_arm']} (frozen) → {contrast['comparison_arm']}"
    )
    lines += [
        f"## {name}: {title_arm} (label {contrast['checkpoint_label']})",
        "",
        f"Difference direction: `{contrast['difference_direction']}`; noise seed"
        f" {contrast['noise_seed']}; max |identity residual|"
        f" {_fmt(contrast['max_identity_residual'])}.",
        "",
        "| Seed | Pairs | ΔI mL/km [CI] | Δln I [CI] | level 0.036·Δμ_d [CI] | shape Δln [CI]"
        " | Δμ_d m/s | Δmean speed m/s | Δtotal mL | Δdistance m | Δroute compl |",
        "|---|---|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for seed, row in contrast["per_seed"].items():
        effects = row["effects"]
        intensity = effects["intensity_ml_per_km"]
        lines.append(
            f"| {seed} | {row['available_pair_count']}/{row['pair_count']} "
            f"| {_fmt(intensity['estimate'])} {_ci(intensity)} "
            f"| {_fmt(effects['ln_intensity']['estimate'])} {_ci(effects['ln_intensity'])} "
            f"| {_fmt(effects['speed_level_term']['estimate'])} {_ci(effects['speed_level_term'])} "
            f"| {_fmt(effects['speed_shape_term']['estimate'])} {_ci(effects['speed_shape_term'])} "
            f"| {_fmt(effects['mu_d']['estimate'])} "
            f"| {_fmt(effects['time_mean_speed_mps']['estimate'])} "
            f"| {_fmt(effects['total_ml']['estimate'])} "
            f"| {_fmt(effects['distance_m']['estimate'])} "
            f"| {_fmt(effects['route_completion']['estimate'])} |"
        )
    lines += [
        "",
        "Training seeds are not resampled; per-seed scenario bootstrap CIs"
        " (0.95, 10000 resamples, percentile) describe scenario uncertainty only.",
        "",
        "### Family split and leave-one-scenario-out",
        "",
        "| Seed | metric | S mean | SC mean | LOO mean range | sign flips |",
        "|---|---|---:|---:|---|---:|",
    ]
    for seed, row in contrast["per_seed"].items():
        for metric in ("ln_intensity", "speed_level_term", "speed_shape_term"):
            families = row["families"][metric]
            loo = row["leave_one_out"][metric]
            s_mean = families.get("S", {}).get("mean")
            sc_mean = families.get("SC", {}).get("mean")
            loo_range = (
                "n/a" if loo is None else f"[{loo['loo_mean_min']:.5g}, {loo['loo_mean_max']:.5g}]"
            )
            flips = "n/a" if loo is None else loo["sign_change_count"]
            lines.append(
                f"| {seed} | {metric} | {_fmt(s_mean)} | {_fmt(sc_mean)} | {loo_range} | {flips} |"
            )
    lines += [
        "",
        "### Acceleration/jerk descriptives (Spearman across paired deltas; descriptive only)",
        "",
        "| Seed | "
        + " | ".join(
            contrast["per_seed"][next(iter(contrast["per_seed"]))]["acceleration_descriptives"]
        )
        + " |",
        "|---|"
        + "---:|"
        * len(contrast["per_seed"][next(iter(contrast["per_seed"]))]["acceleration_descriptives"]),
    ]
    for seed, row in contrast["per_seed"].items():
        values = " | ".join(_fmt(v, 4) for v in row["acceleration_descriptives"].values())
        lines.append(f"| {seed} | {values} |")
    extremes = contrast["extremes"]
    lines += [
        "",
        "### Extremes and availability",
        "",
        f"- Non-completed episodes: {extremes['non_completed_episodes'] or 'none'}",
        f"- Truncated / non-arrival episodes: {len(extremes['truncated_or_non_arrival_episodes'])}"
        " (see summary.json for the full list)",
        f"- Lowest-distance pairs: {extremes['lowest_distance_pairs']}",
        f"- Lowest-route-completion pairs: {extremes['lowest_route_completion_pairs']}",
        "",
    ]


def write_report(result: dict[str, Any], output: Path, files: list[str]) -> None:
    parity = result["parity"]
    lines = [
        "# Fuel proxy structural proof and E-060 speed attribution",
        "",
        f"Study: `{result['study_name']}`; status: **{result['status']}**.",
        "",
        "Offline diagnostic over the E-060 matched evaluation artifacts"
        " (Issue #113 Task 1). The proxy consumes only end-of-substep speed norms"
        " and executed center displacements; acceleration timing, braking losses,"
        " idle fuel and road grade are structurally invisible to it.",
        "",
        "## Parity audit (trace → formula → summary)",
        "",
        f"- Episodes in scope: {parity['episode_count']}; audited: {parity['audited_count']}"
        f"; not audited: {parity['not_audited'] or 'none'}",
        f"- Max step-level relative formula error: {_fmt(parity['max_step_relative_error'])}",
        f"- Max per-episode identity residual: {_fmt(parity['max_episode_identity_residual'])}",
        f"- Warmup steps inside the audited executed windows: {parity['warmup_steps_total']}",
        f"- Undefined-intensity episodes: {parity['undefined_intensity_episodes'] or 'none'}",
        f"- **Parity passed: {parity['parity_passed']}**",
        f"- Step mismatches: {parity['step_mismatch_episodes'] or 'none'}",
        f"- Failed checks: {parity['failed_check_episodes'] or 'none'}",
        "",
    ]
    lines += ["## Synthetic counterfactuals (through the real provider)", ""]
    profiles = result["counterfactuals"]["profiles"]
    lines += [
        "| Profile | Steps | time-mean m/s | dist m | total mL | I mL/km | mu_d m/s | ln shape |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in profiles.items():
        lines.append(
            f"| {name} | {row['step_count']} | {_fmt(row['time_mean_speed_mps'])} "
            f"| {_fmt(row['distance_m'])} | {_fmt(row['total_ml'])} "
            f"| {_fmt(row['intensity_ml_per_km'])} | {_fmt(row['mu_d'])} "
            f"| {_fmt(row['speed_shape_ln'])} |"
        )
    lines += ["", "Invariants:", ""]
    for name, row in result["counterfactuals"]["invariants"].items():
        flags = {key: value for key, value in row.items() if isinstance(value, bool)}
        lines.append(f"- **{name}**: {flags} — {row['note']}")
    lines.append("")
    if result["status"] == "completed":
        for contrasts in result["attribution"].values():
            _contrast_section("Primary contrast", contrasts["primary"], lines)
            _contrast_section("Context contrast", contrasts["context"], lines)
        lines += ["## Gate P1 inputs", "", "```json"]
        import json

        lines.append(json.dumps(result["gate_p1_inputs"], indent=2))
        lines += ["```", ""]
    else:
        lines += [
            "## Attribution withheld",
            "",
            f"Reason: {result.get('reason')}",
            "",
            "Physical interpretation is paused until the parity failure is located.",
            "",
        ]
    for name in files:
        if name.endswith(".png"):
            lines += [f"![{Path(name).stem}](<{name}>)", "", f"[SVG](<{name[:-4]}.svg>)", ""]
    lines += [
        "",
        "This is an offline recomputation over existing execution traces under the"
        " MetaDrive kinematic waypoint boundary; it does not establish real-vehicle"
        " energy consumption, FASTSim agreement, or any retrained-policy outcome.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
