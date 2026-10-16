"""Presentation of the E-062 fixed-trace offline physics review."""

from __future__ import annotations

import json
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
    interval = effect.get("ci95")
    if interval is None:
        return "n/a"
    low, high = interval
    return f"[{low:.5g}, {high:.5g}]"


def plot(result: dict[str, Any], output: Path) -> list[str]:
    from eco_planner.reporting.plots import heatmap, plt, save

    if result["status"] != "completed":
        return []
    label = result["config"]["checkpoint_labels"][0]
    primary = result["contrasts"][label]["primary"]
    seeds = sorted(primary["per_seed"])
    files: list[str] = []
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(seeds), dtype=np.float64)
    width = 0.25
    for offset, provider in enumerate(("proxy", "fastsim", "physics")):
        estimates, errors_low, errors_high = [], [], []
        for seed in seeds:
            effect = primary["per_seed"][seed]["providers"][provider]["effects"]["ln_intensity"]
            estimates.append(effect["estimate"])
            interval = effect.get("ci95")
            if interval is None or effect["estimate"] is None:
                errors_low.append(0.0)
                errors_high.append(0.0)
            else:
                errors_low.append(effect["estimate"] - interval[0])
                errors_high.append(interval[1] - effect["estimate"])
        ax.bar(
            x + (offset - 1) * width,
            estimates,
            width,
            yerr=np.asarray([errors_low, errors_high]),
            capsize=3,
            label=provider,
        )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x, seeds)
    ax.set_xlabel("training seed")
    ax.set_ylabel("paired mean Δln intensity (provider-native units)")
    ax.set_title(
        f"Primary contrast {primary['reference_arm']}→{primary['comparison_arm']}"
        f" (label {label}, window {primary['resolution_key']} s)"
    )
    ax.legend()
    files.extend(save(fig, output, "primary-provider-ln-intensity-delta"))

    scenarios = sorted(
        {p["scenario"] for seed in seeds for p in primary["per_seed"][seed]["pairs"]}
    )
    values: list[list[float | None]] = [[None] * len(seeds) for _ in scenarios]
    for column, seed in enumerate(seeds):
        by_scenario = {
            p["scenario"]: p
            for p in primary["per_seed"][seed]["pairs"]
            if p["available"]["fastsim"]
        }
        for row, scenario in enumerate(scenarios):
            pair = by_scenario.get(scenario)
            values[row][column] = None if pair is None else pair["deltas"]["fastsim_ln_intensity"]
    files.extend(
        heatmap(
            output,
            "primary-scenario-fastsim-ln-intensity-delta",
            values,
            scenarios,
            seeds,
            f"Per-scenario FASTSim Δln Wh/km ({primary['reference_arm']}→"
            f"{primary['comparison_arm']})",
            number_format=".3g",
        )
    )
    return files


def _census_section(census: dict[str, Any], lines: list[str]) -> None:
    for key, row in census.items():
        lines += [
            f"## Replay census — window {key} s",
            "",
            f"- Measured episodes: {row['measured_episode_count']}; followed:"
            f" **{row['followed_episode_count']}**; unfollowed:"
            f" {row['unfollowed_episode_count']}; unavailable (no completed trace):"
            f" {row['unavailable_episode_count']}",
            f"- Max window-distance relative residual:"
            f" {_fmt(row['max_window_distance_relative_residual'])}",
            f"- Mean FASTSim wall time per followed episode:"
            f" {_fmt(row['mean_fastsim_elapsed_ms'])} ms",
            f"- Per arm/seed: {json.dumps(row['per_arm_seed'])}",
            "",
        ]
        if row["unfollowed_episodes"]:
            lines += ["Unfollowed episodes (fail-closed; excluded from FASTSim statistics):", ""]
            for item in row["unfollowed_episodes"]:
                message = (item["failure_message"] or "").replace("\n", " ")[:220]
                lines.append(
                    f"- {item['arm']}/seed_{item['training_seed']}/{item['scenario']}: {message}"
                )
            lines.append("")


def _contrast_section(name: str, contrast: dict[str, Any], lines: list[str]) -> None:
    title_arm = (
        f"{contrast['reference_arm']} → {contrast['comparison_arm']}"
        if contrast["reference_has_training_seed"]
        else f"{contrast['reference_arm']} (frozen) → {contrast['comparison_arm']}"
    )
    descriptive = contrast["descriptive_only"]
    lines += [
        f"## {name}: {title_arm} (label {contrast['checkpoint_label']},"
        f" window {contrast['resolution_key']} s"
        + (", descriptive only" if descriptive else "")
        + ")",
        "",
        "Difference direction: `comparison - reference`; per-provider paired Δln of"
        " provider-native intensity (proxy: mL/km, FASTSim/physics: Wh/km). Provider"
        " units are never mixed or cross-calibrated.",
        "",
        "| Seed | proxy pairs | Δln proxy | FASTSim pairs | Δln fastsim | physics pairs"
        " | Δln physics | Δdistance m | Δmean speed m/s |",
        "|---|---:|---|---:|---|---:|---|---:|---:|",
    ]
    for seed, row in contrast["per_seed"].items():
        cells = []
        for provider in ("proxy", "fastsim", "physics"):
            view = row["providers"][provider]
            effect = view["effects"]["ln_intensity"]
            cells.append(f"{view['available_pair_count']}/{row['pair_count']}")
            cells.append(
                f"{_fmt(effect['estimate'])} {_ci(effect)}"
                if not descriptive
                else _fmt(effect["estimate"])
            )
        shared = row["shared"]["effects"]
        lines.append(
            f"| {seed} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} | {cells[4]}"
            f" | {cells[5]} | {_fmt(shared['distance_m']['estimate'])}"
            f" | {_fmt(shared['time_mean_speed_mps']['estimate'])} |"
        )
    if not descriptive:
        lines += [
            "",
            "Per-training-seed scenario bootstrap CIs (0.95, 10000 resamples, percentile)"
            " describe scenario uncertainty only; training seeds are never resampled.",
            "",
            "### Cross-seed consistency per provider",
            "",
            "| Provider | per-seed Δln | mean | all seeds same non-zero sign |",
            "|---|---|---:|---|",
        ]
        for metric, row in contrast["across_seeds"].items():
            if not metric.endswith("_ln_intensity"):
                continue
            estimates = ", ".join(
                f"{seed}: {_fmt(value)}" for seed, value in row["per_seed_estimates"].items()
            )
            lines.append(
                f"| {metric.removesuffix('_ln_intensity')} | {estimates} |"
                f" {_fmt(row['mean_of_estimates'])} | {row['all_seeds_same_nonzero_sign']} |"
            )
        lines += [
            "",
            "### Family split (mean Δln per provider, S / SC)",
            "",
            "| Seed | proxy S | proxy SC | fastsim S | fastsim SC | physics S | physics SC |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for seed, row in contrast["per_seed"].items():
            values = []
            for provider in ("proxy", "fastsim", "physics"):
                families = row["providers"][provider]["family_split"]
                values.append(_fmt(families.get("S", {}).get("mean")))
                values.append(_fmt(families.get("SC", {}).get("mean")))
            lines.append(f"| {seed} | " + " | ".join(values) + " |")
    lines.append("")


def _diagnostics_section(diagnostics: dict[str, Any], lines: list[str]) -> None:
    lines += [
        "## Synthetic diagnostic profiles (separate from the real held-out results)",
        "",
        "All profiles start from rest with feasible accelerations and replay through"
        " the unmodified domain provider (stock cold-start vehicle).",
        "",
        "| Profile | Steps | time-mean m/s | dist m | proxy mL | proxy mL/km"
        " | FASTSim J | FASTSim Wh/km | physics J | physics Wh/km |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in diagnostics["profiles"].items():
        lines.append(
            f"| {name} | {row['step_count']} | {_fmt(row['time_mean_speed_mps'])}"
            f" | {_fmt(row['distance_m'])} | {_fmt(row['proxy']['total_ml'])}"
            f" | {_fmt(row['proxy']['ml_per_km'])} | {_fmt(row['fastsim']['fuel_j'])}"
            f" | {_fmt(row['fastsim']['wh_per_km'])} | {_fmt(row['physics']['fuel_j'])}"
            f" | {_fmt(row['physics']['wh_per_km'])} |"
        )
    lines += ["", "### Provider-response invariants", ""]
    for name, row in diagnostics["invariants"].items():
        lines.append(f"- **{name}**: {json.dumps(row)}")
    lines.append("")


def write_report(result: dict[str, Any], output: Path, files: list[str]) -> None:
    lines = [
        "# Fixed-trace offline physics review (FASTSim replay + physics proxy)",
        "",
        f"Study: `{result['study_name']}`; status: **{result['status']}**.",
        "",
        "Offline diagnostic over the E-060 matched execution artifacts (Issue #113"
        " Task 2). Every provider consumes the same distance-preserving"
        " window-mean reconstruction of the identical executed trace; FASTSim"
        " replay failures are fail-closed and reported, never substituted.",
        "",
        "Cycle reconstruction: warm-engine vehicle copy, ramp prefix excluded by"
        " per-step cumulative fuel accounting, flat road at the pinned ambient"
        " conditions. Raw 0.1 s traces are kinematic waypoint outputs and are"
        " physically unfollowable (sub-step acceleration artifacts up to ~47 m/s²).",
        "",
    ]
    _census_section(result["census"], lines)
    if result["status"] == "completed":
        for contrasts in result["contrasts"].values():
            _contrast_section("Primary contrast", contrasts["primary"], lines)
            _contrast_section("Context contrast", contrasts["context"], lines)
            _contrast_section(
                "Secondary resolution (primary contrast)", contrasts["secondary_descriptive"], lines
            )
        _diagnostics_section(result["diagnostics"], lines)
        lines += ["## Gate P2 inputs", "", "```json"]
        lines.append(json.dumps(result["gate_p2_inputs"], indent=2))
        lines += ["```", ""]
    for name in files:
        if name.endswith(".png"):
            lines += [f"![{Path(name).stem}](<{name}>)", "", f"[SVG](<{name[:-4]}.svg>)", ""]
    lines += [
        "",
        "Within-provider paired relative differences only; proxy mL/km is never"
        " numerically compared with FASTSim or physics Wh/km. Sign agreement is"
        " evidence of offline-model support under the selected FASTSim vehicle"
        " configuration, not real-vehicle evidence. This study does not retrain"
        " policies, does not rerun the simulator and does not modify the E-060"
        " Gate F historical verdict.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
