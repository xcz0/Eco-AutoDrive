"""Completion, safety and conditional energy evidence for scalar reward comparisons."""

from pathlib import Path

from .markdown import number

_PAIRED_REPORT_METRICS = (
    ("energy_ml", "Energy total (mL)"),
    ("energy_distance_m", "Energy distance (m)"),
    ("energy_ml_per_km", "Energy intensity (mL/km)"),
    ("mean_speed_mps", "Mean speed (m/s)"),
    ("distance_m", "Distance (m)"),
    ("stopped_fraction", "Stopped fraction"),
    ("route_completion", "Route completion"),
)


def _checkpoint_order(label: str) -> tuple[int, str]:
    if label == "initial":
        return (0, label)
    if label == "final":
        return (2, label)
    return (1, label)


def _checkpoints(data: dict) -> list[str]:
    return sorted({r["checkpoint_label"] for r in data["runs"]}, key=_checkpoint_order)


def _arms(data: dict, label: str) -> list[tuple[str, dict]]:
    return (
        [(data["baseline_arm"].upper(), data["baseline"])] if data["baseline"] is not None else []
    ) + [
        (f"{r['arm'].upper()} seed {r['training_seed']}", r["outcomes"])
        for r in sorted(data["runs"], key=lambda r: (r["arm"], r["training_seed"]))
        if r["checkpoint_label"] == label
    ]


def render_scalar(data: dict) -> str:
    lines = [data["interpretation"], "", "## 1. Completion / availability", ""]
    lines += [
        "Completed includes normal collision/out-of-road termination; it does not mean arrival.",
        "Arrival and progress use completed episodes only; failed episodes remain unknown.",
        "",
        "| Arm | Completed / total | Completed rate | Failed | Arrival / available | "
        "Arrival rate | Mean route completion |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    arms = _arms(data, "final")
    for name, outcomes in arms:
        c = outcomes["completion"]
        lines.append(
            f"| {name} | {c['completed_count']} / {c['episode_count']} | "
            f"{number(c['completed_rate'])} | {c['failed_count']} | "
            f"{c['arrive_dest_count']} / {c['arrival_denominator']} | "
            f"{number(c['arrive_dest_rate'])} | {number(c['route_completion_mean'])} |"
        )
    lines += [
        "",
        "| Contrast | Training seed | Available pairs / total | Available rate |",
        "| --- | ---: | ---: | ---: |",
    ]
    for contrast, checkpoints in data["contrasts"].items():
        for effect in checkpoints["final"]["effects"]:
            pairs = effect["comparison"]
            counts = (
                f"{pairs['available_pair_count']} / {pairs['pair_count']}"
                if pairs
                else "unavailable"
            )
            rate = number(pairs["available_rate"]) if pairs else "unavailable"
            lines.append(f"| {contrast.upper()} | {effect['training_seed']} | {counts} | {rate} |")
    lines += [
        "",
        "## 2. Safety guardrails",
        "",
        "Denominators include all completed episodes in each arm, before pair filtering.",
        "",
        "| Arm | Available | Unavailable | Collision count / rate | "
        "Out-of-road count / rate | Wrong-direction count / rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, outcomes in arms:
        s = outcomes["safety"]
        cells = [
            f"{s[m]['count']} / {number(s[m]['rate'])}"
            for m in ("collision", "out_of_road", "wrong_direction")
        ]
        lines.append(
            f"| {name} | {s['denominator']} | {s['unavailable_count']} | "
            + " | ".join(cells)
            + " |"
        )
    lines += [
        "",
        "## 3. Paired energy effects",
        "",
        f"Primary contrast: **{next(iter(data['contrasts'])).upper()}**; "
        "contrasts follow the declared protocol.",
        "Each contrast uses its own jointly-completed matched episodes. "
        "Early termination can reduce energy; read completion and safety first.",
        "",
        f"Percentile scenario bootstrap: {100 * data['bootstrap']['confidence_level']:g}% CI, "
        f"{data['bootstrap']['n_resamples']} resamples, RNG seed "
        f"{data['bootstrap']['bootstrap_seed']}. "
        "Direction counts use point-estimate signs, independently of CI overlap with zero.",
        "",
    ]
    for contrast, checkpoints in data["contrasts"].items():
        entry = checkpoints["final"]
        counts = entry["direction_counts"]
        lines += [
            "",
            f"### {contrast.upper()}",
            "",
            f"Lower {counts['lower']}/{counts['total']}; zero {counts['zero']}/{counts['total']}; "
            f"higher {counts['higher']}/{counts['total']}; unavailable "
            f"{counts['unavailable']}/{counts['total']}. "
            + ("**Partial results.**" if entry["partial"] else "All seeds available."),
            "",
            "| Training seed | Mean delta (mL) | Scenario 95% CI (mL) | "
            "Pairs | CI includes zero | Unavailable reason |",
            "| ---: | ---: | --- | ---: | --- | --- |",
        ]
        for effect in entry["effects"]:
            ci = effect["ci95"]
            interval = f"[{number(ci[0])}, {number(ci[1])}]" if ci else "unavailable"
            lines.append(
                f"| {effect['training_seed']} | {number(effect['estimate'])} | "
                f"{interval} | {effect['sample_count']} | "
                f"{effect['ci_crosses_zero'] if ci else 'unavailable'} | "
                f"{effect['unavailable_reason'] or ''} |"
            )
    lines += [
        "",
        "## 4. Behavior / task and energy by checkpoint",
        "",
        "Per-arm aggregates over completed episodes at each declared checkpoint "
        "(checkpoint 0 = `initial`, then any intermediate checkpoint, then `final`). "
        "Speed range spans episode minima/maxima; energy distance is the executed fuel-proxy "
        "trace distance.",
    ]
    for label in _checkpoints(data):
        arms = _arms(data, label)
        if not arms:
            continue
        lines += [
            "",
            f"### Checkpoint `{label}`",
            "",
            "| Arm | Mean speed (m/s) | Speed min-max (m/s) | Distance (m) | Stopped frac | "
            "Route | Arrive / available | Collision / OOR / wrong-dir | Terminated / truncated |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for name, outcomes in arms:
            b, c = outcomes["behavior"], outcomes["completion"]
            s, t = outcomes["safety"], outcomes["termination"]
            lines.append(
                f"| {name} | {number(b['mean_speed_mps'])} | "
                f"{number(b['speed_min_mps'])}-{number(b['speed_max_mps'])} | "
                f"{number(b['distance_m'])} | {number(b['stopped_fraction'])} | "
                f"{number(c['route_completion_mean'])} | "
                f"{c['arrive_dest_count']} / {c['arrival_denominator']} | "
                f"{s['collision']['count']} / {s['out_of_road']['count']} / "
                f"{s['wrong_direction']['count']} | "
                f"{t['terminated_count']} / {t['truncated_count']} |"
            )
        lines += [
            "",
            "| Arm | Energy total (mL) | Energy distance (m) | Energy intensity (mL/km) |",
            "| --- | ---: | ---: | ---: |",
        ]
        for name, outcomes in arms:
            e = outcomes["energy"]
            lines.append(
                f"| {name} | {number(e['total_ml'])} | {number(e['distance_m'])} | "
                f"{number(e['ml_per_km'])} |"
            )
    lines += [
        "",
        "## 5. Paired per-metric effects by checkpoint",
        "",
        "Deltas are comparison - reference on jointly-completed matched episodes. "
        "CIs are scenario bootstrap over available pairs; they condition on each trained "
        "policy and are not uncertainty across training seeds.",
    ]
    for contrast, checkpoints in data["contrasts"].items():
        lines += ["", f"### {contrast.upper()}"]
        for label in sorted(checkpoints, key=_checkpoint_order):
            entry = checkpoints[label]
            lines += [
                "",
                f"Checkpoint `{label}`:",
                "",
                "| Metric | Training seed | Mean delta | Scenario 95% CI | Pairs |",
                "| --- | ---: | ---: | --- | ---: |",
            ]
            for effect in entry["effects"]:
                metrics = effect.get("metrics")
                if metrics is None:
                    lines.append(f"| (unavailable) | {effect['training_seed']} | | | |")
                    continue
                for metric, title in _PAIRED_REPORT_METRICS:
                    m = metrics[metric]
                    ci = m["ci95"]
                    interval = f"[{number(ci[0])}, {number(ci[1])}]" if ci else "unavailable"
                    lines.append(
                        f"| {title} | {effect['training_seed']} | "
                        f"{number(m['estimate'])} | {interval} | {m['sample_count']} |"
                    )
    if any("initial" in v for v in data["contrasts"].values()):
        lines += [
            "",
            "## Initial / update-0 diagnostics",
            "",
            "Excluded from final direction counts; full diagnostic outcomes and pairs "
            "remain in analysis.json.",
            "",
            "| Contrast | Training seed | Mean delta (mL) | Scenario 95% CI (mL) | Pairs |",
            "| --- | ---: | ---: | --- | ---: |",
        ]
        for contrast, checkpoints in data["contrasts"].items():
            for effect in checkpoints.get("initial", {}).get("effects", []):
                ci = effect["ci95"]
                interval = f"[{number(ci[0])}, {number(ci[1])}]" if ci else "unavailable"
                lines.append(
                    f"| {contrast.upper()} | {effect['training_seed']} | "
                    f"{number(effect['estimate'])} | {interval} | {effect['sample_count']} |"
                )
    lines += [
        "",
        "Full episode keys, failed-run reasons and paired evidence: "
        "[analysis.json](analysis.json).",
    ]
    return "\n".join(lines)


def scalar_effect_figures(data: dict, output: Path) -> list[str]:
    import matplotlib.pyplot as plt

    from .plots import save

    files = []
    for contrast, checkpoints in data["contrasts"].items():
        entry = checkpoints["final"]
        effects = entry["effects"]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.axvline(0, color="0.5", linewidth=1, linestyle="--")
        for y, effect in enumerate(effects):
            estimate, ci = effect["estimate"], effect["ci95"]
            if estimate is None:
                ax.text(0.02, y, "unavailable", transform=ax.get_yaxis_transform(), color="0.4")
                continue
            color = f"C{y % 10}"
            if ci is not None:
                ax.hlines(y, ci[0], ci[1], color=color, linewidth=2)
                ax.plot(ci, [y, y], "|", color=color, markersize=10)
            ax.plot(estimate, y, "o", color=color, markersize=7)
            ax.text(
                1.02,
                y,
                f"n={effect['sample_count']}" + ("; CI unavailable" if ci is None else ""),
                transform=ax.get_yaxis_transform(),
                va="center",
            )
        counts = entry["direction_counts"]
        ax.set_yticks(range(len(effects)), [f"seed {e['training_seed']}" for e in effects])
        ax.set_ylim(len(effects) - 0.5, -0.5)
        ax.set_xlabel("Mean paired energy delta (mL); negative = lower MetaDrive fuel proxy")
        ax.set_title(
            f"{contrast.upper()} final: scenario bootstrap 95% CI\n"
            f"Lower {counts['lower']}/{counts['total']}; higher "
            f"{counts['higher']}/{counts['total']}; zero {counts['zero']}; "
            f"unavailable {counts['unavailable']}"
        )
        fig.text(
            0.12,
            -0.02,
            "Conditional on each trained policy and jointly-completed scenarios.",
            fontsize=9,
        )
        files += save(fig, output, contrast + "-seed-effects")
    return files
