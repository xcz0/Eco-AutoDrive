"""Completion, safety and conditional energy evidence for scalar reward comparisons."""

from pathlib import Path

from .fixed import number


def _arms(data: dict, label: str) -> list[tuple[str, dict]]:
    return [("A0", data["baseline"])] + [
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
        "Primary contrast: **A2 - A1**. A1 - A0 and A2 - A0 provide baseline context.",
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
