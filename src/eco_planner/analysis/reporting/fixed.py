from typing import Any


def number(value: float | None) -> str:
    return "undefined" if value is None else f"{value:.9g}"


def render_report(summary: dict[str, Any], *, batch_origin: str = "New batch") -> str:
    lines = [
        "# Lambda identifiability: fixed update-0 batch",
        "",
        f"{batch_origin}; actor objective only; no optimizer steps. No automatic gate threshold.",
        "",
        summary["undefined_reason"],
        "",
        "Component std uses population variance; advantage std uses sample variance.",
        "",
        "## Reward: component scale",
        "",
        "| Component | Mean | Std | Quantiles |",
        "| --- | ---: | ---: | --- |",
    ]
    for name, stats in summary["components"].items():
        lines.append(
            f"| {name} | {number(stats['mean'])} | {number(stats['std'])} | {stats['quantiles']} |"
        )
    lines += [
        "",
        "## Credit assignment to actor gradient",
        "",
        "| Lambda | Raw A mean | Raw A std | Norm A std | Head norm | Trunk norm |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm in summary["arms"]:
        lines.append(
            f"| {arm['lambda']:g} | {number(arm['raw_advantage']['mean'])} | "
            f"{number(arm['raw_advantage']['std'])} | "
            f"{number(arm['normalized_advantage']['std'])} | "
            f"{number(arm['gradient_norms']['actor_head'])} | "
            f"{number(arm['gradient_norms']['shared_trunk'])} |"
        )
    lines += [
        "",
        "| Lambda i → j | Pearson | Spearman | Sign flip | Head cosine | Norm j/i |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for pair in summary["pairs"]:
        grad = pair["gradients"]["actor_head"]
        lines.append(
            f"| {pair['lambda_i']:g} → {pair['lambda_j']:g} | {pair['pearson']} | "
            f"{pair['spearman']} | {pair['sign_flip_fraction']} | "
            f"{grad['cosine']} | {grad['norm_ratio_j_over_i']} |"
        )
    lines += [
        "",
        "Full arm statistics, dimension-specific gradients and per-scenario matched "
        "differences: [summary.json](summary.json). Per-transition values and all gradient "
        "vectors: [diagnostics.npz](diagnostics.npz), indexed by "
        "[sample_index.json](sample_index.json).",
        "",
        "These measurements concern this batch and initial policy only. They do not "
        "establish learned behavioral separation or select Task B/C.",
        "",
    ]
    return "\n".join(lines)


def render_decomposition_report(summary: dict[str, Any]) -> str:
    gate = summary["gate"]
    lines = [
        "# Task C: objective decomposition + normalization attribution",
        "",
        "Reused fixed source batch; actor objective only; no optimizer steps. "
        "Endpoint and stress comparisons use the actor-head gradient.",
        "",
        summary["undefined_reason"],
        "",
        "Component std uses population variance; advantage std uses sample variance.",
        "",
        "| Arm | Reward mean | Reward std | Raw A mean | Raw A std | Z A std | "
        "Head grad norm raw / center / z |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for arm in summary["arms"]:
        norms = arm["gradient_norms"]
        lines.append(
            f"| {arm['label']} | {number(arm['reward']['mean'])} | "
            f"{number(arm['reward']['std'])} | "
            f"{number(arm['raw_advantage']['mean'])} | {number(arm['raw_advantage']['std'])} | "
            f"{number(arm['normalized_advantage']['std'])} | "
            f"{number(norms['raw']['actor_head'])} / {number(norms['center']['actor_head'])} / "
            f"{number(norms['z']['actor_head'])} |"
        )
    endpoint = next(
        pair
        for pair in summary["pairs"]
        if pair["arm_i"] == "r0" and pair["arm_j"] == "energy_only"
    )
    lines += [
        "",
        "## Credit assignment to actor gradient: R0 vs Energy-only",
        "",
        "| Advantage form | Pearson | Spearman | Sign flip | Advantage RMSE | "
        "Head cosine | Head norm ratio |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for form, entry in (
        ("z (current preprocessing)", endpoint),
        ("raw", summary["endpoint_forms"]["raw"]),
        ("center-only", summary["endpoint_forms"]["center"]),
    ):
        head = entry["gradients"]["actor_head"]
        rmse = entry.get("normalized_advantage_rmse", entry.get("advantage_rmse"))
        lines.append(
            f"| {form} | {entry['pearson']} | {entry['spearman']} | "
            f"{entry['sign_flip_fraction']} | {number(rmse)} | {head['cosine']} | "
            f"{head['norm_ratio_j_over_i']} |"
        )
    lines += [
        "",
        "## Actor gradient response: R0 -> finite lambda -> Energy-only",
        "",
        "| Lambda | Head cosine vs R0 | Angular separation (rad) | "
        "Fraction of endpoint separation | Head norm ratio |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for entry in gate["stress"]:
        lines.append(
            f"| {entry['lambda']:g} | {number(entry['actor_head_cosine'])} | "
            f"{number(entry['angular_separation_rad'])} | "
            f"{number(entry['fraction_of_endpoint_separation'])} | "
            f"{number(entry['norm_ratio_j_over_i'])} |"
        )
    verdict = "PASSED" if gate["gate_c_passed"] else "FAILED"
    attribution = gate["attribution"] or "none"
    lines += [
        "",
        "## Gate C",
        "",
        f"Verdict: **{verdict}**; attribution: `{attribution}`.",
        "",
        f"Endpoint actor-head cosine: {number(gate['endpoint']['actor_head_cosine'])} "
        f"(threshold {gate['thresholds']['endpoint_max_actor_head_cosine']}); "
        f"normalized-advantage RMSE: {number(gate['endpoint']['normalized_advantage_rmse'])} "
        f"(threshold {gate['thresholds']['min_normalized_advantage_rmse']}); "
        f"sign-flip fraction: {number(gate['endpoint']['sign_flip_fraction'])} "
        f"(threshold {gate['thresholds']['min_sign_flip_fraction']}).",
        "",
    ]
    if gate["failure_reasons"]:
        lines.append("Failure reasons:")
        lines.extend(f"- {reason}" for reason in gate["failure_reasons"])
        lines.append("")
    lines += [
        "Full arm and pair statistics, per-form gradients and per-scenario matched "
        "differences: [summary.json](summary.json). Per-transition values and all gradient "
        "vectors: [diagnostics.npz](diagnostics.npz), indexed by "
        "[sample_index.json](sample_index.json).",
        "",
        "These measurements concern this batch and initial policy only. They do not "
        "establish learned behavioral separation, do not select a training reward, and do "
        "not run any optimizer step.",
        "",
    ]
    return "\n".join(lines)


def render_ablation_report(summary: dict[str, Any]) -> str:
    attribution = summary["attribution"]
    lines = [
        "# Task C4: critic / GAE common-term ablation",
        "",
        "Reused the fixed E-035 source batch, initial policy, actions, old log-probs and "
        "episode boundaries. R0 vs Energy-only under three temporal-credit forms; actor "
        "objective only; no optimizer steps. Primary comparisons use the full-batch "
        "z-normalized signal; raw / center-only rows are attribution aids only.",
        "",
        summary["undefined_reason"],
        "",
        "Advantage std uses sample variance.",
        "",
        "## Credit assignment to actor gradient",
        "",
        "| Arm | Credit form | Raw A mean | Raw A std | Z A std | Head grad norm (z) |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for arm in summary["arms"]:
        for form, entry in arm["credit_forms"].items():
            lines.append(
                f"| {arm['label']} | {form} | {number(entry['raw_advantage']['mean'])} | "
                f"{number(entry['raw_advantage']['std'])} | "
                f"{number(entry['normalized_advantage']['std'])} | "
                f"{number(entry['gradient_norms']['z']['actor_head'])} |"
            )
    lines += [
        "",
        "## Endpoint signal: R0 vs Energy-only (z-normalized)",
        "",
        "| Credit form | Pearson | Spearman | Sign flip | Z-adv RMSE |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for pair in summary["pairs"]:
        lines.append(
            f"| {pair['credit_form']} | {number(pair['pearson'])} | {number(pair['spearman'])} | "
            f"{number(pair['sign_flip_fraction'])} | "
            f"{number(pair['normalized_advantage_rmse'])} |"
        )
    lines += [
        "",
        "## Endpoint actor gradients (z-normalized)",
        "",
        "| Credit form | Head cosine | Head norm ratio | Lateral cosine | "
        "Longitudinal cosine | Trunk cosine |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for pair in summary["pairs"]:
        gradients = pair["gradients"]

        def cell(group: str, key: str, values: dict = gradients) -> str:
            value = values[group][key]
            return "undefined" if value is None else f"{number(value)}"

        lines.append(
            f"| {pair['credit_form']} | {cell('actor_head', 'cosine')} | "
            f"{cell('actor_head', 'norm_ratio_j_over_i')} | {cell('lateral', 'cosine')} | "
            f"{cell('longitudinal', 'cosine')} | {cell('shared_trunk', 'cosine')} |"
        )
    lines += [
        "",
        "## Attribution aids: raw / center-only advantage forms",
        "",
        "| Credit form | Advantage form | Pearson | Spearman | Sign flip | RMSE | Head cosine |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for pair in summary["pairs"]:
        for advantage_form, entry in pair["advantage_forms"].items():
            head = entry["gradients"]["actor_head"]["cosine"]
            lines.append(
                f"| {pair['credit_form']} | {advantage_form} | {number(entry['pearson'])} | "
                f"{number(entry['spearman'])} | {number(entry['sign_flip_fraction'])} | "
                f"{number(entry['advantage_rmse'])} | "
                f"{'undefined' if head is None else f'{number(head)}'} |"
            )
    verdict = (
        "PASSED" if attribution["gate_c_endpoint_identifiable_under_standard_gae"] else "FAILED"
    )
    lines += [
        "",
        "## C4 attribution",
        "",
        f"Standard-GAE endpoint identifiability (Gate C endpoint thresholds): **{verdict}**.",
        "",
        f"C4 attribution: `{attribution['attribution'] or 'none'}`.",
        "",
    ]
    for form in summary["credit_forms"]:
        entry = attribution["endpoint"][form]
        lines.append(
            f"- {form}: head cosine {number(entry['actor_head_cosine'])}, "
            f"z-adv RMSE {number(entry['normalized_advantage_rmse'])}, "
            f"sign flip {number(entry['sign_flip_fraction'])}, "
            f"identifiable = {entry['identifiable']}."
        )
    lines += [
        "",
        "Full arm and pair statistics with per-form gradients: [summary.json](summary.json). "
        "Per-transition values and all gradient vectors: [diagnostics.npz](diagnostics.npz), "
        "indexed by [sample_index.json](sample_index.json).",
        "",
        "This is a pure offline attribution of the E-035 Gate C failure. It does not modify "
        "the PPO training definition, does not establish learned behavior, and Task D "
        "(guidance control authority) follows regardless of the attribution branch.",
        "",
    ]
    return "\n".join(lines)
