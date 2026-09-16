"""Experiment-specific figure composition from analysis evidence."""

from pathlib import Path


def experiment_figures(experiment: str, data: dict, output: Path) -> list[str]:
    from .plots import curves, heatmap

    if experiment in ("reward", "credit"):
        files = curves(
            output,
            "distributions",
            {
                name: (
                    [float(q) for q in value["all"]["quantiles"]],
                    list(value["all"]["quantiles"].values()),
                )
                for name, value in data["distributions"].items()
                if name.endswith("__reward_total") or name.endswith("__reward")
            },
            "quantile",
            "reward (dimensionless)",
        )
        if experiment == "credit":
            groups = data["recorded"]["gradient_groups"]
            pairs = data["pairs"]
            labels = [
                f"{p['comparison']}-{p['reference']} / {p['credit_form']} / {p['advantage_form']}"
                for p in pairs
            ]
            for metric in ("cosine", "one_minus_cosine", "norm_ratio_j_over_i"):
                values = []
                for pair in pairs:
                    row = []
                    for group in groups:
                        stats = pair["gradients"][group]
                        value = stats["cosine"] if metric == "one_minus_cosine" else stats[metric]
                        row.append(
                            1 - value
                            if metric == "one_minus_cosine" and value is not None
                            else value
                        )
                    values.append(row)
                files += heatmap(output, "gradient-" + metric, values, labels, groups, metric)
        else:
            names = list(data["distributions"])
            files += heatmap(
                output,
                "component-means",
                [[data["distributions"][name]["all"]["mean"]] for name in names],
                names,
                ["mean"],
                "Reward component means (dimensionless)",
            )
        return files
    if experiment == "training":
        runs = data.get("runs", data.get("arms", []))
        series = {}
        for index, record in enumerate(runs):
            metrics = record.get("metrics")
            if metrics is not None:
                values = metrics["post_update_kl"]
                series[str(record.get("label", record.get("source", index)))] = (
                    list(range(len(values))),
                    values,
                )
        if series:
            return curves(output, "post-update-kl", series, "PPO update", "KL (dimensionless)")
        if data["kind"] == "training-evaluation":
            return curves(
                output,
                "completion",
                {
                    str(index): (
                        ["deterministic", *record["stochastic"]],
                        [
                            record["deterministic"]["completion"]["completed_rate"],
                            *[
                                v["outcomes"]["completion"]["completed_rate"]
                                for v in record["stochastic"].values()
                            ],
                        ],
                    )
                    for index, record in enumerate(runs)
                },
                "policy action condition",
                "completed episode fraction",
            )
        return []
    if experiment == "reward-sanity":
        cases = data["cases"]
        keys = list(next(iter(cases.values()))["components"])
        return heatmap(
            output,
            "reward-components",
            [[v["components"][k] for k in keys] for v in cases.values()],
            list(cases),
            keys,
            "Reward components",
            limits=(0, 1),
        )
    if experiment == "execution-backend":
        files = curves(
            output,
            "job-elapsed",
            {
                k: (list(range(len(v["job_elapsed_s"]))), v["job_elapsed_s"])
                for k, v in data["modes"].items()
            },
            "job index",
            "elapsed seconds",
            scatter=True,
        )
        files += curves(
            output,
            "outer-wall",
            {
                "wall time": (
                    list(data["modes"]),
                    [v["outer_wall_s"]["median"] for v in data["modes"].values()],
                )
            },
            "execution mode",
            "wall seconds",
        )
        return files
    if experiment in ("energy-sweep", "scalar-reward"):
        if experiment == "scalar-reward":
            from .scalar import scalar_effect_figures

            files = scalar_effect_figures(data, output)
            files += curves(
                output,
                "training-rewards",
                {
                    f"{r['arm']}-seed-{r['training_seed']}-{r['checkpoint_label']}": (
                        r["training_curve"]["update"],
                        r["training_curve"]["reward"],
                    )
                    for r in data["runs"]
                },
                "PPO update",
                "total reward",
            )
            return files
        comparisons = data["comparisons"]
        files = []
        for label, entry in comparisons.items():
            if "pairs" not in entry:
                continue
            rows = entry["pairs"]
            name = label.replace("/", "-")
            for metric in ("energy_ml", "route_completion"):
                files += curves(
                    output,
                    name + "-" + metric,
                    {
                        side: ([str(r["key"][0]) for r in rows], [r[side][metric] for r in rows])
                        for side in ("reference", "comparison")
                    },
                    "matched scenario",
                    metric,
                )
            files += curves(
                output,
                name + "-energy-progress",
                {
                    side: (
                        [r[side]["route_completion"] for r in rows],
                        [r[side]["energy_ml"] for r in rows],
                    )
                    for side in ("reference", "comparison")
                },
                "route completion",
                "MetaDrive fuel proxy (mL)",
                scatter=True,
            )
        return files
    return []


def scalar_run_figures(data: dict, output: Path, *, training: bool) -> list[str]:
    from .plots import curves

    if training:
        return curves(
            output,
            "training-reward",
            {
                "reward": (
                    [u["update_index"] for u in data["updates"]],
                    [u["total_reward"] for u in data["updates"]],
                )
            },
            "PPO update",
            "total reward",
        )
    return curves(
        output,
        "energy-progress",
        {
            "episodes": (
                [r["route_completion"] for r in data["episodes"]],
                [r["energy_ml"] for r in data["episodes"]],
            )
        },
        "route completion",
        "MetaDrive fuel proxy (mL)",
        scatter=True,
    )
