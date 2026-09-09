"""Experiment-specific figure composition from analysis evidence."""

from pathlib import Path


def experiment_figures(experiment: str, data: dict, output: Path) -> list[str]:
    from .plots import curves, fixed_figures, heatmap

    if experiment in ("lambda-identifiability", "objective-decomposition", "critic-gae-ablation"):
        return fixed_figures(data, output)
    if experiment == "reward-calibration":
        files = fixed_figures(data["original"], output, "original-")
        files += fixed_figures(data["calibrated"], output, "calibrated-")
        keys = list(data["original"]["components"])
        for metric in ("mean", "std"):
            files += heatmap(
                output,
                f"calibration-component-{metric}",
                [
                    [data[label]["components"][k][metric] for k in keys]
                    for label in ("original", "calibrated")
                ],
                ["original", "calibrated"],
                keys,
                f"Component {metric}",
            )
        for key, entry in data["distributions"].items():
            q = entry["all"]["quantiles"]
            order = sorted(q, key=float)
            files += curves(
                output,
                "audit-" + key,
                {key: ([float(v) for v in order], [q[v] for v in order])},
                "quantile",
                key + " (native audit units)",
            )
        return files
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
    if experiment == "ppo-reproducibility":
        return curves(
            output,
            "replay-reward",
            {
                f"seed {r['training_seed']} replay {r['replay_id']}": (r["updates"], r["rewards"])
                for r in data["runs"]
            },
            "PPO update",
            "total reward",
        )
    if experiment == "ppo-stability":
        files = []
        for metric in ("total_reward", "mean_approximate_kl", "mean_episode_length"):
            series = {
                label: (entry["update"], entry[metric])
                for label, entry in data["training_curves"].items()
            }
            if series:
                files += curves(output, "training-" + metric, series, "PPO update", metric)
        for stage, entry in data["stages"].items():
            records = entry["records"]
            labels = [f"config {r['config_id']} seed {r['training_seed']}" for r in records]
            if records:
                files += curves(
                    output,
                    f"stage-{stage}-episode-retention",
                    {
                        "training retention": (
                            labels,
                            [r["minimum_episode_length_retention"] for r in records],
                        )
                    },
                    "candidate / seed",
                    "minimum episode length retention",
                )
                files += curves(
                    output,
                    f"stage-{stage}-route-retention",
                    {
                        "evaluation retention": (
                            labels,
                            [
                                r["evaluation"]["route_progress_retention"]
                                if r["evaluation"] is not None
                                else None
                                for r in records
                            ],
                        )
                    },
                    "candidate / seed",
                    "route progress retention",
                )
        return files
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
        comparisons = (
            data["comparisons"]
            if experiment == "energy-sweep"
            else {
                f"{r['arm']}-seed-{r['training_seed']}-{r['checkpoint_label']}": r["comparison"]
                for r in data["runs"]
            }
        )
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
        if experiment == "scalar-reward":
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
    return []


def scalar_run_figures(data: dict, output: Path, *, training: bool) -> list[str]:
    import matplotlib.pyplot as plt

    from .plots import curves

    with plt.style.context("default"):
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
