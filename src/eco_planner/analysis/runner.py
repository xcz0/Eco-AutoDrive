"""Explicit experiment dispatch and shared report publication."""

from pathlib import Path
from typing import Any

from .fixed_batch import calibration, recompute
from .io import read_json, write_json
from .reporting.markdown import write_report

EXPERIMENTS = (
    "lambda-identifiability",
    "reward-calibration",
    "objective-decomposition",
    "critic-gae-ablation",
    "scalar-reward",
    "energy-sweep",
    "ppo-stability",
    "reward-sanity",
    "ppo-reproducibility",
    "execution-backend",
)


def analyze(
    experiment: str,
    source: Path,
    output: Path,
    *,
    figures: bool = True,
    comparison_config: Path | None = None,
    source_file: Path | None = None,
) -> dict[str, Any]:
    source, output = source.resolve(strict=True), output.resolve()
    if source == output or source.is_relative_to(output) or output.is_relative_to(source):
        raise ValueError("offline output must be separate from the source directory")
    return publish(
        experiment,
        source,
        output,
        figures=figures,
        comparison_config=comparison_config,
        source_file=source_file,
    )


def publish(
    experiment: str,
    source: Path,
    output: Path,
    *,
    figures: bool = True,
    comparison_config: Path | None = None,
    source_file: Path | None = None,
) -> dict[str, Any]:
    """Also used by experiment writers, after all original artifacts and guards are complete."""
    study = None
    seed = 0
    if experiment in ("lambda-identifiability", "objective-decomposition", "critic-gae-ablation"):
        data = recompute(source)
    elif experiment == "reward-calibration":
        data = calibration(source)
    elif experiment == "energy-sweep":
        from .evaluation import energy_sweep

        data = energy_sweep(source)
    elif experiment == "scalar-reward":
        from .evaluation import scalar_reward

        if comparison_config is None:
            raise ValueError("scalar-reward analysis requires --comparison-config")
        data = scalar_reward(comparison_config.resolve(strict=True))
    elif experiment == "ppo-stability":
        from .stability import analyze as stability_analysis

        data, study, seed = stability_analysis(source)
    elif experiment == "reward-sanity":
        data = read_json(source / "sanity_report.json")
    elif experiment == "ppo-reproducibility":
        from .simple import reproducibility

        data = reproducibility(source)
    elif experiment == "execution-backend":
        from .simple import execution

        data = execution(source_file or source / "evaluation_modes.json")
    else:
        raise ValueError(f"unsupported experiment: {experiment}")
    output.mkdir(parents=True, exist_ok=True)
    files = []
    if figures:
        import matplotlib.pyplot as plt

        with plt.style.context("default"):
            from .reporting.experiments import experiment_figures

            files = experiment_figures(experiment, data, output)
            if study is not None:
                from .stability import figures as stability_figures

                files += stability_figures(data, study, seed, output)
    payload = {
        "experiment": experiment,
        "source": str(source.resolve()),
        "evidence": data,
        "figures": files,
    }
    write_json(output / "analysis.json", payload)
    write_report(experiment, source, output, data, files)
    return payload


def publish_scalar_run(source: Path, *, training: bool, figures: bool = True) -> None:
    """Single-run presentation; cross-arm comparisons require explicit input grouping."""
    from .evaluation import episode_records
    from .reporting.experiments import scalar_run_figures

    if training:
        from eco_planner.rl.artifacts.summaries import TrainingRunSummary

        summary = TrainingRunSummary.model_validate_json(
            (source / "summary.json").read_text(encoding="utf-8")
        )
        data = summary.model_dump(mode="json")
    else:
        from eco_planner.evaluation.artifacts.io import load_job_summary

        rows = episode_records(load_job_summary(source / "summary.json"))
        data = {"episodes": rows}
    files = scalar_run_figures(data, source, training=training) if figures else []
    write_json(source / "analysis.json", {"evidence": data, "figures": files})
    write_report("scalar-reward", source, source, data, files)
