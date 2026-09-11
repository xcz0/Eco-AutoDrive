"""Explicit experiment dispatch and shared report publication."""

from pathlib import Path
from typing import Any

from .evaluation import ScalarComparison
from .fixed_batch import calibration, recompute
from .io import read_json, write_json
from .reporting import write_report


def analyze(
    experiment: str,
    source: Path,
    output: Path,
    *,
    figures: bool = True,
    scalar_comparison: ScalarComparison | None = None,
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
        scalar_comparison=scalar_comparison,
        source_file=source_file,
    )


def publish(
    experiment: str,
    source: Path,
    output: Path,
    *,
    figures: bool = True,
    scalar_comparison: ScalarComparison | None = None,
    source_file: Path | None = None,
) -> dict[str, Any]:
    """Also used by experiment writers, after all original artifacts and guards are complete."""
    study = None
    seed = 0
    if experiment == "guidance-control-authority":
        from .guidance import recompute as guidance_analysis

        data, episodes = guidance_analysis(source)
    elif experiment in ("lambda-identifiability", "objective-decomposition", "critic-gae-ablation"):
        data = recompute(source)
    elif experiment == "reward-calibration":
        data = calibration(source)
    elif experiment == "energy-sweep":
        from .evaluation import energy_sweep

        data = energy_sweep(source)
    elif experiment == "scalar-reward":
        from .evaluation import scalar_reward

        if scalar_comparison is None:
            raise ValueError("scalar-reward analysis requires a validated comparison")
        data = scalar_reward(scalar_comparison)
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
    if study is not None:
        write_json(output / "stage-a-summary.json", data["search_summary"])
    if experiment == "guidance-control-authority":
        write_json(output / "summary.json", {"status": "completed", **data})
    files = []
    if figures:
        from .reporting.plots import plt

        with plt.style.context("default"):
            from .reporting.experiments import experiment_figures

            if experiment == "guidance-control-authority":
                from .reporting.guidance import plot

                files = plot(data, episodes, output)
            else:
                files = experiment_figures(experiment, data, output)
            if study is not None:
                from .reporting.stability import figures as stability_figures

                rendered = stability_figures(data, study, seed, output)
                files += rendered.files
                data["unavailable_figures"] = rendered.unavailable
    payload = _write_analysis(output, data, files, experiment=experiment, source=source)
    if experiment == "guidance-control-authority":
        from .reporting.guidance import write_report as write_guidance_report

        write_guidance_report(data, output, files)
        return {"status": "completed", "output_dir": str(output), "gate_d": data["gate_d"]}
    write_report(experiment, source, output, data, files)
    return payload


def _write_analysis(
    output: Path,
    data: dict[str, Any],
    files: list[str],
    *,
    experiment: str | None = None,
    source: Path | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"evidence": data, "figures": files}
    if experiment is not None:
        payload["experiment"] = experiment
    if source is not None:
        payload["source"] = str(source.resolve())
    write_json(output / "analysis.json", payload)
    return payload


def publish_scalar_run(source: Path, *, training: bool, figures: bool = True) -> None:
    """Single-run presentation; cross-arm comparisons require explicit input grouping."""
    from .evaluation import episode_records

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
    files = []
    if figures:
        from .reporting.experiments import scalar_run_figures
        from .reporting.plots import plt

        with plt.style.context("default"):
            files = scalar_run_figures(data, source, training=training)
    _write_analysis(source, data, files)
    write_report("scalar-reward", source, source, data, files)
