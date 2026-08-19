"""Fixed-seed Diffusion Planner closed-loop evaluation orchestration."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from time import perf_counter

from hydra.utils import to_absolute_path
from omegaconf import OmegaConf

from eco_planner.evaluation.artifacts.io import write_json
from eco_planner.evaluation.artifacts.metadata import write_runtime_metadata
from eco_planner.evaluation.artifacts.models import JobSummary
from eco_planner.evaluation.config import EvaluationJobConfig
from eco_planner.evaluation.episode import run_scenario
from eco_planner.evaluation.runtime.engine import (
    create_fabric_inference_runtime,
)
from eco_planner.evaluation.runtime.resources import configure_job_execution


def run_evaluation(config: EvaluationJobConfig, output_dir: Path) -> JobSummary:
    """Run all configured scenarios and write reproducible artifacts."""

    started = perf_counter()
    if not isinstance(config, EvaluationJobConfig):
        raise TypeError("config must be an EvaluationJobConfig")
    execution = configure_job_execution(config)
    scenarios = config.scenarios
    args_path = Path(to_absolute_path(config.model.args_path))
    checkpoint_path = Path(to_absolute_path(config.model.checkpoint_path))
    runtime = create_fabric_inference_runtime(
        config.runtime,
        config.sampler,
        config.guidance,
        args_path,
        checkpoint_path,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(
        OmegaConf.create(config.model_dump(mode="python")),
        output_dir / "resolved_config.yaml",
        resolve=True,
    )
    summaries = [run_scenario(spec, runtime, config, output_dir) for spec in scenarios]
    summary = JobSummary.model_validate(
        {
            "status": (
                "failed"
                if any(episode.status == "failed" for episode in summaries)
                else "completed"
            ),
            "runtime": asdict(runtime.report),
            "checkpoint": asdict(runtime.checkpoint_report),
            "sampler": asdict(runtime.sampler_report),
            "guidance": asdict(runtime.guidance_config),
            "episodes": tuple(summaries),
        }
    )
    write_json(output_dir / "summary.json", summary)
    write_runtime_metadata(
        output_dir,
        runtime.report,
        runtime.sampler_report,
        runtime.guidance_config,
        execution,
        perf_counter() - started,
    )
    return summary
