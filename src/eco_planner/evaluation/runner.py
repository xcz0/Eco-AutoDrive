"""Fixed-seed Diffusion Planner closed-loop evaluation orchestration."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import torch
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf

from eco_planner.artifacts import collect_repository_metadata, write_tracked_diff
from eco_planner.evaluation.artifacts import JobSummary, RuntimeMetadata, write_json
from eco_planner.evaluation.config import EvaluationJobConfig
from eco_planner.evaluation.execution import run_scenario, run_vector_scenarios
from eco_planner.evaluation.runtime import (
    ExecutionReport,
    configure_job_execution,
    create_fabric_inference_runtime,
)
from eco_planner.models import GuidanceConfig, SamplerReport
from eco_planner.runtime.fabric import InferenceRuntimeReport


def run_evaluation(config: EvaluationJobConfig, output_dir: Path) -> JobSummary:
    """Run all configured scenarios and write reproducible artifacts."""

    started = perf_counter()
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
    vector_slots = config.evaluation.execution.vector_env_slots
    if vector_slots is None:
        summaries = [run_scenario(spec, runtime, config, output_dir) for spec in scenarios]
    else:
        summaries = run_vector_scenarios(scenarios, runtime, config, output_dir)
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


def write_runtime_metadata(
    output_dir: Path,
    runtime_report: InferenceRuntimeReport,
    sampler_report: SamplerReport,
    guidance_config: GuidanceConfig,
    execution_report: ExecutionReport,
    elapsed_seconds: float,
) -> None:
    """Write reproducibility metadata and the possibly empty tracked diff."""

    repository_root = Path(to_absolute_path("."))
    metadata = RuntimeMetadata.model_validate(
        {
            **collect_repository_metadata(repository_root),
            "inference_runtime": asdict(runtime_report),
            "sampler": asdict(sampler_report),
            "guidance": asdict(guidance_config),
            "execution": asdict(execution_report),
            "elapsed_seconds": elapsed_seconds,
            "cuda_memory": _cuda_memory_report(runtime_report),
        }
    )
    write_json(output_dir / "runtime_metadata.json", metadata)
    write_tracked_diff(output_dir / "tracked_diff.patch", repository_root)


def _cuda_memory_report(runtime_report: InferenceRuntimeReport) -> dict[str, int] | None:
    if runtime_report.resolved_accelerator != "cuda":
        return None
    return {
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }
