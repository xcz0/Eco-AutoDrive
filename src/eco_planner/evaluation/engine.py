"""Fixed-seed Diffusion Planner closed-loop evaluation orchestration."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import torch
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf

from eco_planner.artifacts import collect_repository_metadata, write_json, write_tracked_diff
from eco_planner.models import GuidanceConfig, SamplerReport
from eco_planner.runtime.fabric import InferenceRuntimeReport, resolve_runtime_settings

from .artifacts import JobSummary, RuntimeMetadata
from .config import EvaluationJobConfig
from .episodes import run_scenario, run_vector_scenarios
from .inference import DiffusionEvaluationAgent, EvaluationAgent, create_fabric_inference_runtime


@dataclass(frozen=True)
class ExecutionReport:
    """Resolved orchestration and process-resource settings for one Hydra job."""

    mode: str
    launcher: str
    worker_count: int
    vector_env_slots: int | None
    torch_threads_per_worker: int | None
    deterministic: bool
    resolved_accelerator: str
    process_id: int
    logical_cpu_count: int
    resource_profile: str | None


def run_evaluation(config: EvaluationJobConfig, output_dir: Path) -> JobSummary:
    """Run all configured scenarios and write reproducible artifacts."""

    args_path = Path(to_absolute_path(config.model.args_path))
    checkpoint_path = Path(to_absolute_path(config.model.checkpoint_path))
    runtime = create_fabric_inference_runtime(
        config.runtime,
        config.sampler,
        config.guidance,
        args_path,
        checkpoint_path,
    )
    return run_evaluation_agent(config, output_dir, DiffusionEvaluationAgent(runtime))


def run_evaluation_agent(
    config: EvaluationJobConfig, output_dir: Path, agent: EvaluationAgent
) -> JobSummary:
    """Execute any planner adapter through the shared evaluation artifact path."""

    started = perf_counter()
    execution = configure_job_execution(config)
    scenarios = config.scenarios
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(
        OmegaConf.create(config.model_dump(mode="python")),
        output_dir / "resolved_config.yaml",
        resolve=True,
    )
    vector_slots = execution.vector_env_slots
    if vector_slots is None:
        summaries = [
            run_scenario(spec, agent, config, output_dir, scenario_index=index)
            for index, spec in enumerate(scenarios)
        ]
    else:
        summaries = run_vector_scenarios(
            scenarios,
            agent,
            config,
            output_dir,
            vector_env_slots=vector_slots,
            torch_threads_per_worker=execution.torch_threads_per_worker,
        )
    summary = JobSummary.model_validate(
        {
            "status": (
                "failed"
                if any(episode.status == "failed" for episode in summaries)
                else "completed"
            ),
            "runtime": asdict(agent.report),
            "checkpoint": asdict(agent.checkpoint_report),
            "sampler": asdict(agent.sampler_report),
            "guidance": asdict(agent.guidance_config),
            "workload": {
                "mode": config.evaluation.mode,
                "profile": config.evaluation.profile,
                "history_warmup_steps": config.evaluation.history_warmup_steps,
                "evaluated_horizon_steps": config.evaluation.evaluated_horizon_steps,
                "scenarios": tuple(item.model_dump(mode="python") for item in config.scenarios),
                "matrix": (
                    None
                    if config.evaluation.matrix is None
                    else config.evaluation.matrix.model_dump(mode="python")
                ),
                "video_enabled": config.video.enabled,
            },
            "episodes": tuple(summaries),
        }
    )
    write_json(output_dir / "summary.json", summary)
    write_runtime_metadata(
        output_dir,
        agent.report,
        agent.sampler_report,
        agent.guidance_config,
        execution,
        perf_counter() - started,
    )
    return summary


def configure_job_execution(config: EvaluationJobConfig) -> ExecutionReport:
    """Resolve topology against numeric resource capacity and configure this process."""

    execution = config.evaluation.execution
    topology = execution.topology
    resources = config.resources
    if topology != "serial" and resources is None:
        raise ValueError(f"{topology} evaluation requires a resource profile")
    mode = "parallel" if topology == "job_parallel" else "serial"
    launcher = "joblib" if topology == "job_parallel" else "basic"
    workers = (
        resources.evaluation_job_worker_count
        if topology == "job_parallel" and resources is not None
        else 1
    )
    vector_slots = (
        resources.evaluation_vector_env_slots
        if topology == "vector" and resources is not None
        else None
    )
    threads = None if resources is None else resources.torch_threads_per_worker

    settings = resolve_runtime_settings(config.runtime)
    logical_cpus = os.cpu_count()
    if logical_cpus is None or logical_cpus <= 0:
        raise RuntimeError("logical CPU count is unavailable")
    if threads is not None:
        if topology == "job_parallel" and settings.resolved_accelerator == "cpu":
            if workers * threads > logical_cpus:
                raise ValueError(
                    "parallel CPU thread budget exceeds the available logical CPU count"
                )
        torch.set_num_threads(threads)

    if settings.resolved_accelerator == "cuda" and execution.deterministic:
        workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        if workspace not in {None, ":4096:8"}:
            raise ValueError("CUBLAS_WORKSPACE_CONFIG must be ':4096:8'")
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)

    if topology == "job_parallel" and settings.resolved_accelerator == "cuda":
        if torch.cuda.device_count() != 1:
            raise ValueError("CUDA parallel execution requires exactly one visible CUDA GPU")
        if not execution.deterministic:
            raise ValueError("CUDA parallel execution requires deterministic=true")

    return ExecutionReport(
        mode=mode,
        launcher=launcher,
        worker_count=workers,
        vector_env_slots=vector_slots,
        torch_threads_per_worker=threads,
        deterministic=execution.deterministic,
        resolved_accelerator=settings.resolved_accelerator,
        process_id=os.getpid(),
        logical_cpu_count=logical_cpus,
        resource_profile=None if resources is None else resources.name,
    )


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
