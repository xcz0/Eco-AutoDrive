"""Validated per-job execution settings for serial and process-parallel evaluation."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch

from eco_planner.evaluation.config import EvaluationJobConfig
from eco_planner.runtime.fabric import resolve_runtime_settings


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


def configure_job_execution(config: EvaluationJobConfig) -> ExecutionReport:
    """Validate orchestration constraints and configure this worker process."""

    execution = config.evaluation.execution
    mode = execution.mode
    launcher = "basic" if mode == "serial" else "joblib"
    workers = 1 if mode == "serial" else config.resources.evaluation_job_worker_count
    threads = execution.torch_threads_per_worker

    settings = resolve_runtime_settings(config.runtime)
    logical_cpus = os.cpu_count()
    if logical_cpus is None or logical_cpus <= 0:
        raise RuntimeError("logical CPU count is unavailable")
    if threads is not None:
        if mode == "parallel" and settings.resolved_accelerator == "cpu":
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

    if mode == "parallel" and settings.resolved_accelerator == "cuda":
        if torch.cuda.device_count() != 1:
            raise ValueError("CUDA parallel execution requires exactly one visible CUDA GPU")
        if not execution.deterministic:
            raise ValueError("CUDA parallel execution requires deterministic=true")

    return ExecutionReport(
        mode=str(mode),
        launcher=launcher,
        worker_count=workers,
        vector_env_slots=execution.vector_env_slots,
        torch_threads_per_worker=threads,
        deterministic=execution.deterministic,
        resolved_accelerator=settings.resolved_accelerator,
        process_id=os.getpid(),
        logical_cpu_count=logical_cpus,
        resource_profile=None if config.resources is None else config.resources.name,
    )
