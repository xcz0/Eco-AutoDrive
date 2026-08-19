"""Evaluation runtime metadata and tracked-diff persistence."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch
from hydra.utils import to_absolute_path

from eco_planner.artifacts import collect_repository_metadata, write_tracked_diff
from eco_planner.evaluation.artifacts.io import write_json
from eco_planner.evaluation.artifacts.models import RuntimeMetadata
from eco_planner.evaluation.runtime.engine import InferenceRuntimeReport
from eco_planner.evaluation.runtime.resources import ExecutionReport
from eco_planner.models import GuidanceConfig, SamplerReport


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
