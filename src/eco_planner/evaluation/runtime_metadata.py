"""Evaluation runtime metadata and tracked-diff persistence."""

from __future__ import annotations

import platform
import subprocess
import sys
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import torch
from hydra.utils import to_absolute_path

from eco_planner.evaluation.artifact_writer import write_json
from eco_planner.evaluation.execution import ExecutionReport
from eco_planner.evaluation.runtime import InferenceRuntimeReport
from eco_planner.evaluation.schema import ARTIFACT_SCHEMA_VERSION, RuntimeMetadata
from eco_planner.models.guidance import GuidanceConfig
from eco_planner.models.sampling import SamplerReport


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
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "git_head": _git_output(repository_root, "rev-parse", "HEAD").strip(),
            "git_status_short": tuple(
                _git_output(repository_root, "status", "--short").splitlines()
            ),
            "platform": platform.platform(),
            "python": sys.version,
            "torch": torch.__version__,
            "lightning": version("lightning"),
            "metadrive": version("metadrive-simulator"),
            "pydantic": version("pydantic"),
            "inference_runtime": asdict(runtime_report),
            "sampler": asdict(sampler_report),
            "guidance": asdict(guidance_config),
            "execution": asdict(execution_report),
            "elapsed_seconds": elapsed_seconds,
            "cuda_memory": _cuda_memory_report(runtime_report),
        }
    )
    write_json(output_dir / "runtime_metadata.json", metadata)
    (output_dir / "tracked_diff.patch").write_text(
        _git_output(repository_root, "diff", "--binary", "--no-ext-diff"), encoding="utf-8"
    )


def _cuda_memory_report(runtime_report: InferenceRuntimeReport) -> dict[str, int] | None:
    if runtime_report.resolved_accelerator != "cuda":
        return None
    return {
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def _git_output(repository_root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout
