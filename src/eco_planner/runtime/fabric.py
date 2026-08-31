"""Lightning Fabric assembly shared by evaluation and policy rollout."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from lightning.fabric import Fabric

from eco_planner.runtime.config import RuntimeConfig


@dataclass(frozen=True)
class InferenceRuntimeReport:
    """Requested and resolved settings persisted with runtime artifacts."""

    requested_accelerator: str
    resolved_accelerator: str
    requested_precision: str
    resolved_precision: str
    device: str
    seed: int
    world_size: int


@dataclass(frozen=True)
class ResolvedRuntimeSettings:
    """Runtime settings resolved against the local CUDA capability."""

    requested_accelerator: str
    resolved_accelerator: str
    requested_precision: str
    resolved_precision: Literal["32-true", "16-mixed", "bf16-mixed"]
    seed: int


def create_single_device_fabric(
    runtime_config: RuntimeConfig, *, configure_cuda_matmul_precision: bool = False
) -> tuple[Fabric, InferenceRuntimeReport]:
    """Resolve settings, seed all RNGs, and create one single-device Fabric."""

    settings = resolve_runtime_settings(runtime_config)
    if configure_cuda_matmul_precision and settings.resolved_accelerator == "cuda":
        torch.set_float32_matmul_precision("high")
    fabric = Fabric(
        accelerator=settings.resolved_accelerator,
        devices=1,
        precision=settings.resolved_precision,
    )
    fabric.seed_everything(settings.seed, workers=True, verbose=False)
    return fabric, InferenceRuntimeReport(
        requested_accelerator=settings.requested_accelerator,
        resolved_accelerator=settings.resolved_accelerator,
        requested_precision=settings.requested_precision,
        resolved_precision=settings.resolved_precision,
        device=str(fabric.device),
        seed=settings.seed,
        world_size=int(fabric.world_size),
    )


def resolve_runtime_settings(runtime_config: RuntimeConfig) -> ResolvedRuntimeSettings:
    """Resolve the configured Fabric accelerator and precision against available hardware."""

    accelerator = runtime_config.accelerator
    precision = runtime_config.precision
    seed = runtime_config.seed

    cuda_available = torch.cuda.is_available()
    if accelerator == "cuda" and not cuda_available:
        raise RuntimeError("CUDA was explicitly requested but is unavailable")
    resolved_accelerator = (
        "cuda" if accelerator == "cuda" or (accelerator == "auto" and cuda_available) else "cpu"
    )

    bf16_supported = resolved_accelerator == "cuda" and torch.cuda.is_bf16_supported()
    if precision == "auto":
        if resolved_accelerator == "cpu":
            resolved_precision: Literal["32-true", "16-mixed", "bf16-mixed"] = "32-true"
        else:
            resolved_precision = "bf16-mixed" if bf16_supported else "16-mixed"
    else:
        resolved_precision = precision
    if resolved_accelerator == "cpu" and resolved_precision != "32-true":
        raise ValueError("CPU inference requires runtime.precision=32-true or auto")
    if resolved_precision == "bf16-mixed" and not bf16_supported:
        raise RuntimeError("bf16-mixed was requested but the CUDA device does not support BF16")

    return ResolvedRuntimeSettings(
        requested_accelerator=accelerator,
        resolved_accelerator=resolved_accelerator,
        requested_precision=precision,
        resolved_precision=resolved_precision,
        seed=seed,
    )
