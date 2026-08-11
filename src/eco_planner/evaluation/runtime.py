"""Single-device Lightning Fabric runtime for closed-loop inference."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
from lightning.fabric import Fabric
from omegaconf import DictConfig
from torch import nn

from eco_planner.models.config import OfficialDiffusionPlannerConfig
from eco_planner.models.pretrained import (
    CheckpointLoadReport,
    load_official_diffusion_planner,
)

_ACCELERATORS = frozenset({"auto", "cpu", "cuda"})
_PRECISIONS = frozenset({"auto", "32-true", "16-mixed", "bf16-mixed"})


@dataclass(frozen=True)
class InferenceRuntimeReport:
    """Requested and resolved execution settings persisted with evaluation artifacts."""

    requested_accelerator: str
    resolved_accelerator: str
    requested_precision: str
    resolved_precision: str
    device: str
    seed: int
    world_size: int


@dataclass(frozen=True)
class _ResolvedRuntimeSettings:
    requested_accelerator: str
    resolved_accelerator: str
    requested_precision: str
    resolved_precision: str
    devices: int
    seed: int


class FabricInferenceRuntime:
    """Own Fabric, the wrapped planner, and inference-time tensor placement."""

    def __init__(
        self,
        fabric: Fabric,
        planner: nn.Module,
        planner_config: OfficialDiffusionPlannerConfig,
        checkpoint_report: CheckpointLoadReport,
        report: InferenceRuntimeReport,
    ) -> None:
        self._fabric = fabric
        self._planner = planner
        self.planner_config = planner_config
        self.checkpoint_report = checkpoint_report
        self.report = report

    @property
    def device(self) -> torch.device:
        return self._fabric.device

    def new_noise_generator(self) -> torch.Generator:
        """Create one persistent per-episode generator from the configured runtime seed."""

        return torch.Generator(device=self.device).manual_seed(self.report.seed)

    def infer(
        self,
        observation: Mapping[str, torch.Tensor],
        generator: torch.Generator,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """Move a raw observation, sample noise, and run one planner forward pass."""

        moved = self._fabric.to_device(dict(observation))
        if not isinstance(moved, dict) or not all(
            isinstance(name, str) and isinstance(value, torch.Tensor)
            for name, value in moved.items()
        ):
            raise TypeError("Fabric must return a string-to-tensor observation mapping")
        device_observation = cast(dict[str, torch.Tensor], moved)
        config = self.planner_config
        noise = torch.randn(
            (1, 1 + config.predicted_neighbor_num, config.future_len, 4),
            dtype=torch.float32,
            device=self.device,
            generator=generator,
        )
        with torch.inference_mode():
            prediction = self._planner(device_observation, noise)
        if not isinstance(prediction, torch.Tensor):
            raise TypeError("Diffusion Planner forward must return a torch.Tensor")
        prediction = prediction.to(dtype=torch.float32)
        expected_shape = (1, 1 + config.predicted_neighbor_num, config.future_len, 4)
        if tuple(prediction.shape) != expected_shape:
            raise RuntimeError(
                f"Diffusion Planner prediction has shape {tuple(prediction.shape)}, "
                f"expected {expected_shape}"
            )
        if prediction.device != self.device or not torch.isfinite(prediction).all():
            raise RuntimeError("Diffusion Planner prediction must be finite on the runtime device")
        return device_observation, noise, prediction


def create_fabric_inference_runtime(
    runtime_config: DictConfig,
    args_path: Path,
    checkpoint_path: Path,
) -> FabricInferenceRuntime:
    """Resolve settings, seed all RNGs, and assemble the frozen planner with Fabric."""

    settings = resolve_runtime_settings(runtime_config)
    fabric = Fabric(
        accelerator=settings.resolved_accelerator,
        devices=settings.devices,
        precision=settings.resolved_precision,
    )
    fabric.seed_everything(settings.seed, workers=True, verbose=False)
    planner, checkpoint_report = load_official_diffusion_planner(args_path, checkpoint_path)
    planner_config = planner.config
    wrapped_planner = fabric.setup_module(planner)
    report = InferenceRuntimeReport(
        requested_accelerator=settings.requested_accelerator,
        resolved_accelerator=settings.resolved_accelerator,
        requested_precision=settings.requested_precision,
        resolved_precision=settings.resolved_precision,
        device=str(fabric.device),
        seed=settings.seed,
        world_size=int(fabric.world_size),
    )
    if report.world_size != 1:
        raise RuntimeError("closed-loop inference requires Fabric world_size=1")
    return FabricInferenceRuntime(
        fabric,
        wrapped_planner,
        planner_config,
        checkpoint_report,
        report,
    )


def resolve_runtime_settings(runtime_config: DictConfig) -> _ResolvedRuntimeSettings:
    """Validate Hydra values and resolve the explicit Fabric accelerator and precision."""

    if not isinstance(runtime_config, DictConfig):
        raise TypeError("runtime configuration must be a DictConfig")
    accelerator = runtime_config.accelerator
    precision = runtime_config.precision
    devices = runtime_config.devices
    seed = runtime_config.seed
    if not isinstance(accelerator, str) or accelerator not in _ACCELERATORS:
        raise ValueError("runtime.accelerator must be one of: auto, cpu, cuda")
    if type(devices) is not int or devices != 1:
        raise ValueError("runtime.devices must be the integer 1 for closed-loop inference")
    if not isinstance(precision, str) or precision not in _PRECISIONS:
        raise ValueError("runtime.precision must be one of: auto, 32-true, 16-mixed, bf16-mixed")
    if type(seed) is not int or seed < 0:
        raise ValueError("runtime.seed must be a non-negative integer")

    cuda_available = torch.cuda.is_available()
    if accelerator == "cuda" and not cuda_available:
        raise RuntimeError("CUDA was explicitly requested but is unavailable")
    resolved_accelerator = (
        "cuda" if accelerator == "cuda" or (accelerator == "auto" and cuda_available) else "cpu"
    )

    bf16_supported = resolved_accelerator == "cuda" and torch.cuda.is_bf16_supported()
    if precision == "auto":
        if resolved_accelerator == "cpu":
            resolved_precision = "32-true"
        else:
            resolved_precision = "bf16-mixed" if bf16_supported else "16-mixed"
    else:
        resolved_precision = precision
    if resolved_accelerator == "cpu" and resolved_precision != "32-true":
        raise ValueError("CPU inference requires runtime.precision=32-true or auto")
    if resolved_precision == "bf16-mixed" and not bf16_supported:
        raise RuntimeError("bf16-mixed was requested but the CUDA device does not support BF16")

    return _ResolvedRuntimeSettings(
        requested_accelerator=accelerator,
        resolved_accelerator=resolved_accelerator,
        requested_precision=precision,
        resolved_precision=resolved_precision,
        devices=devices,
        seed=seed,
    )
