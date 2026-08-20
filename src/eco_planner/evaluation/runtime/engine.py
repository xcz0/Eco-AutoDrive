"""Single-device Lightning Fabric runtime for closed-loop inference."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import torch
from lightning.fabric import Fabric
from torch import nn

from eco_planner.evaluation.artifacts.models import FailurePhase
from eco_planner.evaluation.config import RuntimeConfig
from eco_planner.evaluation.failures import EpisodeFailure
from eco_planner.evaluation.runtime.contracts import (
    DeferredHostTensors,
    HostExecutionResult,
    HostGuidanceDiagnostics,
    HostInferenceResult,
    copy_execution_trajectory,
    defer_host_tensors,
)
from eco_planner.models import (
    CheckpointLoadReport,
    GuidanceConfig,
    NoGuidanceConfig,
    OfficialDiffusionPlannerConfig,
    PlannerInferenceResult,
    SamplerConfig,
    SamplerReport,
    load_official_diffusion_planner,
    sampler_report,
)


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
    seed: int


class HostInferenceDecision:
    """Execution data plus an explicitly resolved full audit result."""

    def __init__(
        self,
        execution: HostExecutionResult,
        resolve_audit: Callable[[], HostInferenceResult],
    ) -> None:
        self._execution = execution
        self._resolve_audit = resolve_audit
        self._audit: HostInferenceResult | None = None

    @property
    def ego_trajectory(self) -> np.ndarray:
        return self._execution.ego_trajectory

    def full_audit(self) -> HostInferenceResult:
        """Wait for and return the complete artifact/replay payload."""

        if self._audit is None:
            self._audit = self._resolve_audit()
        return self._audit

    def __getattr__(self, name: str) -> object:
        return getattr(self.full_audit(), name)


class FabricInferenceRuntime:
    """Own Fabric, the wrapped planner, and inference-time tensor placement."""

    def __init__(
        self,
        fabric: Fabric,
        planner: nn.Module,
        planner_config: OfficialDiffusionPlannerConfig,
        checkpoint_report: CheckpointLoadReport,
        report: InferenceRuntimeReport,
        sampler_report: SamplerReport,
        guidance_config: GuidanceConfig,
    ) -> None:
        self._fabric = fabric
        self._planner = planner
        self.planner_config = planner_config
        self.checkpoint_report = checkpoint_report
        self.report = report
        self.sampler_report = sampler_report
        self.guidance_config = guidance_config

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
    ) -> HostInferenceDecision:
        """Run one planner pass and synchronously transfer only its executable trajectory."""

        raw_observation = dict(observation)
        _validate_artifact_observation_fields(raw_observation, self.planner_config)

        moved = self._fabric.to_device(raw_observation)
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
        if isinstance(self.guidance_config, NoGuidanceConfig):
            with torch.inference_mode():
                result = self._planner(device_observation, noise, generator)
        else:
            with torch.enable_grad():
                result = self._planner(device_observation, noise, generator)
        if not isinstance(result, PlannerInferenceResult):
            raise TypeError("Diffusion Planner forward must return PlannerInferenceResult")
        prediction = result.prediction.detach()
        expected_shape = (1, 1 + config.predicted_neighbor_num, config.future_len, 4)
        if tuple(prediction.shape) != expected_shape:
            raise RuntimeError(
                f"Diffusion Planner prediction has shape {tuple(prediction.shape)}, "
                f"expected {expected_shape}"
            )
        if prediction.device != self.device:
            raise RuntimeError("Diffusion Planner prediction must remain on the runtime device")
        _validate_optional_guidance_result(
            result,
            self.guidance_config,
            expected_shape,
            self.sampler_report.num_steps,
            self.device,
        )
        return _to_host_decision(noise, result, self.device)


def create_fabric_inference_runtime(
    runtime_config: RuntimeConfig,
    sampler_config: SamplerConfig,
    guidance_config: GuidanceConfig,
    args_path: Path,
    checkpoint_path: Path,
) -> FabricInferenceRuntime:
    """Resolve settings, seed all RNGs, and assemble the frozen planner with Fabric."""

    settings = resolve_runtime_settings(runtime_config)
    if settings.resolved_accelerator == "cuda":
        torch.set_float32_matmul_precision("high")
    fabric = Fabric(
        accelerator=settings.resolved_accelerator,
        devices=1,
        precision=settings.resolved_precision,
    )
    fabric.seed_everything(settings.seed, workers=True, verbose=False)
    planner, checkpoint_report = load_official_diffusion_planner(
        args_path,
        checkpoint_path,
        sampler_config,
        guidance_config,
    )
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
        sampler_report(sampler_config),
        guidance_config,
    )


def _validate_optional_guidance_result(
    result: PlannerInferenceResult,
    guidance_config: GuidanceConfig,
    expected_prediction_shape: tuple[int, ...],
    num_steps: int,
    device: torch.device,
) -> None:
    if isinstance(guidance_config, NoGuidanceConfig):
        if any(
            value is not None
            for value in (
                result.reference_prediction,
                result.guidance_action,
                result.guidance_diagnostics,
            )
        ):
            raise RuntimeError("unguided planner returned guidance audit values")
        return
    if result.reference_prediction is None or result.guidance_action is None:
        raise RuntimeError("guided planner must return reference prediction and action")
    if result.guidance_diagnostics is None:
        raise RuntimeError("guided planner must return guidance diagnostics")
    if tuple(result.reference_prediction.shape) != expected_prediction_shape:
        raise RuntimeError("reference prediction shape disagrees with planner prediction")
    if result.reference_prediction.device != device:
        raise RuntimeError("reference prediction must remain on the runtime device")
    if tuple(result.guidance_action.shape) != (expected_prediction_shape[0], 2):
        raise RuntimeError("guidance action must have shape [B, 2]")
    diagnostics = result.guidance_diagnostics
    batch = expected_prediction_shape[0]
    future_len = expected_prediction_shape[2]
    if tuple(diagnostics.longitudinal_target_speed_delta_mps.shape) != (batch, future_len):
        raise RuntimeError("longitudinal guidance target must have shape [B, T]")
    for name in (
        "lateral_objective_delta",
        "longitudinal_objective_delta",
        "applied_gradient_l2",
        "applied_gradient_max_abs",
        "raw_neighbor_gradient_l2",
        "zero_speed_count",
    ):
        value = getattr(diagnostics, name)
        if tuple(value.shape) != (batch, num_steps) or value.device != device:
            raise RuntimeError(f"guidance diagnostic {name} has an invalid shape or device")


def _validate_artifact_observation_fields(
    observation: Mapping[str, torch.Tensor],
    config: OfficialDiffusionPlannerConfig,
) -> None:
    extra_fields = {
        "route_lanes_speed_limit": ((1, config.route_num, 1), torch.float32),
        "route_lanes_has_speed_limit": ((1, config.route_num, 1), torch.bool),
    }
    for name, (shape, dtype) in extra_fields.items():
        value = observation.get(name)
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
            raise ValueError(f"raw observation field {name!r} must have shape {shape}")
        if value.device.type != "cpu" or value.dtype != dtype:
            raise TypeError(f"raw observation field {name!r} has an invalid device or dtype")
        if dtype.is_floating_point and not torch.isfinite(value).all():
            raise ValueError(f"raw observation field {name!r} must be finite")


def _to_host_decision(
    noise: torch.Tensor,
    result: PlannerInferenceResult,
    device: torch.device,
) -> HostInferenceDecision:
    tensors: dict[str, tuple[torch.Tensor, torch.dtype]] = {
        "initial_noise": (noise.detach(), torch.float32),
        "prediction": (result.prediction.detach(), torch.float32),
    }
    if result.reference_prediction is not None:
        tensors["reference_prediction"] = (result.reference_prediction.detach(), torch.float32)
    if result.guidance_action is not None:
        tensors["guidance_action"] = (result.guidance_action.detach(), torch.float32)
    diagnostics = result.guidance_diagnostics
    if diagnostics is not None:
        for name in (
            "lateral_target_offset_m",
            "longitudinal_target_speed_fraction",
            "longitudinal_target_speed_delta_mps",
            "lateral_objective_delta",
            "longitudinal_objective_delta",
            "applied_gradient_l2",
            "applied_gradient_max_abs",
            "raw_neighbor_gradient_l2",
        ):
            tensors[name] = (getattr(diagnostics, name).detach(), torch.float32)
        tensors["zero_speed_count"] = (diagnostics.zero_speed_count.detach(), torch.int64)

    deferred = defer_host_tensors(tensors, device)
    execution = copy_execution_trajectory(result.prediction, device)
    _validate_execution_trajectory(execution.ego_trajectory)
    return HostInferenceDecision(
        execution,
        lambda: _host_result_from_tensors(deferred, diagnostics is not None),
    )


def _host_result_from_tensors(
    deferred: DeferredHostTensors, diagnostics_present: bool
) -> HostInferenceResult:
    host_tensors = deferred.resolve()
    arrays = {name: tensor.numpy() for name, tensor in host_tensors.items()}
    for name, value in arrays.items():
        if value.dtype.kind in "fc" and not np.isfinite(value).all():
            raise EpisodeFailure(
                FailurePhase.INFERENCE,
                RuntimeError(f"Diffusion Planner result {name!r} contains non-finite values"),
            )
    host_diagnostics = (
        None
        if not diagnostics_present
        else HostGuidanceDiagnostics(
            lateral_target_offset_m=arrays["lateral_target_offset_m"],
            longitudinal_target_speed_fraction=arrays["longitudinal_target_speed_fraction"],
            longitudinal_target_speed_delta_mps=arrays["longitudinal_target_speed_delta_mps"],
            lateral_objective_delta=arrays["lateral_objective_delta"],
            longitudinal_objective_delta=arrays["longitudinal_objective_delta"],
            applied_gradient_l2=arrays["applied_gradient_l2"],
            applied_gradient_max_abs=arrays["applied_gradient_max_abs"],
            raw_neighbor_gradient_l2=arrays["raw_neighbor_gradient_l2"],
            zero_speed_count=arrays["zero_speed_count"],
        )
    )
    return HostInferenceResult(
        initial_noise=arrays["initial_noise"],
        prediction=arrays["prediction"],
        reference_prediction=arrays.get("reference_prediction"),
        guidance_action=arrays.get("guidance_action"),
        guidance_diagnostics=host_diagnostics,
    )


def _validate_execution_trajectory(trajectory: np.ndarray) -> None:
    if not np.isfinite(trajectory).all():
        raise EpisodeFailure(
            FailurePhase.INFERENCE,
            RuntimeError("Diffusion Planner ego trajectory contains non-finite values"),
        )


def resolve_runtime_settings(runtime_config: RuntimeConfig) -> _ResolvedRuntimeSettings:
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
        seed=seed,
    )
