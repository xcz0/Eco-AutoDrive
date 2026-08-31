"""Evaluation inference runtime and process resource configuration."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import cast

import numpy as np
import torch
from lightning.fabric import Fabric
from tensordict import TensorDict, TensorDictBase
from torch import nn

from eco_planner.envs.array_types import BatchObservation
from eco_planner.evaluation.config import EvaluationJobConfig
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
from eco_planner.runtime.config import RuntimeConfig
from eco_planner.runtime.contracts import HostExecutionResult
from eco_planner.runtime.fabric import (
    InferenceRuntimeReport,
    create_single_device_fabric,
    resolve_runtime_settings,
)
from eco_planner.runtime.host_transfer import (
    DeferredHostTensors,
    copy_execution_trajectory,
    defer_host_tensors,
)


@dataclass(frozen=True)
class BatchInferenceTiming:
    """Synchronized component timings captured only for an explicit benchmark pass."""

    host_to_device_s: float
    execution_s: float
    execution_to_host_s: float


class InferenceDecision:
    """Batched execution data plus an explicitly resolved audit result."""

    def __init__(
        self,
        execution: HostExecutionResult,
        resolve_audit: Callable[[], TensorDictBase],
        timing: BatchInferenceTiming | None = None,
    ) -> None:
        self._execution = execution
        self._resolve_audit = resolve_audit
        self._audit: TensorDictBase | None = None
        self._timing = timing

    @property
    def ego_trajectory(self) -> np.ndarray:
        if self._execution.ego_trajectory.shape[0] != 1:
            raise RuntimeError("ego_trajectory is only available for a batch-one decision")
        return self._execution.ego_trajectory[0]

    @property
    def ego_trajectories(self) -> np.ndarray:
        """Return executable ego trajectories with shape ``[B, T, 4]``."""

        return self._execution.ego_trajectory

    @property
    def timing(self) -> BatchInferenceTiming | None:
        """Return synchronized component timing for an explicit profiled call."""

        return self._timing

    def audit_result(self) -> TensorDictBase:
        """Wait for and return the complete artifact/replay payload."""

        if self._audit is None:
            self._audit = self._resolve_audit()
        return self._audit


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

    def sample_noise(self, generators: Sequence[torch.Generator]) -> torch.Tensor:
        """Draw one standard-normal planner input from each slot-owned RNG stream."""

        config = self.planner_config
        return torch.cat(
            [
                torch.randn(
                    (1, 1 + config.predicted_neighbor_num, config.future_len, 4),
                    dtype=torch.float32,
                    device=self.device,
                    generator=generator,
                )
                for generator in generators
            ]
        )

    def infer(
        self,
        observation: BatchObservation | Mapping[str, torch.Tensor],
        generator: torch.Generator,
    ) -> InferenceDecision:
        """Run one planner pass through the shared batched inference path."""

        return self.infer_batch(observation, self.sample_noise((generator,)), (generator,))

    def infer_batch(
        self,
        observation: BatchObservation | Mapping[str, torch.Tensor],
        standard_normal_noise: torch.Tensor,
        transition_generators: Sequence[torch.Generator | None],
        *,
        profile: bool = False,
    ) -> InferenceDecision:
        """Run a batch with independently owned per-slot diffusion RNG streams."""

        raw_observation = cast(dict[str, torch.Tensor], dict(observation))
        batch = _validate_artifact_observation_fields(raw_observation, self.planner_config)
        config = self.planner_config
        expected_shape = (batch, 1 + config.predicted_neighbor_num, config.future_len, 4)
        if tuple(standard_normal_noise.shape) != expected_shape:
            raise ValueError(
                f"standard_normal_noise has shape {tuple(standard_normal_noise.shape)}, "
                f"expected {expected_shape}"
            )
        if (
            standard_normal_noise.dtype != torch.float32
            or standard_normal_noise.device != self.device
        ):
            raise TypeError("standard_normal_noise must be float32 on the runtime device")
        if len(transition_generators) != batch:
            raise ValueError("transition_generators must contain one generator per batch item")

        h2d_started = perf_counter() if profile else 0.0
        moved = self._fabric.to_device(raw_observation)
        _synchronize_if_cuda(self.device, profile)
        host_to_device_s = perf_counter() - h2d_started if profile else 0.0
        if not isinstance(moved, dict) or not all(
            isinstance(name, str) and isinstance(value, torch.Tensor)
            for name, value in moved.items()
        ):
            raise TypeError("Fabric must return a string-to-tensor observation mapping")
        device_observation = cast(dict[str, torch.Tensor], moved)
        execution_started = perf_counter() if profile else 0.0
        if isinstance(self.guidance_config, NoGuidanceConfig):
            with torch.inference_mode():
                result = self._planner(
                    device_observation, standard_normal_noise, transition_generators
                )
        else:
            with torch.enable_grad():
                result = self._planner(
                    device_observation, standard_normal_noise, transition_generators
                )
        _synchronize_if_cuda(self.device, profile)
        execution_s = perf_counter() - execution_started if profile else 0.0
        prediction = result.prediction.detach()
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
        execution, resolve_audit, execution_to_host_s = _prepare_batch_inference_decision(
            standard_normal_noise,
            result,
            self.device,
            profile=profile,
        )
        timing = (
            BatchInferenceTiming(
                host_to_device_s=host_to_device_s,
                execution_s=execution_s,
                execution_to_host_s=execution_to_host_s,
            )
            if profile
            else None
        )
        return InferenceDecision(execution, resolve_audit, timing)


def create_fabric_inference_runtime(
    runtime_config: RuntimeConfig,
    sampler_config: SamplerConfig,
    guidance_config: GuidanceConfig,
    args_path: Path,
    checkpoint_path: Path,
) -> FabricInferenceRuntime:
    """Resolve settings, seed all RNGs, and assemble the frozen planner with Fabric."""

    fabric, report = create_single_device_fabric(
        runtime_config, configure_cuda_matmul_precision=True
    )
    planner, checkpoint_report = load_official_diffusion_planner(
        args_path,
        checkpoint_path,
        sampler_config,
        guidance_config,
    )
    planner_config = planner.config
    wrapped_planner = fabric.setup_module(planner)
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
) -> int:
    ego_current_state = observation.get("ego_current_state")
    if not isinstance(ego_current_state, torch.Tensor) or ego_current_state.ndim < 1:
        raise ValueError("raw observation field 'ego_current_state' must have a batch dimension")
    batch = ego_current_state.shape[0]
    if batch <= 0:
        raise ValueError("raw observation batch dimension must be positive")
    extra_fields = {
        "route_lanes_speed_limit": ((batch, config.route_num, 1), torch.float32),
        "route_lanes_has_speed_limit": ((batch, config.route_num, 1), torch.bool),
    }
    for name, (shape, dtype) in extra_fields.items():
        value = observation.get(name)
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
            raise ValueError(f"raw observation field {name!r} must have shape {shape}")
        if value.device.type != "cpu" or value.dtype != dtype:
            raise TypeError(f"raw observation field {name!r} has an invalid device or dtype")
        if dtype.is_floating_point and not torch.isfinite(value).all():
            raise ValueError(f"raw observation field {name!r} must be finite")
    return batch


def _prepare_batch_inference_decision(
    noise: torch.Tensor,
    result: PlannerInferenceResult,
    device: torch.device,
    *,
    profile: bool,
) -> tuple[HostExecutionResult, Callable[[], TensorDictBase], float]:
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
    d2h_started = perf_counter() if profile else 0.0
    execution = copy_execution_trajectory(result.prediction, device)
    execution_to_host_s = perf_counter() - d2h_started if profile else 0.0
    return (
        execution,
        lambda: _host_result_from_tensors(deferred, diagnostics is not None),
        execution_to_host_s,
    )


def _host_result_from_tensors(
    deferred: DeferredHostTensors, diagnostics_present: bool
) -> TensorDictBase:
    host_tensors = deferred.resolve()
    arrays = {name: tensor.numpy() for name, tensor in host_tensors.items()}
    for name, value in arrays.items():
        if value.dtype.kind in "fc" and not np.isfinite(value).all():
            raise RuntimeError(f"Diffusion Planner result {name!r} contains non-finite values")
    if diagnostics_present != ("lateral_target_offset_m" in arrays):
        raise RuntimeError("guidance audit tensors disagree with the planner result")
    return TensorDict(host_tensors, batch_size=[arrays["prediction"].shape[0]])


def _synchronize_if_cuda(device: torch.device, enabled: bool) -> None:
    if enabled and device.type == "cuda":
        torch.cuda.synchronize(device)


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
    if mode == "serial":
        workers = 1
    else:
        resources = config.resources
        if resources is None:
            raise ValueError("parallel evaluation requires a resource profile")
        workers = resources.evaluation_job_worker_count
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
