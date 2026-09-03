"""Inference result validation and host-transfer decision boundary."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter

import numpy as np
import torch
from tensordict import TensorDict, TensorDictBase

from eco_planner.models import (
    GuidanceConfig,
    NoGuidanceConfig,
    OfficialDiffusionPlannerConfig,
    PlannerInferenceResult,
)
from eco_planner.runtime.contracts import HostTrajectories
from eco_planner.runtime.host_transfer import DeferredHostTensors, HostTransfer


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
        execution: HostTrajectories,
        resolve_audit: Callable[[], TensorDictBase],
        timing: BatchInferenceTiming | None = None,
    ) -> None:
        self._execution = execution
        self._resolve_audit = resolve_audit
        self._audit: TensorDictBase | None = None
        self._timing = timing

    @property
    def ego_trajectory(self) -> np.ndarray:
        if self._execution.ego.shape[0] != 1:
            raise RuntimeError("ego_trajectory is only available for a batch-one decision")
        return self._execution.ego[0]

    @property
    def ego_trajectories(self) -> np.ndarray:
        """Return executable ego trajectories with shape ``[B, T, 4]``."""

        return self._execution.ego

    @property
    def timing(self) -> BatchInferenceTiming | None:
        """Return synchronized component timing for an explicit profiled call."""

        return self._timing

    def audit_result(self) -> TensorDictBase:
        """Wait for and return the complete artifact/replay payload."""

        if self._audit is None:
            self._audit = self._resolve_audit()
        return self._audit


def validate_optional_guidance_result(
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


def validate_artifact_observation_fields(
    observation: TensorDictBase,
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


def prepare_batch_inference_decision(
    noise: torch.Tensor,
    result: PlannerInferenceResult,
    host_transfer: HostTransfer,
    *,
    profile: bool,
) -> tuple[HostTrajectories, Callable[[], TensorDictBase], float]:
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

    deferred = host_transfer.defer(tensors, profile=profile)
    d2h_started = perf_counter() if profile else 0.0
    execution = host_transfer.execution_trajectories(result.prediction)
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


def synchronize_if_cuda(device: torch.device, enabled: bool) -> None:
    if enabled and device.type == "cuda":
        torch.cuda.synchronize(device)
