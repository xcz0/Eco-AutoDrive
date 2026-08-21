"""CPU-resident inference results and deferred CUDA-to-host transfer contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class HostGuidanceDiagnostics:
    lateral_target_offset_m: np.ndarray
    longitudinal_target_speed_fraction: np.ndarray
    longitudinal_target_speed_delta_mps: np.ndarray
    lateral_objective_delta: np.ndarray
    longitudinal_objective_delta: np.ndarray
    applied_gradient_l2: np.ndarray
    applied_gradient_max_abs: np.ndarray
    raw_neighbor_gradient_l2: np.ndarray
    zero_speed_count: np.ndarray


@dataclass(frozen=True)
class HostExecutionResult:
    """Batched ego trajectories required before advancing simulator workers."""

    ego_trajectory: np.ndarray


class DeferredHostTensors:
    """Host tensors whose CUDA copies complete only when audit data is requested."""

    def __init__(self, tensors: dict[str, torch.Tensor], stream: torch.cuda.Stream | None) -> None:
        self._tensors = tensors
        self._stream = stream
        self._resolved = False

    def resolve(self) -> dict[str, torch.Tensor]:
        """Wait for the optional audit transfer and return its CPU tensors."""

        if not self._resolved and self._stream is not None:
            self._stream.synchronize()
        self._resolved = True
        return self._tensors


@dataclass(frozen=True)
class HostInferenceResult:
    """One planner result transferred to CPU exactly once."""

    initial_noise: np.ndarray
    prediction: np.ndarray
    reference_prediction: np.ndarray | None = None
    guidance_action: np.ndarray | None = None
    guidance_diagnostics: HostGuidanceDiagnostics | None = None

    @property
    def ego_trajectory(self) -> np.ndarray:
        """Return the batch-zero ego prediction as a view, not a copy."""

        return self.prediction[0, 0]


def copy_execution_trajectory(
    prediction: torch.Tensor, device: torch.device
) -> HostExecutionResult:
    """Synchronously copy only the batched ego trajectories required by MetaDrive."""

    host = _copy_tensors_to_host(
        {"ego_trajectory": (prediction[:, 0].detach(), torch.float32)},
        device,
        synchronize=True,
    )
    return HostExecutionResult(host["ego_trajectory"].numpy())


def defer_host_tensors(
    tensors: Mapping[str, tuple[torch.Tensor, torch.dtype]], device: torch.device
) -> DeferredHostTensors:
    """Schedule audit copies on a separate CUDA stream without blocking execution."""

    if device.type != "cuda":
        return DeferredHostTensors(
            {
                name: value.detach().to(device="cpu", dtype=dtype)
                for name, (value, dtype) in tensors.items()
            },
            None,
        )
    source_stream = torch.cuda.current_stream(device)
    transfer_stream = torch.cuda.Stream(device=device)
    transfer_stream.wait_stream(source_stream)
    with torch.cuda.stream(transfer_stream):
        copied = _copy_tensors_to_host(tensors, device, synchronize=False)
    return DeferredHostTensors(copied, transfer_stream)


def _copy_tensors_to_host(
    tensors: Mapping[str, tuple[torch.Tensor, torch.dtype]],
    device: torch.device,
    *,
    synchronize: bool,
) -> dict[str, torch.Tensor]:
    if device.type != "cuda":
        return {
            name: value.detach().to(device="cpu", dtype=dtype)
            for name, (value, dtype) in tensors.items()
        }
    copied: dict[str, torch.Tensor] = {}
    for name, (value, dtype) in tensors.items():
        destination = torch.empty(value.shape, dtype=dtype, device="cpu", pin_memory=True)
        destination.copy_(value.detach().to(dtype=dtype), non_blocking=True)
        copied[name] = destination
    if synchronize:
        torch.cuda.current_stream(device).synchronize()
    return copied
