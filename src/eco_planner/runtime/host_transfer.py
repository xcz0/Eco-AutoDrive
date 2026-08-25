"""Synchronous execution and deferred CUDA-to-host audit transfers."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from eco_planner.runtime.contracts import HostExecutionResult


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
