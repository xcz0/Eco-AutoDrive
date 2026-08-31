"""Synchronous execution and deferred CUDA-to-host audit transfers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from time import perf_counter

import torch

from eco_planner.runtime.contracts import HostExecutionResult


@dataclass(frozen=True)
class DeferredHostTransferTiming:
    """Profile-only device copy duration and remaining host wait."""

    accelerator_s: float
    resolve_wait_wall_s: float


class DeferredHostTensors:
    """Host tensors whose CUDA copies complete only when audit data is requested."""

    def __init__(
        self,
        tensors: dict[str, torch.Tensor],
        stream: torch.cuda.Stream | None,
        *,
        start_event: torch.cuda.Event | None = None,
        end_event: torch.cuda.Event | None = None,
        cpu_copy_s: float | None = None,
    ) -> None:
        self._tensors = tensors
        self._stream = stream
        self._start_event = start_event
        self._end_event = end_event
        self._cpu_copy_s = cpu_copy_s
        self._resolved = False
        self._timing: DeferredHostTransferTiming | None = None

    def resolve(self) -> dict[str, torch.Tensor]:
        """Wait for the optional audit transfer and return its CPU tensors."""

        started = perf_counter()
        if not self._resolved and self._stream is not None:
            self._stream.synchronize()
        resolve_wait_s = perf_counter() - started
        if not self._resolved and self._start_event is not None and self._end_event is not None:
            self._timing = DeferredHostTransferTiming(
                accelerator_s=self._start_event.elapsed_time(self._end_event) / 1000.0,
                resolve_wait_wall_s=resolve_wait_s,
            )
        elif not self._resolved and self._cpu_copy_s is not None:
            self._timing = DeferredHostTransferTiming(
                accelerator_s=self._cpu_copy_s,
                resolve_wait_wall_s=resolve_wait_s,
            )
        self._resolved = True
        return self._tensors

    @property
    def timing(self) -> DeferredHostTransferTiming | None:
        """Return profile-only transfer timing after the tensors are resolved."""

        return self._timing


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
    tensors: Mapping[str, tuple[torch.Tensor, torch.dtype]],
    device: torch.device,
    *,
    profile: bool = False,
) -> DeferredHostTensors:
    """Schedule audit copies on a separate CUDA stream without blocking execution."""

    if device.type != "cuda":
        started = perf_counter() if profile else 0.0
        copied = {
            name: value.detach().to(device="cpu", dtype=dtype)
            for name, (value, dtype) in tensors.items()
        }
        return DeferredHostTensors(
            copied,
            None,
            cpu_copy_s=perf_counter() - started if profile else None,
        )
    source_stream = torch.cuda.current_stream(device)
    transfer_stream = torch.cuda.Stream(device=device)
    transfer_stream.wait_stream(source_stream)
    start_event = torch.cuda.Event(enable_timing=True) if profile else None
    end_event = torch.cuda.Event(enable_timing=True) if profile else None
    with torch.cuda.stream(transfer_stream):
        if start_event is not None:
            start_event.record()
        copied = _copy_tensors_to_host(tensors, device, synchronize=False)
        if end_event is not None:
            end_event.record()
    return DeferredHostTensors(
        copied,
        transfer_stream,
        start_event=start_event,
        end_event=end_event,
    )


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
