"""Structural validation and parsing for the pinned official checkpoint."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import torch

OFFICIAL_EMA_TENSOR_COUNT = 276
OFFICIAL_PARAMETER_COUNT = 6_042_628


def extract_official_ema_state_dict(
    checkpoint: object,
) -> OrderedDict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict) or set(checkpoint) != {"model", "ema_state_dict"}:
        raise ValueError("official checkpoint must contain exactly model and ema_state_dict")
    ema = checkpoint["ema_state_dict"]
    if not isinstance(ema, (dict, OrderedDict)):
        raise ValueError("ema_state_dict must be a state-dict mapping")
    stripped: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key, value in ema.items():
        if not isinstance(key, str) or not key.startswith("module."):
            raise ValueError("every EMA checkpoint key must have exactly one module. prefix")
        name = key[len("module.") :]
        if name.startswith("module.") or name in stripped:
            raise ValueError("EMA checkpoint contains an invalid or duplicate stripped key")
        if not isinstance(value, torch.Tensor):
            raise ValueError("EMA checkpoint values must all be tensors")
        stripped[name] = value
    if len(stripped) != OFFICIAL_EMA_TENSOR_COUNT:
        message = f"EMA checkpoint must contain {OFFICIAL_EMA_TENSOR_COUNT} tensors"
        message += f", got {len(stripped)}"
        raise ValueError(message)
    parameter_count = sum(value.numel() for value in stripped.values())
    if parameter_count != OFFICIAL_PARAMETER_COUNT:
        message = f"EMA checkpoint must contain {OFFICIAL_PARAMETER_COUNT} parameters"
        message += f", got {parameter_count}"
        raise ValueError(message)
    return stripped


@dataclass(frozen=True)
class CheckpointLoadReport:
    ema_tensor_count: int
    parameter_count: int
    runtime_device: str
