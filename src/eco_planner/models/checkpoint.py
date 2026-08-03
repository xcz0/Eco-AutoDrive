"""Integrity checks and parsing for the pinned official checkpoint."""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import torch

OFFICIAL_EMA_TENSOR_COUNT = 276
OFFICIAL_PARAMETER_COUNT = 6_042_628


def sha256_file(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"file does not exist: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_sha256(path: Path, expected: str) -> None:
    if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
        raise ValueError("expected SHA-256 must be a 64-character lowercase hexadecimal string")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch for {path}: expected {expected}, got {actual}")


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
    args_sha256: str
    checkpoint_sha256: str
    ema_tensor_count: int
    parameter_count: int
    runtime_device: str
