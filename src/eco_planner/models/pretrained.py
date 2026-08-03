"""Strict official-checkpoint loading and inference facade."""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from eco_planner.models.baseline_sampler import BaselineDpmSampler
from eco_planner.models.config import OfficialDiffusionPlannerConfig
from eco_planner.models.diffusion_planner import DiffusionPlanner

OFFICIAL_EMA_TENSOR_COUNT = 276
OFFICIAL_PARAMETER_COUNT = 6_042_628
_REQUIRED_INPUT_SHAPES = {
    "ego_current_state": (10,),
    "neighbor_agents_past": (32, 21, 11),
    "static_objects": (5, 10),
    "lanes": (70, 20, 12),
    "lanes_speed_limit": (70, 1),
    "lanes_has_speed_limit": (70, 1),
    "route_lanes": (25, 20, 12),
}


def sha256_file(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"file does not exist: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_hash(path: Path, expected: str) -> None:
    if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
        raise ValueError("expected SHA-256 must be a 64-character lowercase hexadecimal string")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch for {path}: expected {expected}, got {actual}")


def _extract_ema_state_dict(checkpoint: object) -> OrderedDict[str, torch.Tensor]:
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


class PretrainedDiffusionPlanner(nn.Module):
    """Frozen, official-EMA model with deterministic baseline sampling."""

    def __init__(
        self,
        config: OfficialDiffusionPlannerConfig,
        model: DiffusionPlanner,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.config = config
        self.model = model.to(device).eval()
        self._device = device
        self._sampler = BaselineDpmSampler()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def predict(
        self,
        observation: Mapping[str, torch.Tensor],
        standard_normal_noise: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_observation(observation)
        if standard_normal_noise.dtype != torch.float32:
            raise TypeError("standard_normal_noise must use torch.float32")
        if standard_normal_noise.device != self._device:
            raise ValueError("standard_normal_noise must be on the explicit runtime device")
        batch = observation["ego_current_state"].shape[0]
        expected_noise_shape = (batch, 11, 80, 4)
        if tuple(standard_normal_noise.shape) != expected_noise_shape:
            raise ValueError(f"standard_normal_noise must have shape {expected_noise_shape}")
        if not torch.isfinite(standard_normal_noise).all():
            raise ValueError("standard_normal_noise must be finite")
        inputs = self.config.observation_normalizer(observation)
        with torch.no_grad():
            encoding = self.model.encoder(inputs)["encoding"]
            ego_current = inputs["ego_current_state"][:, None, :4]
            neighbors_current = inputs["neighbor_agents_past"][:, :10, -1, :4]
            neighbor_current_mask = torch.sum(torch.ne(neighbors_current, 0), dim=-1) == 0
            current_states = torch.cat([ego_current, neighbors_current], dim=1)
            initial = torch.cat(
                [current_states[:, :, None], 0.5 * standard_normal_noise], dim=2
            ).reshape(batch, 11, -1)

            def denoiser(sample: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
                return self.model.decoder.decoder.denoise(
                    sample, timestep, encoding, inputs["route_lanes"], neighbor_current_mask
                )

            def constrain(sample: torch.Tensor) -> torch.Tensor:
                constrained = sample.reshape(batch, 11, 81, 4).clone()
                constrained[:, :, 0] = current_states
                return constrained.reshape(batch, 11, -1)

            normalized = self._sampler.sample(initial, denoiser, constrain).reshape(
                batch, 11, 81, 4
            )
            return self.config.state_normalizer.inverse(normalized)[:, :, 1:]

    def _validate_observation(self, observation: Mapping[str, torch.Tensor]) -> None:
        missing = sorted(set(_REQUIRED_INPUT_SHAPES) - set(observation))
        if missing:
            raise ValueError(f"observation is missing required fields: {missing}")
        batch: int | None = None
        for name, tail_shape in _REQUIRED_INPUT_SHAPES.items():
            value = observation[name]
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"observation field {name!r} must be a torch.Tensor")
            if value.device != self._device:
                raise ValueError(
                    f"observation field {name!r} must be on the explicit runtime device"
                )
            if tuple(value.shape[1:]) != tail_shape:
                raise ValueError(
                    f"observation field {name!r} must have shape "
                    f"[B, {', '.join(map(str, tail_shape))}]"
                )
            if batch is None:
                batch = value.shape[0]
            elif value.shape[0] != batch:
                raise ValueError("all observation tensors must share a batch dimension")
            if name == "lanes_has_speed_limit":
                if value.dtype != torch.bool:
                    raise TypeError("lanes_has_speed_limit must use torch.bool")
            else:
                if value.dtype != torch.float32:
                    raise TypeError(f"observation field {name!r} must use torch.float32")
                if not torch.isfinite(value).all():
                    raise ValueError(f"observation field {name!r} must be finite")


def load_official_diffusion_planner(
    args_path: Path,
    checkpoint_path: Path,
    expected_args_sha256: str,
    expected_checkpoint_sha256: str,
    device: torch.device,
) -> tuple[PretrainedDiffusionPlanner, CheckpointLoadReport]:
    """Load the pinned official EMA checkpoint without accepting compatibility fallbacks."""

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was explicitly requested but is unavailable")
    _verify_hash(args_path, expected_args_sha256)
    _verify_hash(checkpoint_path, expected_checkpoint_sha256)
    config = OfficialDiffusionPlannerConfig.from_json(args_path)
    checkpoint: Any = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = _extract_ema_state_dict(checkpoint)
    model = DiffusionPlanner(config)
    model.load_state_dict(state_dict, strict=True)
    planner = PretrainedDiffusionPlanner(config, model, device)
    return planner, CheckpointLoadReport(
        args_sha256=expected_args_sha256,
        checkpoint_sha256=expected_checkpoint_sha256,
        ema_tensor_count=OFFICIAL_EMA_TENSOR_COUNT,
        parameter_count=OFFICIAL_PARAMETER_COUNT,
        runtime_device=str(device),
    )
