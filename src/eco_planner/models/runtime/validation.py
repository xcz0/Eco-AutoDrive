"""Runtime tensor contracts for official Diffusion Planner inference."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from eco_planner.models.checkpoint.config import OfficialDiffusionPlannerConfig


def observation_shapes(config: OfficialDiffusionPlannerConfig) -> dict[str, tuple[int, ...]]:
    """Derive the runtime observation schema from the loaded architecture."""

    return {
        "ego_current_state": (config.observation_feature_dimensions["ego_current_state"],),
        "neighbor_agents_past": (
            config.agent_num,
            config.time_len,
            config.agent_state_dim,
        ),
        "static_objects": (
            config.static_objects_num,
            config.static_objects_state_dim,
        ),
        "lanes": (config.lane_num, config.lane_len, config.lane_state_dim),
        "lanes_speed_limit": (config.lane_num, 1),
        "lanes_has_speed_limit": (config.lane_num, 1),
        "route_lanes": (config.route_num, config.route_len, config.route_state_dim),
    }


def validate_official_observation(
    observation: Mapping[str, torch.Tensor],
    device: torch.device,
    config: OfficialDiffusionPlannerConfig,
) -> int:
    required_shapes = observation_shapes(config)
    missing = sorted(set(required_shapes) - set(observation))
    if missing:
        raise ValueError(f"observation is missing required fields: {missing}")
    batch: int | None = None
    for name, tail_shape in required_shapes.items():
        value = observation[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"observation field {name!r} must be a torch.Tensor")
        if value.device != device:
            raise ValueError(f"observation field {name!r} must be on the runtime device {device}")
        if tuple(value.shape[1:]) != tail_shape:
            raise ValueError(
                f"observation field {name!r} must have shape [B, {', '.join(map(str, tail_shape))}]"
            )
        if batch is None:
            batch = value.shape[0]
            if batch <= 0:
                raise ValueError("observation batch dimension must be positive")
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
    if batch is None:
        raise RuntimeError("official observation contract has no required fields")
    return batch


def validate_standard_normal_noise(
    noise: torch.Tensor,
    *,
    batch: int,
    participants: int,
    future_len: int,
    device: torch.device,
) -> None:
    if not isinstance(noise, torch.Tensor):
        raise TypeError("standard_normal_noise must be a torch.Tensor")
    if noise.dtype != torch.float32:
        raise TypeError("standard_normal_noise must use torch.float32")
    if noise.device != device:
        raise ValueError(f"standard_normal_noise must be on the runtime device {device}")
    expected_shape = (batch, participants, future_len, 4)
    if tuple(noise.shape) != expected_shape:
        raise ValueError(f"standard_normal_noise must have shape {expected_shape}")
    if not torch.isfinite(noise).all():
        raise ValueError("standard_normal_noise must be finite")
