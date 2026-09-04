"""Configuration and normalization for the diffusion planner."""

from __future__ import annotations

import json
from collections.abc import Mapping
from copy import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import torch
from omegaconf import DictConfig, OmegaConf

from eco_planner.contracts import (
    AGENT_COUNT,
    AGENT_HISTORY_DIM,
    LANE_COUNT,
    LANE_FEATURE_DIM,
    LANE_POINTS,
    PLANNER_HORIZON,
    ROUTE_LANE_COUNT,
    STATIC_OBJECT_COUNT,
    STATIC_OBJECT_DIM,
    TRAFFIC_HISTORY_FRAMES,
)

DDIM5_TIMESTEPS = (1.0, 0.8, 0.6, 0.4, 0.2, 0.0)

_DeviceDtype = tuple[torch.device, torch.dtype]
_NormalizationConstants = tuple[torch.Tensor, torch.Tensor]


class StateNormalizer:
    def __init__(self, mean: object, std: object) -> None:
        self.mean = torch.as_tensor(mean, dtype=torch.float32)
        self.std = torch.as_tensor(std, dtype=torch.float32)
        self._cached_constants: dict[_DeviceDtype, _NormalizationConstants] = {}

    def inverse(self, data: torch.Tensor) -> torch.Tensor:
        mean, std = self._constants(data)
        return data * std + mean

    def _constants(self, data: torch.Tensor) -> _NormalizationConstants:
        key = (data.device, data.dtype)
        if key not in self._cached_constants:
            self._cached_constants[key] = (
                self.mean.to(device=data.device, dtype=data.dtype),
                self.std.to(device=data.device, dtype=data.dtype),
            )
        return self._cached_constants[key]


class ObservationNormalizer:
    """Apply checkpoint normalization while preserving all-zero padding."""

    def __init__(self, normalization: Mapping[str, Mapping[str, object]]) -> None:
        self._normalization = {
            name: {
                "mean": torch.as_tensor(values["mean"], dtype=torch.float32),
                "std": torch.as_tensor(values["std"], dtype=torch.float32),
            }
            for name, values in normalization.items()
        }
        self._cached_constants: dict[str, dict[_DeviceDtype, _NormalizationConstants]] = {}

    def __call__(self, data: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        normalized = copy(dict(data))
        for name in self._normalization:
            if name not in data:
                continue
            tensor = data[name]
            padding = torch.sum(torch.ne(tensor, 0), dim=-1) == 0
            mean, std = self._constants(name, tensor)
            result = (tensor - mean) / std
            result[padding] = 0
            normalized[name] = result
        return normalized

    def _constants(self, name: str, data: torch.Tensor) -> _NormalizationConstants:
        key = (data.device, data.dtype)
        name_cache = self._cached_constants.setdefault(name, {})
        if key not in name_cache:
            values = self._normalization[name]
            name_cache[key] = (
                values["mean"].to(device=data.device, dtype=data.dtype),
                values["std"].to(device=data.device, dtype=data.dtype),
            )
        return name_cache[key]

    def feature_dimension(self, name: str) -> int:
        return int(self._normalization[name]["mean"].numel())

    @property
    def feature_dimensions(self) -> dict[str, int]:
        return {name: int(values["mean"].numel()) for name, values in self._normalization.items()}


@dataclass(frozen=True)
class OfficialDiffusionPlannerConfig:
    future_len: int
    time_len: int
    agent_state_dim: int
    agent_num: int
    static_objects_state_dim: int
    static_objects_num: int
    lane_len: int
    lane_state_dim: int
    lane_num: int
    map_len: int
    map_state_dim: int
    map_num: int
    route_len: int
    route_state_dim: int
    route_num: int
    encoder_drop_path_rate: float
    decoder_drop_path_rate: float
    checkpoint_device: str
    encoder_depth: int
    decoder_depth: int
    num_heads: int
    hidden_dim: int
    diffusion_model_type: str
    predicted_neighbor_num: int
    state_normalizer: StateNormalizer
    observation_normalizer: ObservationNormalizer

    def __post_init__(self) -> None:
        fixed_dimensions = {
            "future_len": PLANNER_HORIZON,
            "time_len": TRAFFIC_HISTORY_FRAMES,
            "agent_state_dim": AGENT_HISTORY_DIM,
            "agent_num": AGENT_COUNT,
            "static_objects_state_dim": STATIC_OBJECT_DIM,
            "static_objects_num": STATIC_OBJECT_COUNT,
            "lane_len": LANE_POINTS,
            "lane_state_dim": LANE_FEATURE_DIM,
            "lane_num": LANE_COUNT,
            "route_len": LANE_POINTS,
            "route_state_dim": LANE_FEATURE_DIM,
            "route_num": ROUTE_LANE_COUNT,
        }
        mismatches = {
            name: (getattr(self, name), expected)
            for name, expected in fixed_dimensions.items()
            if getattr(self, name) != expected
        }
        if mismatches:
            details = ", ".join(
                f"{name}={actual!r} (expected {expected})"
                for name, (actual, expected) in mismatches.items()
            )
            raise ValueError(f"official planner dimensions do not match the fixed ABI: {details}")

    @property
    def observation_feature_dimensions(self) -> dict[str, int]:
        return self.observation_normalizer.feature_dimensions

    @classmethod
    def from_json(cls, path: Path) -> OfficialDiffusionPlannerConfig:
        with path.open("r", encoding="utf-8") as handle:
            raw: dict[str, Any] = json.load(handle)
        checkpoint_device = raw.pop("device")
        state_normalizer = StateNormalizer(**raw.pop("state_normalizer"))
        observation_normalizer = ObservationNormalizer(raw.pop("observation_normalizer"))
        return cls(
            **raw,
            checkpoint_device=checkpoint_device,
            state_normalizer=state_normalizer,
            observation_normalizer=observation_normalizer,
        )


@dataclass(frozen=True)
class Dpm10SamplerConfig:
    name: Literal["dpm10"] = "dpm10"
    num_steps: int = 10
    initial_noise_scale: float = 0.5
    ddim_stochasticity: float = 0.0
    parity_label: Literal["official_diffusion_planner_baseline"] = (
        "official_diffusion_planner_baseline"
    )
    implementation: Literal["diffusers"] = "diffusers"


@dataclass(frozen=True)
class Ddim5SamplerConfig:
    name: Literal["ddim5"]
    num_steps: int
    timesteps: tuple[float, ...]
    initial_noise_scale: float
    ddim_stochasticity: float
    parity_label: Literal["plannerrft_paper_text", "project_noise_scale_0_5"]
    implementation: Literal["diffusers"] = "diffusers"


SamplerConfig = Dpm10SamplerConfig | Ddim5SamplerConfig


@dataclass(frozen=True)
class SamplerReport:
    name: str
    num_steps: int
    timesteps: tuple[float, ...] | None
    initial_noise_scale: float
    ddim_stochasticity: float
    parity_label: str
    implementation: str


def parse_sampler_config(config: DictConfig) -> SamplerConfig:
    raw = cast(dict[str, Any], OmegaConf.to_container(config, resolve=True))
    if raw["name"] == "dpm10":
        return Dpm10SamplerConfig(**raw)
    raw["timesteps"] = tuple(raw["timesteps"])
    return Ddim5SamplerConfig(**raw)


def sampler_report(config: SamplerConfig) -> SamplerReport:
    return SamplerReport(
        name=config.name,
        num_steps=config.num_steps,
        timesteps=None if isinstance(config, Dpm10SamplerConfig) else config.timesteps,
        initial_noise_scale=config.initial_noise_scale,
        ddim_stochasticity=config.ddim_stochasticity,
        parity_label=config.parity_label,
        implementation=config.implementation,
    )


@dataclass(frozen=True)
class NoGuidanceConfig:
    name: Literal["none"] = "none"


@dataclass(frozen=True)
class OrthogonalReferenceGuidanceConfig:
    name: Literal["orthogonal_reference"]
    formula_label: Literal["centered_energy_gradient_delta_v1"]
    lateral_scale: float
    longitudinal_scale: float
    lateral_max_offset_m: float
    longitudinal_max_speed_fraction: float
    trajectory_dt_s: float
    gradient_step_coefficient: float
    reference_refresh_cycles: int
    share_scene_encoding: bool
    share_initial_noise: bool
    share_transition_noise: bool
    heading_norm_epsilon: float
    zero_speed_tolerance_mps: float

    @property
    def fixed_action(self) -> tuple[float, float]:
        return (self.lateral_scale, self.longitudinal_scale)


@dataclass(frozen=True)
class OrthogonalPolicyGuidanceConfig:
    name: Literal["orthogonal_policy"]
    formula_label: Literal["centered_energy_gradient_delta_v1"]
    lateral_max_offset_m: float
    longitudinal_max_speed_fraction: float
    trajectory_dt_s: float
    gradient_step_coefficient: float
    reference_refresh_cycles: int
    share_scene_encoding: bool
    share_initial_noise: bool
    share_transition_noise: bool
    heading_norm_epsilon: float
    zero_speed_tolerance_mps: float


GuidanceConfig = (
    NoGuidanceConfig | OrthogonalReferenceGuidanceConfig | OrthogonalPolicyGuidanceConfig
)


def parse_guidance_config(config: DictConfig) -> GuidanceConfig:
    raw = cast(dict[str, Any], OmegaConf.to_container(config, resolve=True))
    if raw["name"] == "none":
        return NoGuidanceConfig(**raw)
    if raw["name"] == "orthogonal_reference":
        return OrthogonalReferenceGuidanceConfig(**raw)
    return OrthogonalPolicyGuidanceConfig(**raw)
