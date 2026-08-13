"""Strict Hydra-facing configuration for diffusion sampler selection."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from omegaconf import DictConfig, OmegaConf

DDIM5_TIMESTEPS = (1.0, 0.8, 0.6, 0.4, 0.2, 0.0)
_DDIM_PARITY_SCALES = {
    "plannerrft_paper_text": 1.0,
    "project_noise_scale_0_5": 0.5,
}


@dataclass(frozen=True)
class Dpm10SamplerConfig:
    """The immutable official Diffusion Planner sampling profile."""

    name: Literal["dpm10"] = "dpm10"
    implementation: Literal["diffusers"] = "diffusers"


@dataclass(frozen=True)
class Ddim5SamplerConfig:
    """The explicit five-transition DDIM reproduction profile."""

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
    """Stable sampler metadata persisted with every evaluation artifact."""

    name: str
    num_steps: int
    timesteps: tuple[float, ...] | None
    initial_noise_scale: float
    ddim_stochasticity: float
    parity_label: str
    implementation: str


def parse_sampler_config(config: DictConfig) -> SamplerConfig:
    """Parse one strict Hydra sampler mapping without hidden defaults."""

    if not isinstance(config, DictConfig):
        raise TypeError("sampler configuration must be a DictConfig")
    raw = OmegaConf.to_container(config, resolve=True)
    if not isinstance(raw, dict):
        raise TypeError("sampler configuration must resolve to a dictionary")
    name = raw.get("name")
    if name == "dpm10":
        _require_exact_keys(raw, {"name", "implementation"}, "dpm10")
        implementation = raw["implementation"]
        if implementation != "diffusers":
            raise ValueError("dpm10 implementation must be 'diffusers'")
        return Dpm10SamplerConfig(implementation=implementation)
    if name != "ddim5":
        raise ValueError("sampler.name must be either 'dpm10' or 'ddim5'")

    required = {
        "name",
        "num_steps",
        "timesteps",
        "initial_noise_scale",
        "ddim_stochasticity",
        "parity_label",
        "implementation",
    }
    _require_exact_keys(raw, required, "ddim5")
    num_steps = raw["num_steps"]
    if type(num_steps) is not int or num_steps != 5:
        raise ValueError("ddim5 num_steps must be the integer 5")
    raw_timesteps = raw["timesteps"]
    if not isinstance(raw_timesteps, list):
        raise TypeError("ddim5 timesteps must be a list")
    timesteps = tuple(_finite_float(value, "ddim5 timestep") for value in raw_timesteps)
    if timesteps != DDIM5_TIMESTEPS:
        raise ValueError(f"ddim5 timesteps must equal {list(DDIM5_TIMESTEPS)}")

    initial_noise_scale = _finite_float(raw["initial_noise_scale"], "initial_noise_scale")
    stochasticity = _finite_float(raw["ddim_stochasticity"], "ddim_stochasticity")
    if not 0.0 <= stochasticity <= 1.0:
        raise ValueError("ddim_stochasticity must be in [0, 1]")
    parity_label = raw["parity_label"]
    if parity_label not in _DDIM_PARITY_SCALES:
        raise ValueError(
            "ddim5 parity_label must be 'plannerrft_paper_text' or 'project_noise_scale_0_5'"
        )
    expected_scale = _DDIM_PARITY_SCALES[parity_label]
    if initial_noise_scale != expected_scale:
        raise ValueError(
            f"ddim5 parity_label {parity_label!r} requires initial_noise_scale={expected_scale}"
        )
    implementation = raw["implementation"]
    if implementation != "diffusers":
        raise ValueError("ddim5 implementation must be 'diffusers'")
    return Ddim5SamplerConfig(
        name="ddim5",
        num_steps=num_steps,
        timesteps=timesteps,
        initial_noise_scale=initial_noise_scale,
        ddim_stochasticity=stochasticity,
        parity_label=parity_label,
        implementation=implementation,
    )


def sampler_report(config: SamplerConfig) -> SamplerReport:
    if isinstance(config, Dpm10SamplerConfig):
        return SamplerReport(
            name="dpm10",
            num_steps=10,
            timesteps=None,
            initial_noise_scale=0.5,
            ddim_stochasticity=0.0,
            parity_label="official_diffusion_planner_baseline",
            implementation=config.implementation,
        )
    return SamplerReport(
        name=config.name,
        num_steps=config.num_steps,
        timesteps=config.timesteps,
        initial_noise_scale=config.initial_noise_scale,
        ddim_stochasticity=config.ddim_stochasticity,
        parity_label=config.parity_label,
        implementation=config.implementation,
    )


def _require_exact_keys(raw: dict[object, object], expected: set[str], name: str) -> None:
    keys = set(raw)
    if keys != expected:
        missing = sorted(expected - keys)
        unexpected = sorted(str(key) for key in keys - expected)
        raise ValueError(
            f"{name} sampler keys mismatch; missing={missing}, unexpected={unexpected}"
        )


def _finite_float(value: object, name: str) -> float:
    if type(value) not in {int, float}:
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result
