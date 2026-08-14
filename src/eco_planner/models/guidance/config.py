"""Strict Hydra-facing configuration for reference guidance."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from omegaconf import DictConfig, OmegaConf

from eco_planner.models.sampling.config import Ddim5SamplerConfig, SamplerConfig

_FORMULA_LABEL = "centered_energy_gradient_delta_v1"


@dataclass(frozen=True)
class NoGuidanceConfig:
    """Disable reference generation and orthogonal guidance."""

    name: Literal["none"] = "none"


@dataclass(frozen=True)
class OrthogonalReferenceGuidanceConfig:
    """Fully explicit fixed-action profile for stage-2 guidance evaluation."""

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
    """Stage-4 learned-action profile sharing the stage-2 geometric guidance contract."""

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
    """Parse one strict Hydra guidance mapping without hidden defaults."""

    if not isinstance(config, DictConfig):
        raise TypeError("guidance configuration must be a DictConfig")
    raw = OmegaConf.to_container(config, resolve=True)
    if not isinstance(raw, dict):
        raise TypeError("guidance configuration must resolve to a dictionary")
    name = raw.get("name")
    if name == "none":
        _require_exact_keys(raw, {"name"}, "none")
        return NoGuidanceConfig()
    if name not in {"orthogonal_reference", "orthogonal_policy"}:
        raise ValueError(
            "guidance.name must be 'none', 'orthogonal_reference', or 'orthogonal_policy'"
        )

    required = {
        "name",
        "formula_label",
        "lateral_max_offset_m",
        "longitudinal_max_speed_fraction",
        "trajectory_dt_s",
        "gradient_step_coefficient",
        "reference_refresh_cycles",
        "share_scene_encoding",
        "share_initial_noise",
        "share_transition_noise",
        "heading_norm_epsilon",
        "zero_speed_tolerance_mps",
    }
    if name == "orthogonal_reference":
        required.add("lateral_scale")
        required.add("longitudinal_scale")
    _require_exact_keys(raw, required, str(name))
    if raw["formula_label"] != _FORMULA_LABEL:
        raise ValueError(f"formula_label must equal {_FORMULA_LABEL!r}")
    lateral_max_offset_m = _positive_float(raw["lateral_max_offset_m"], "lateral_max_offset_m")
    longitudinal_max_speed_fraction = _positive_float(
        raw["longitudinal_max_speed_fraction"], "longitudinal_max_speed_fraction"
    )
    trajectory_dt_s = _positive_float(raw["trajectory_dt_s"], "trajectory_dt_s")
    gradient_step_coefficient = _finite_float(
        raw["gradient_step_coefficient"], "gradient_step_coefficient"
    )
    if gradient_step_coefficient != 1.0:
        raise ValueError("stage-2 guidance requires a unit coefficient")
    reference_refresh_cycles = raw["reference_refresh_cycles"]
    if type(reference_refresh_cycles) is not int or reference_refresh_cycles != 1:
        raise ValueError("reference must refresh every planning cycle")
    for field in ("share_scene_encoding", "share_initial_noise", "share_transition_noise"):
        if raw[field] is not True:
            raise ValueError(f"{field} must be true for stage-2 guidance")
    heading_norm_epsilon = _positive_float(raw["heading_norm_epsilon"], "heading_norm_epsilon")
    zero_speed_tolerance_mps = _positive_float(
        raw["zero_speed_tolerance_mps"], "zero_speed_tolerance_mps"
    )
    common = dict(
        formula_label=_FORMULA_LABEL,
        lateral_max_offset_m=lateral_max_offset_m,
        longitudinal_max_speed_fraction=longitudinal_max_speed_fraction,
        trajectory_dt_s=trajectory_dt_s,
        gradient_step_coefficient=gradient_step_coefficient,
        reference_refresh_cycles=reference_refresh_cycles,
        share_scene_encoding=True,
        share_initial_noise=True,
        share_transition_noise=True,
        heading_norm_epsilon=heading_norm_epsilon,
        zero_speed_tolerance_mps=zero_speed_tolerance_mps,
    )
    if name == "orthogonal_policy":
        return OrthogonalPolicyGuidanceConfig(name="orthogonal_policy", **common)
    return OrthogonalReferenceGuidanceConfig(
        name="orthogonal_reference",
        lateral_scale=_bounded_scale(raw["lateral_scale"], "lateral_scale"),
        longitudinal_scale=_bounded_scale(raw["longitudinal_scale"], "longitudinal_scale"),
        **common,
    )


def validate_guidance_sampler(config: GuidanceConfig, sampler: SamplerConfig) -> None:
    """Reject active guidance outside the standard-Gaussian DDIM-5 profile."""

    if isinstance(config, NoGuidanceConfig):
        return
    if not isinstance(sampler, Ddim5SamplerConfig):
        raise ValueError("orthogonal reference guidance requires DDIM-5")
    if sampler.initial_noise_scale != 1.0 or sampler.parity_label != "plannerrft_paper_text":
        raise ValueError("orthogonal reference guidance requires standard-Gaussian DDIM-5")


def _require_exact_keys(raw: dict[object, object], expected: set[str], name: str) -> None:
    keys = set(raw)
    if keys != expected:
        raise ValueError(
            f"{name} guidance keys mismatch; missing={sorted(expected - keys)}, "
            f"unexpected={sorted(str(key) for key in keys - expected)}"
        )


def _finite_float(value: object, name: str) -> float:
    if type(value) not in {int, float}:
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_float(value: object, name: str) -> float:
    result = _finite_float(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _bounded_scale(value: object, name: str) -> float:
    result = _finite_float(value, name)
    if not -1.0 <= result <= 1.0:
        raise ValueError(f"{name} must be in [-1, 1]")
    return result
