from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from eco_planner.models.guidance import (
    NoGuidanceConfig,
    OrthogonalReferenceGuidanceConfig,
    parse_guidance_config,
    validate_guidance_sampler,
)
from eco_planner.models.sampling_config import Ddim5SamplerConfig, Dpm10SamplerConfig


def _active_config(**overrides: object) -> object:
    values: dict[str, object] = {
        "name": "orthogonal_reference",
        "formula_label": "centered_energy_gradient_delta_v1",
        "lateral_scale": 0.0,
        "longitudinal_scale": 0.0,
        "lateral_max_offset_m": 2.5,
        "longitudinal_max_speed_fraction": 0.25,
        "trajectory_dt_s": 0.1,
        "gradient_step_coefficient": 1.0,
        "reference_refresh_cycles": 1,
        "share_scene_encoding": True,
        "share_initial_noise": True,
        "share_transition_noise": True,
        "heading_norm_epsilon": 1e-6,
        "zero_speed_tolerance_mps": 1e-6,
    }
    values.update(overrides)
    return OmegaConf.create(values)


def _ddim(scale: float = 1.0, label: str = "plannerrft_paper_text") -> Ddim5SamplerConfig:
    return Ddim5SamplerConfig(
        name="ddim5",
        num_steps=5,
        timesteps=(1.0, 0.8, 0.6, 0.4, 0.2, 0.0),
        initial_noise_scale=scale,
        ddim_stochasticity=0.0,
        parity_label=label,  # type: ignore[arg-type]
    )


def test_guidance_config_parses_none_and_supported_orthogonal_profile() -> None:
    none = parse_guidance_config(OmegaConf.create({"name": "none"}))
    active = parse_guidance_config(_active_config())  # type: ignore[arg-type]

    assert isinstance(none, NoGuidanceConfig)
    assert isinstance(active, OrthogonalReferenceGuidanceConfig)
    assert active.fixed_action == (0.0, 0.0)
    validate_guidance_sampler(none, Dpm10SamplerConfig())
    validate_guidance_sampler(active, _ddim())


@pytest.mark.parametrize(
    ("sampler", "message"),
    [
        (Dpm10SamplerConfig(), "DDIM-5"),
        (_ddim(0.5, "project_noise_scale_0_5"), "standard-Gaussian"),
    ],
)
def test_active_guidance_rejects_incompatible_sampler(
    sampler: Dpm10SamplerConfig | Ddim5SamplerConfig,
    message: str,
) -> None:
    guidance = parse_guidance_config(_active_config())  # type: ignore[arg-type]

    with pytest.raises(ValueError, match=message):
        validate_guidance_sampler(guidance, sampler)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"lateral_scale": 1.1}, "lateral_scale"),
        ({"longitudinal_scale": float("nan")}, "longitudinal_scale"),
        ({"lateral_max_offset_m": 0.0}, "lateral_max_offset_m"),
        ({"longitudinal_max_speed_fraction": -0.1}, "longitudinal_max_speed_fraction"),
        ({"trajectory_dt_s": 0.0}, "trajectory_dt_s"),
        ({"gradient_step_coefficient": 0.5}, "unit coefficient"),
        ({"reference_refresh_cycles": 2}, "every planning cycle"),
        ({"share_scene_encoding": False}, "share_scene_encoding"),
        ({"heading_norm_epsilon": 0.0}, "heading_norm_epsilon"),
    ],
)
def test_active_guidance_rejects_values_outside_stage2_contract(
    overrides: dict[str, object], message: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        parse_guidance_config(_active_config(**overrides))  # type: ignore[arg-type]


def test_active_guidance_rejects_missing_or_unexpected_fields() -> None:
    missing = _active_config()
    del missing["trajectory_dt_s"]  # type: ignore[index]
    with pytest.raises(ValueError, match="missing"):
        parse_guidance_config(missing)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="unexpected"):
        parse_guidance_config(_active_config(extra=True))  # type: ignore[arg-type]
