from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from eco_planner.models.sampling_config import (
    Ddim5SamplerConfig,
    Dpm10SamplerConfig,
    parse_sampler_config,
    sampler_report,
)


def _ddim_config(**overrides: object) -> object:
    values: dict[str, object] = {
        "name": "ddim5",
        "implementation": "legacy",
        "num_steps": 5,
        "timesteps": [1.0, 0.8, 0.6, 0.4, 0.2, 0.0],
        "initial_noise_scale": 1.0,
        "ddim_stochasticity": 0.0,
        "parity_label": "plannerrft_paper_text",
    }
    values.update(overrides)
    return OmegaConf.create(values)


def test_sampler_config_parses_strict_profiles() -> None:
    dpm = parse_sampler_config(OmegaConf.create({"name": "dpm10", "implementation": "legacy"}))
    ddim = parse_sampler_config(_ddim_config())  # type: ignore[arg-type]

    assert isinstance(dpm, Dpm10SamplerConfig)
    assert isinstance(ddim, Ddim5SamplerConfig)
    assert sampler_report(dpm).initial_noise_scale == 0.5
    assert sampler_report(ddim).parity_label == "plannerrft_paper_text"
    assert sampler_report(ddim).implementation == "legacy"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"num_steps": 4}, "num_steps"),
        ({"timesteps": [1.0, 0.0]}, "timesteps"),
        ({"ddim_stochasticity": 1.1}, "stochasticity"),
        ({"initial_noise_scale": 0.5}, "requires"),
        ({"parity_label": "unknown"}, "parity_label"),
    ],
)
def test_sampler_config_rejects_invalid_ddim_profiles(
    overrides: dict[str, object], message: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        parse_sampler_config(_ddim_config(**overrides))  # type: ignore[arg-type]


def test_dpm_config_rejects_irrelevant_math_fields() -> None:
    with pytest.raises(ValueError, match="unexpected"):
        parse_sampler_config(
            OmegaConf.create({"name": "dpm10", "implementation": "legacy", "num_steps": 10})
        )


def test_sampler_config_allows_only_certified_backends() -> None:
    config = parse_sampler_config(_ddim_config(implementation="diffusers"))  # type: ignore[arg-type]
    assert config.implementation == "diffusers"
    with pytest.raises(ValueError, match="until DPM parity"):
        parse_sampler_config(OmegaConf.create({"name": "dpm10", "implementation": "diffusers"}))
