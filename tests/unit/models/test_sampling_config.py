from omegaconf import OmegaConf

from eco_planner.models.config import (
    DDIM5_TIMESTEPS,
    Ddim5SamplerConfig,
    Dpm10SamplerConfig,
    parse_sampler_config,
    sampler_report,
)


def test_sampler_config_parses_canonical_profiles() -> None:
    dpm = parse_sampler_config(OmegaConf.create({"name": "dpm10", "implementation": "diffusers"}))
    ddim = parse_sampler_config(
        OmegaConf.create(
            {
                "name": "ddim5",
                "implementation": "diffusers",
                "num_steps": 5,
                "timesteps": list(DDIM5_TIMESTEPS),
                "initial_noise_scale": 1.0,
                "ddim_stochasticity": 0.0,
                "parity_label": "plannerrft_paper_text",
            }
        )
    )

    assert isinstance(dpm, Dpm10SamplerConfig)
    assert isinstance(ddim, Ddim5SamplerConfig)
    assert ddim.timesteps == DDIM5_TIMESTEPS
    assert sampler_report(dpm).initial_noise_scale == 0.5
    assert sampler_report(ddim).parity_label == "plannerrft_paper_text"
