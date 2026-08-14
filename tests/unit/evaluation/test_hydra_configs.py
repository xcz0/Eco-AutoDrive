from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from eco_planner.models.guidance import (
    OrthogonalReferenceGuidanceConfig,
    parse_guidance_config,
    validate_guidance_sampler,
)
from eco_planner.models.sampling import parse_sampler_config, sampler_report


@pytest.mark.parametrize(
    ("profile", "scale", "label"),
    [
        ("dpm10", 0.5, "official_diffusion_planner_baseline"),
        ("ddim5", 1.0, "plannerrft_paper_text"),
        ("ddim5_project_noise", 0.5, "project_noise_scale_0_5"),
    ],
)
def test_evaluation_composes_every_sampler_profile(profile: str, scale: float, label: str) -> None:
    config_dir = Path(__file__).resolve().parents[3] / "configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        config = compose(config_name="evaluation/no_traffic", overrides=[f"sampler={profile}"])

    report = sampler_report(parse_sampler_config(config.sampler))
    assert report.initial_noise_scale == scale
    assert report.parity_label == label


def test_evaluation_defaults_to_none_guidance_and_composes_active_profile() -> None:
    config_dir = Path(__file__).resolve().parents[3] / "configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        baseline = compose(config_name="evaluation/no_traffic")
        active = compose(
            config_name="evaluation/no_traffic",
            overrides=["sampler=ddim5", "guidance=orthogonal_reference"],
        )

    assert parse_guidance_config(baseline.guidance).name == "none"
    parsed_guidance = parse_guidance_config(active.guidance)
    parsed_sampler = parse_sampler_config(active.sampler)
    assert isinstance(parsed_guidance, OrthogonalReferenceGuidanceConfig)
    validate_guidance_sampler(parsed_guidance, parsed_sampler)


def test_traffic_matrix_composes_joblib_execution_grid() -> None:
    config_dir = Path(__file__).resolve().parents[3] / "configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        config = compose(config_name="evaluation/traffic_matrix", return_hydra_config=True)

    assert config.evaluation.profile == "matrix"
    assert list(config.evaluation.matrix.seeds) == [0, 1, 2]
    assert list(config.evaluation.matrix.traffic_densities) == [0.05, 0.10]
    assert config.evaluation.execution.mode == "parallel"
    assert config.hydra.launcher.n_jobs == 2
    assert config.hydra.launcher.backend == "loky"
    assert config.video.enabled is False
