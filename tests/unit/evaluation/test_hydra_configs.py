from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from eco_planner.evaluation.config import parse_evaluation_config
from eco_planner.models import (
    OrthogonalReferenceGuidanceConfig,
    parse_guidance_config,
    parse_sampler_config,
    sampler_report,
)


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
        config = compose(
            config_name="experiment/evaluate_no_traffic_full",
            overrides=[f"planner/sampler={profile}"],
        )

    report = sampler_report(parse_sampler_config(config.sampler))
    assert report.initial_noise_scale == scale
    assert report.parity_label == label


def test_evaluation_defaults_to_none_guidance_and_composes_active_profile() -> None:
    config_dir = Path(__file__).resolve().parents[3] / "configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        baseline = compose(config_name="experiment/evaluate_no_traffic_full")
        active = compose(
            config_name="experiment/evaluate_no_traffic_full",
            overrides=["planner/sampler=ddim5", "planner/guidance=orthogonal_reference"],
        )

    assert parse_guidance_config(baseline.guidance).name == "none"
    parsed_guidance = parse_guidance_config(active.guidance)
    parse_sampler_config(active.sampler)
    assert isinstance(parsed_guidance, OrthogonalReferenceGuidanceConfig)


def test_energy_guidance_profiles_compose_with_expected_longitudinal_scales() -> None:
    config_dir = Path(__file__).resolve().parents[3] / "configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        configs = [
            compose(
                config_name="experiment/evaluate_energy_structures",
                overrides=[f"planner/guidance=energy_longitudinal_{name}"],
            )
            for name in ("negative", "zero", "positive")
        ]

    assert [parse_guidance_config(config.guidance).longitudinal_scale for config in configs] == [
        -1.0,
        0.0,
        1.0,
    ]


def test_traffic_matrix_composes_joblib_execution_grid() -> None:
    config_dir = Path(__file__).resolve().parents[3] / "configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        config = compose(config_name="experiment/evaluate_traffic_matrix", return_hydra_config=True)

    assert config.evaluation.profile == "matrix"
    assert list(config.evaluation.matrix.seeds) == [0, 1, 2]
    assert list(config.evaluation.matrix.traffic_densities) == [0.05, 0.10]
    assert config.evaluation.execution.mode == "parallel"
    assert config.hydra.launcher.n_jobs == config.resources.evaluation_job_worker_count
    assert config.hydra.launcher.backend == "loky"
    assert config.video.enabled is False


def test_traffic_matrix_launcher_tracks_resolved_job_worker_count() -> None:
    config_dir = Path(__file__).resolve().parents[3] / "configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        config = compose(
            config_name="experiment/evaluate_traffic_matrix",
            overrides=["resources.evaluation_job_worker_count=3"],
            return_hydra_config=True,
        )

    assert config.resources.evaluation_job_worker_count == 3
    assert config.hydra.launcher.n_jobs == 3


def test_resource_profile_switch_keeps_evaluation_semantics_and_changes_execution_budget() -> None:
    config_dir = Path(__file__).resolve().parents[3] / "configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        baseline = parse_evaluation_config(compose(config_name="experiment/evaluate_traffic_full"))
        profiled = parse_evaluation_config(
            compose(
                config_name="experiment/evaluate_traffic_full",
                overrides=["resources=rtx_a4000"],
            )
        )

    assert profiled.resources is not None
    assert profiled.resources.name == "rtx_a4000"
    assert profiled.evaluation.execution.torch_threads_per_worker == 12
    assert profiled.evaluation.mode == baseline.evaluation.mode
    assert profiled.evaluation.profile == baseline.evaluation.profile
    assert profiled.sampler == baseline.sampler
    assert profiled.guidance == baseline.guidance


@pytest.mark.parametrize(
    ("name", "mode", "profile", "horizon", "video_enabled"),
    [
        ("evaluate_no_traffic_smoke", "no_traffic", "smoke", 20, False),
        ("evaluate_traffic_smoke", "traffic", "smoke", 100, False),
        ("evaluate_no_traffic_full", "no_traffic", "full", 200, False),
        ("evaluate_traffic_full", "traffic", "full", 3000, False),
        ("evaluate_no_traffic_matrix", "no_traffic", "matrix", 200, False),
        ("evaluate_traffic_matrix", "traffic", "matrix", 300, False),
        ("evaluate_energy_structures", "no_traffic", "energy_matrix", 300, False),
        ("evaluate_energy_speed_profile", "no_traffic", "energy_matrix", 300, False),
        ("evaluate_energy_traffic", "traffic", "energy_matrix", 600, False),
    ],
)
def test_experiment_profiles_compose_complete_evaluation_jobs(
    name: str, mode: str, profile: str, horizon: int, video_enabled: bool
) -> None:
    config_dir = Path(__file__).resolve().parents[3] / "configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        config = compose(config_name=f"experiment/{name}")

    parsed = parse_evaluation_config(config)
    assert parsed.evaluation.mode == mode
    assert parsed.evaluation.profile == profile
    assert parsed.evaluation.evaluated_horizon_steps == horizon
    assert parsed.video.enabled is video_enabled
