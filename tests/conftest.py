from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from eco_planner.evaluation.config import RuntimeConfig
from eco_planner.evaluation.runtime import (
    FabricInferenceRuntime,
    create_fabric_inference_runtime,
)
from eco_planner.models.config import OfficialDiffusionPlannerConfig
from eco_planner.models.guidance import (
    NoGuidanceConfig,
    OrthogonalReferenceGuidanceConfig,
)
from eco_planner.models.pretrained import (
    CheckpointLoadReport,
    PretrainedDiffusionPlanner,
    load_official_diffusion_planner,
)
from eco_planner.models.sampling_config import Ddim5SamplerConfig, Dpm10SamplerConfig


@pytest.fixture
def official_config_args() -> dict[str, object]:
    observation_dimensions = {
        "ego_current_state": 10,
        "neighbor_agents_past": 11,
        "static_objects": 10,
        "lanes": 12,
        "lanes_speed_limit": 1,
        "route_lanes": 12,
        "route_lanes_speed_limit": 1,
    }
    return {
        "future_len": 80,
        "time_len": 21,
        "agent_state_dim": 11,
        "agent_num": 32,
        "static_objects_state_dim": 10,
        "static_objects_num": 5,
        "lane_len": 20,
        "lane_state_dim": 12,
        "lane_num": 70,
        "map_len": 10,
        "map_state_dim": 4,
        "map_num": 5,
        "route_len": 20,
        "route_state_dim": 12,
        "route_num": 25,
        "encoder_drop_path_rate": 0.1,
        "decoder_drop_path_rate": 0.1,
        "device": "cuda",
        "encoder_depth": 3,
        "decoder_depth": 3,
        "num_heads": 6,
        "hidden_dim": 192,
        "diffusion_model_type": "x_start",
        "predicted_neighbor_num": 10,
        "state_normalizer": {
            "mean": [[[10.0, 0.0, 0.0, 0.0]] for _ in range(11)],
            "std": [[[20.0, 20.0, 1.0, 1.0]] for _ in range(11)],
        },
        "observation_normalizer": {
            name: {"mean": [0.0] * size, "std": [1.0] * size}
            for name, size in observation_dimensions.items()
        },
    }


@pytest.fixture
def official_model_config(
    tmp_path: Path, official_config_args: dict[str, object]
) -> OfficialDiffusionPlannerConfig:
    args_path = tmp_path / "args.json"
    args_path.write_text(json.dumps(official_config_args), encoding="utf-8")
    return OfficialDiffusionPlannerConfig.from_json(args_path)


@pytest.fixture
def stage0_observation() -> dict[str, torch.Tensor]:
    device = torch.device("cpu")
    observation = {
        "ego_current_state": torch.zeros((1, 10), dtype=torch.float32, device=device),
        "neighbor_agents_past": torch.zeros((1, 32, 21, 11), dtype=torch.float32, device=device),
        "static_objects": torch.zeros((1, 5, 10), dtype=torch.float32, device=device),
        "lanes": torch.zeros((1, 70, 20, 12), dtype=torch.float32, device=device),
        "lanes_speed_limit": torch.zeros((1, 70, 1), dtype=torch.float32, device=device),
        "lanes_has_speed_limit": torch.zeros((1, 70, 1), dtype=torch.bool, device=device),
        "route_lanes": torch.zeros((1, 25, 20, 12), dtype=torch.float32, device=device),
    }
    observation["ego_current_state"][0] = torch.tensor(
        [0.0, 0.0, 1.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=device
    )
    timesteps = torch.arange(21, dtype=torch.float32, device=device)
    neighbor = observation["neighbor_agents_past"][0, 0]
    neighbor[:, 0] = 12.0 + timesteps
    neighbor[:, 2] = 1.0
    neighbor[:, 4] = 10.0
    neighbor[:, 6] = 1.8
    neighbor[:, 7] = 4.8
    neighbor[:, 8] = 1.0
    points = torch.arange(20, dtype=torch.float32, device=device)
    for name in ("lanes", "route_lanes"):
        lane = observation[name][0, 0]
        lane[:, 0] = points * 5.0
        lane[:, 2] = 1.0
        lane[:, 4] = 1.75
        lane[:, 6] = -1.75
        lane[:, 8] = 1.0
    observation["lanes_speed_limit"][0, 0, 0] = 13.89
    observation["lanes_has_speed_limit"][0, 0, 0] = True
    return observation


@pytest.fixture(scope="session")
def stage0_checkpoint_dir() -> Path:
    checkpoint_dir = Path(__file__).resolve().parents[1] / "checkpoints" / "DP-Origin"
    required_assets = (checkpoint_dir / "args.json", checkpoint_dir / "model.pth")
    missing_assets = [str(path) for path in required_assets if not path.is_file()]
    if missing_assets:
        pytest.fail(f"stage 0 checkpoint assets are required: {', '.join(missing_assets)}")
    return checkpoint_dir


@pytest.fixture(scope="session")
def stage0_planner(
    stage0_checkpoint_dir: Path,
) -> tuple[PretrainedDiffusionPlanner, CheckpointLoadReport]:
    return load_official_diffusion_planner(
        stage0_checkpoint_dir / "args.json",
        stage0_checkpoint_dir / "model.pth",
        Dpm10SamplerConfig(),
    )


@pytest.fixture(scope="session")
def stage0_runtime(stage0_checkpoint_dir: Path) -> FabricInferenceRuntime:
    runtime_config = RuntimeConfig(accelerator="cpu", devices=1, precision="32-true", seed=0)
    return create_fabric_inference_runtime(
        runtime_config,
        Dpm10SamplerConfig(),
        NoGuidanceConfig(),
        stage0_checkpoint_dir / "args.json",
        stage0_checkpoint_dir / "model.pth",
    )


@pytest.fixture(scope="session")
def stage1_ddim_runtime(stage0_checkpoint_dir: Path) -> FabricInferenceRuntime:
    runtime_config = RuntimeConfig(accelerator="cpu", devices=1, precision="32-true", seed=0)
    sampler_config = Ddim5SamplerConfig(
        name="ddim5",
        num_steps=5,
        timesteps=(1.0, 0.8, 0.6, 0.4, 0.2, 0.0),
        initial_noise_scale=1.0,
        ddim_stochasticity=0.0,
        parity_label="plannerrft_paper_text",
    )
    return create_fabric_inference_runtime(
        runtime_config,
        sampler_config,
        NoGuidanceConfig(),
        stage0_checkpoint_dir / "args.json",
        stage0_checkpoint_dir / "model.pth",
    )


@pytest.fixture(scope="session")
def stage2_guided_runtime(stage0_checkpoint_dir: Path) -> FabricInferenceRuntime:
    runtime_config = RuntimeConfig(accelerator="cpu", devices=1, precision="32-true", seed=0)
    sampler_config = Ddim5SamplerConfig(
        name="ddim5",
        num_steps=5,
        timesteps=(1.0, 0.8, 0.6, 0.4, 0.2, 0.0),
        initial_noise_scale=1.0,
        ddim_stochasticity=0.0,
        parity_label="plannerrft_paper_text",
    )
    guidance_config = OrthogonalReferenceGuidanceConfig(
        name="orthogonal_reference",
        formula_label="centered_energy_gradient_delta_v1",
        lateral_scale=1.0,
        longitudinal_scale=0.0,
        lateral_max_offset_m=2.5,
        longitudinal_max_speed_fraction=0.25,
        trajectory_dt_s=0.1,
        gradient_step_coefficient=1.0,
        reference_refresh_cycles=1,
        share_scene_encoding=True,
        share_initial_noise=True,
        share_transition_noise=True,
        heading_norm_epsilon=1e-6,
        zero_speed_tolerance_mps=1e-6,
    )
    return create_fabric_inference_runtime(
        runtime_config,
        sampler_config,
        guidance_config,
        stage0_checkpoint_dir / "args.json",
        stage0_checkpoint_dir / "model.pth",
    )
