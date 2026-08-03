from __future__ import annotations

from pathlib import Path

import pytest
import torch

from eco_planner.models.pretrained import (
    CheckpointLoadReport,
    PretrainedDiffusionPlanner,
    load_official_diffusion_planner,
)

ARGS_SHA256 = "7e62b89a50953f133d55484777e54490f7f24e58feec1efcf696bcc7b91bdf10"
CHECKPOINT_SHA256 = "7a441df91ebe1c912d8262010c40486da24f425f757e2b4228072e251ab67d45"


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


@pytest.fixture(scope="session")
def stage0_checkpoint_dir() -> Path:
    checkpoint_dir = Path(__file__).resolve().parents[1] / "checkpoints" / "DP-origin"
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
        ARGS_SHA256,
        CHECKPOINT_SHA256,
        torch.device("cpu"),
    )
