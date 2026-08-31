from __future__ import annotations

import json
from pathlib import Path

import pytest

from eco_planner.models import OfficialDiffusionPlannerConfig


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
