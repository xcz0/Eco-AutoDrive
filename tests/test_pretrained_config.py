from __future__ import annotations

import json

import pytest
import torch

from eco_planner.models.config import OfficialDiffusionPlannerConfig
from eco_planner.models.normalization import ObservationNormalizer, StateNormalizer
from eco_planner.models.pretrained import _extract_ema_state_dict, _verify_hash


def _valid_args() -> dict[str, object]:
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


def _write_args(tmp_path: pytest.TempPathFactory, contents: dict[str, object]) -> object:
    path = tmp_path / "args.json"
    path.write_text(json.dumps(contents), encoding="utf-8")
    return path


def test_config_rejects_missing_field(tmp_path: pytest.TempPathFactory) -> None:
    contents = _valid_args()
    contents.pop("future_len")
    with pytest.raises(ValueError, match="keys mismatch"):
        OfficialDiffusionPlannerConfig.from_json(_write_args(tmp_path, contents))


def test_config_rejects_route_length_mismatch(tmp_path: pytest.TempPathFactory) -> None:
    contents = _valid_args()
    contents["route_len"] = 19
    with pytest.raises(ValueError, match="route_len"):
        OfficialDiffusionPlannerConfig.from_json(_write_args(tmp_path, contents))


def test_config_rejects_nonpositive_standard_deviation(tmp_path: pytest.TempPathFactory) -> None:
    contents = _valid_args()
    contents["observation_normalizer"]["lanes"]["std"][0] = 0.0  # type: ignore[index]
    with pytest.raises(ValueError, match="positive"):
        OfficialDiffusionPlannerConfig.from_json(_write_args(tmp_path, contents))


def test_observation_normalizer_preserves_padding() -> None:
    normalizer = ObservationNormalizer({"lanes": {"mean": [10.0, 0.0], "std": [20.0, 1.0]}})
    lanes = torch.tensor([[[0.0, 0.0], [10.0, 1.0]]])
    result = normalizer({"lanes": lanes})["lanes"]
    assert torch.equal(result[0, 0], torch.zeros(2))
    torch.testing.assert_close(result[0, 1], torch.tensor([0.0, 1.0]))


def test_state_normalizer_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match="shape"):
        StateNormalizer([0.0], [1.0])


def test_hash_verification_rejects_mismatch(tmp_path: pytest.TempPathFactory) -> None:
    path = tmp_path / "asset.bin"
    path.write_bytes(b"stage-zero")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        _verify_hash(path, "0" * 64)


def test_ema_loader_rejects_invalid_prefix() -> None:
    with pytest.raises(ValueError, match="module"):
        _extract_ema_state_dict({"model": {}, "ema_state_dict": {"invalid": torch.ones(1)}})


def test_ema_loader_rejects_missing_ema() -> None:
    with pytest.raises(ValueError, match="exactly model"):
        _extract_ema_state_dict({"model": {}})


def test_ema_loader_rejects_incomplete_state_dict() -> None:
    with pytest.raises(ValueError, match="276 tensors"):
        _extract_ema_state_dict({"model": {}, "ema_state_dict": {"module.partial": torch.ones(1)}})
