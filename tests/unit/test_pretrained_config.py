from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from eco_planner.models.config import OfficialDiffusionPlannerConfig
from eco_planner.models.normalization import ObservationNormalizer, StateNormalizer
from eco_planner.models.pretrained import _extract_ema_state_dict, _verify_hash


def _write_args(tmp_path: Path, contents: dict[str, object]) -> Path:
    path = tmp_path / "args.json"
    path.write_text(json.dumps(contents), encoding="utf-8")
    return path


def test_config_rejects_missing_field(
    tmp_path: Path, official_config_args: dict[str, object]
) -> None:
    official_config_args.pop("future_len")
    with pytest.raises(ValueError, match="keys mismatch"):
        OfficialDiffusionPlannerConfig.from_json(_write_args(tmp_path, official_config_args))


def test_config_rejects_route_length_mismatch(
    tmp_path: Path, official_config_args: dict[str, object]
) -> None:
    official_config_args["route_len"] = 19
    with pytest.raises(ValueError, match="route_len"):
        OfficialDiffusionPlannerConfig.from_json(_write_args(tmp_path, official_config_args))


def test_config_rejects_nonpositive_standard_deviation(
    tmp_path: Path, official_config_args: dict[str, object]
) -> None:
    observation_normalizer = official_config_args["observation_normalizer"]
    assert isinstance(observation_normalizer, dict)
    lanes = observation_normalizer["lanes"]
    assert isinstance(lanes, dict)
    standard_deviations = lanes["std"]
    assert isinstance(standard_deviations, list)
    standard_deviations[0] = 0.0
    with pytest.raises(ValueError, match="positive"):
        OfficialDiffusionPlannerConfig.from_json(_write_args(tmp_path, official_config_args))


def test_observation_normalizer_preserves_padding() -> None:
    normalizer = ObservationNormalizer({"lanes": {"mean": [10.0, 0.0], "std": [20.0, 1.0]}})
    lanes = torch.tensor([[[0.0, 0.0], [10.0, 1.0]]])
    result = normalizer({"lanes": lanes})["lanes"]
    assert torch.equal(result[0, 0], torch.zeros(2))
    torch.testing.assert_close(result[0, 1], torch.tensor([0.0, 1.0]))


def test_state_normalizer_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match="shape"):
        StateNormalizer([0.0], [1.0])


def test_hash_verification_rejects_mismatch(tmp_path: Path) -> None:
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
