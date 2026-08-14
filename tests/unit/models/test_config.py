from __future__ import annotations

import json
from pathlib import Path

import pytest

from eco_planner.models.checkpoint.config import OfficialDiffusionPlannerConfig


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


def test_config_rejects_non_integer_dimension(
    tmp_path: Path, official_config_args: dict[str, object]
) -> None:
    official_config_args["future_len"] = 80.0
    with pytest.raises(ValueError, match="integer"):
        OfficialDiffusionPlannerConfig.from_json(_write_args(tmp_path, official_config_args))


def test_config_rejects_wrong_normalizer_channel_count(
    tmp_path: Path, official_config_args: dict[str, object]
) -> None:
    observation = official_config_args["observation_normalizer"]
    assert isinstance(observation, dict)
    lanes = observation["lanes"]
    assert isinstance(lanes, dict)
    lanes["mean"] = [0.0]
    with pytest.raises(ValueError, match="length 12"):
        OfficialDiffusionPlannerConfig.from_json(_write_args(tmp_path, official_config_args))


def test_config_rejects_invalid_state_normalizer_mapping(
    tmp_path: Path, official_config_args: dict[str, object]
) -> None:
    official_config_args["state_normalizer"] = {"mean": []}
    with pytest.raises(ValueError, match="state normalizer"):
        OfficialDiffusionPlannerConfig.from_json(_write_args(tmp_path, official_config_args))


def test_config_rejects_nonpositive_standard_deviation(
    tmp_path: Path, official_config_args: dict[str, object]
) -> None:
    observation = official_config_args["observation_normalizer"]
    assert isinstance(observation, dict)
    lanes = observation["lanes"]
    assert isinstance(lanes, dict)
    standard_deviations = lanes["std"]
    assert isinstance(standard_deviations, list)
    standard_deviations[0] = 0.0
    with pytest.raises(ValueError, match="positive"):
        OfficialDiffusionPlannerConfig.from_json(_write_args(tmp_path, official_config_args))
