from __future__ import annotations

import pytest
import torch

from eco_planner.models.checkpoint.config import OfficialDiffusionPlannerConfig
from eco_planner.models.runtime.validation import (
    validate_official_observation,
    validate_standard_normal_noise,
)


def test_observation_contract_returns_batch(
    stage0_observation: dict[str, torch.Tensor],
    official_model_config: OfficialDiffusionPlannerConfig,
) -> None:
    assert (
        validate_official_observation(
            stage0_observation, torch.device("cpu"), official_model_config
        )
        == 1
    )


def test_observation_contract_rejects_missing_field(
    stage0_observation: dict[str, torch.Tensor],
    official_model_config: OfficialDiffusionPlannerConfig,
) -> None:
    del stage0_observation["lanes"]
    with pytest.raises(ValueError, match="missing required fields"):
        validate_official_observation(
            stage0_observation, torch.device("cpu"), official_model_config
        )


def test_observation_contract_rejects_mismatched_batch(
    stage0_observation: dict[str, torch.Tensor],
    official_model_config: OfficialDiffusionPlannerConfig,
) -> None:
    stage0_observation["static_objects"] = torch.zeros((2, 5, 10))
    with pytest.raises(ValueError, match="share a batch"):
        validate_official_observation(
            stage0_observation, torch.device("cpu"), official_model_config
        )


def test_observation_contract_rejects_empty_batch(
    stage0_observation: dict[str, torch.Tensor],
    official_model_config: OfficialDiffusionPlannerConfig,
) -> None:
    empty = {name: value[:0] for name, value in stage0_observation.items()}
    with pytest.raises(ValueError, match="positive"):
        validate_official_observation(empty, torch.device("cpu"), official_model_config)


def test_observation_contract_rejects_wrong_dtype(
    stage0_observation: dict[str, torch.Tensor],
    official_model_config: OfficialDiffusionPlannerConfig,
) -> None:
    stage0_observation["ego_current_state"] = stage0_observation["ego_current_state"].double()
    with pytest.raises(TypeError, match="float32"):
        validate_official_observation(
            stage0_observation, torch.device("cpu"), official_model_config
        )


def test_noise_contract_rejects_wrong_shape() -> None:
    noise = torch.zeros((1, 11, 79, 4))
    with pytest.raises(ValueError, match="shape"):
        validate_standard_normal_noise(
            noise, batch=1, participants=11, future_len=80, device=torch.device("cpu")
        )


def test_noise_contract_rejects_nonfinite_value() -> None:
    noise = torch.zeros((1, 11, 80, 4))
    noise[0, 0, 0, 0] = torch.inf
    with pytest.raises(ValueError, match="finite"):
        validate_standard_normal_noise(
            noise, batch=1, participants=11, future_len=80, device=torch.device("cpu")
        )
