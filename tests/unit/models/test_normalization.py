from __future__ import annotations

import pytest
import torch

from eco_planner.models.normalization import ObservationNormalizer, StateNormalizer


def test_observation_normalizer_preserves_padding() -> None:
    normalizer = ObservationNormalizer({"lanes": {"mean": [10.0, 0.0], "std": [20.0, 1.0]}})
    lanes = torch.tensor([[[0.0, 0.0], [10.0, 1.0]]])
    result = normalizer({"lanes": lanes})["lanes"]
    assert torch.equal(result[0, 0], torch.zeros(2))
    torch.testing.assert_close(result[0, 1], torch.tensor([0.0, 1.0]))


def test_observation_normalizer_rejects_scalar_metadata() -> None:
    with pytest.raises(ValueError, match="vectors"):
        ObservationNormalizer({"lanes": {"mean": 0.0, "std": 1.0}})


def test_observation_normalizer_rejects_non_mapping_metadata() -> None:
    with pytest.raises(ValueError, match="mean and std"):
        ObservationNormalizer({"lanes": [0.0, 1.0]})  # type: ignore[dict-item]


def test_state_normalizer_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match="shape"):
        StateNormalizer([0.0], [1.0])
