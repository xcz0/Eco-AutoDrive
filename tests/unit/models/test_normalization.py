import torch

from eco_planner.models.config import ObservationNormalizer, StateNormalizer


def test_observation_normalizer_preserves_padding() -> None:
    normalizer = ObservationNormalizer({"lanes": {"mean": [1.0, 2.0], "std": [2.0, 4.0]}})
    lanes = torch.tensor([[[0.0, 0.0], [3.0, 6.0]]])

    normalized = normalizer({"lanes": lanes})["lanes"]

    torch.testing.assert_close(normalized, torch.tensor([[[0.0, 0.0], [1.0, 1.0]]]))
    assert normalizer.feature_dimension("lanes") == 2


def test_state_normalizer_inverts_normalized_trajectory() -> None:
    normalizer = StateNormalizer(torch.zeros((11, 1, 4)), torch.full((11, 1, 4), 2.0))

    physical = normalizer.inverse(torch.ones((1, 11, 2, 4)))

    torch.testing.assert_close(physical, torch.full((1, 11, 2, 4), 2.0))
