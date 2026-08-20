import pytest
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


def test_normalizers_reuse_cpu_constants_for_matching_dtype() -> None:
    state_normalizer = StateNormalizer(torch.zeros(1), torch.full((1,), 2.0))
    observation_normalizer = ObservationNormalizer(
        {"lanes": {"mean": [1.0, 2.0], "std": [2.0, 4.0]}}
    )
    state = torch.ones(1)
    lanes = torch.tensor([[3.0, 6.0]])

    state_normalizer.inverse(state)
    observation_normalizer({"lanes": lanes})
    state_constants = state_normalizer._cached_constants[(state.device, state.dtype)]
    observation_constants = observation_normalizer._cached_constants["lanes"][
        (lanes.device, lanes.dtype)
    ]

    state_normalizer.inverse(state.clone())
    observation_normalizer({"lanes": lanes.clone()})

    assert state_normalizer._cached_constants[(state.device, state.dtype)] is state_constants
    assert (
        observation_normalizer._cached_constants["lanes"][(lanes.device, lanes.dtype)]
        is observation_constants
    )


@pytest.mark.gpu
def test_normalizers_cache_cuda_constants_per_dtype() -> None:
    state_normalizer = StateNormalizer(torch.zeros(1), torch.full((1,), 2.0))
    observation_normalizer = ObservationNormalizer(
        {"lanes": {"mean": [1.0, 2.0], "std": [2.0, 4.0]}}
    )

    for dtype in (torch.float32, torch.bfloat16):
        state = torch.ones(1, dtype=dtype, device="cuda")
        lanes = torch.tensor([[3.0, 6.0]], dtype=dtype, device="cuda")
        state_normalizer.inverse(state)
        observation_normalizer({"lanes": lanes})

        state_constants = state_normalizer._cached_constants[(state.device, dtype)]
        observation_constants = observation_normalizer._cached_constants["lanes"][
            (state.device, dtype)
        ]
        assert state_constants[0].dtype == dtype
        assert observation_constants[0].dtype == dtype
        assert state_constants[0].device == state.device
        assert observation_constants[0].device == state.device
