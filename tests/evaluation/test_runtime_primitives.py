from __future__ import annotations

import torch

from eco_planner.runtime.host_transfer import HostTransfer
from eco_planner.runtime.random import sample_batched_standard_normal


def test_batched_standard_normal_matches_slot_loop_and_generator_states() -> None:
    actual_generators = [torch.Generator().manual_seed(seed) for seed in (11, 17, 23)]
    expected_generators = [torch.Generator().manual_seed(seed) for seed in (11, 17, 23)]

    actual = sample_batched_standard_normal(
        actual_generators,
        (2, 3, 4),
        device=torch.device("cpu"),
    )
    expected = torch.cat(
        [torch.randn((1, 2, 3, 4), generator=generator) for generator in expected_generators]
    )

    assert torch.equal(actual, expected)
    for actual_generator, expected_generator in zip(
        actual_generators, expected_generators, strict=True
    ):
        assert torch.equal(actual_generator.get_state(), expected_generator.get_state())


def test_cpu_host_transfer_separates_execution_and_audit_payloads() -> None:
    prediction = torch.arange(2 * 11 * 80 * 4, dtype=torch.float32).reshape(2, 11, 80, 4)
    transfer = HostTransfer(torch.device("cpu"))

    deferred = transfer.defer({"prediction": (prediction, torch.float32)})
    trajectories = transfer.execution_trajectories(prediction)
    assert trajectories.ego.shape == (2, 80, 4)
    torch.testing.assert_close(torch.from_numpy(trajectories.ego), prediction[:, 0])
    torch.testing.assert_close(deferred.resolve()["prediction"], prediction)
