from __future__ import annotations

import torch

from eco_planner.models.baseline_sampler import BaselineDpmSampler


def _sample_with_call_counts() -> tuple[torch.Tensor, int, int]:
    model_calls = 0
    constraint_calls = 0

    def model(sample: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        nonlocal model_calls
        model_calls += 1
        assert tuple(timestep.shape) == (sample.shape[0],)
        return torch.zeros_like(sample)

    def constrain(sample: torch.Tensor) -> torch.Tensor:
        nonlocal constraint_calls
        constraint_calls += 1
        constrained = sample.clone()
        constrained[:, 0] = 1.0
        return constrained

    result = BaselineDpmSampler().sample(torch.zeros((2, 8)), model, constrain)
    return result, model_calls, constraint_calls


def test_baseline_sampler_is_deterministic_and_applies_constraints() -> None:
    first, model_calls, constraint_calls = _sample_with_call_counts()
    second, _, _ = _sample_with_call_counts()

    assert torch.equal(first, second)
    assert torch.equal(first[:, 0], torch.ones(2))
    assert model_calls == 11
    assert constraint_calls == 12
