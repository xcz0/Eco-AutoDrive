"""Deterministic independent rollout random streams."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from .runtime import FabricRolloutRuntime

_SEED_NAMESPACE = 6_002_024


@dataclass(frozen=True)
class RolloutRandomStreams:
    """Persistent random streams and their provenance for one logical scenario."""

    noise_seed: int
    policy_action_seed: int
    diffusion: torch.Generator
    policy: torch.Generator

    def __post_init__(self) -> None:
        for seed, generator in (
            (self.noise_seed, self.diffusion),
            (self.policy_action_seed, self.policy),
        ):
            if type(seed) is not int or seed < 0 or generator.initial_seed() != seed:
                raise ValueError("rollout generator seed must match its logical slot provenance")


def create_rollout_streams(
    runtime: FabricRolloutRuntime,
    noise_seeds: tuple[int, ...],
    policy_seeds: tuple[int, ...],
) -> tuple[RolloutRandomStreams, ...]:
    return tuple(
        RolloutRandomStreams(
            noise_seed,
            policy_seed,
            runtime.new_noise_generator(noise_seed),
            runtime.new_policy_generator(policy_seed),
        )
        for noise_seed, policy_seed in zip(noise_seeds, policy_seeds, strict=True)
    )


def derive_rollout_seeds(
    training_seed: int, scenario_count: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if type(scenario_count) is not int or scenario_count <= 0:
        raise ValueError("scenario_count must be a positive integer")
    sequence = np.random.SeedSequence([_SEED_NAMESPACE, training_seed])
    values = tuple(
        int(child.generate_state(1, dtype=np.uint32)[0])
        for child in sequence.spawn(2 * scenario_count)
    )
    if len(set(values)) != len(values):
        raise RuntimeError("seed derivation produced duplicate random streams")
    return values[:scenario_count], values[scenario_count:]
