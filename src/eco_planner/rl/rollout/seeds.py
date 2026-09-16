"""Deterministic independent rollout random streams."""

import numpy as np

_SEED_NAMESPACE = 6_002_024


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
