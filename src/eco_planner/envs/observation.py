"""Single-environment observation collation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import torch


def to_cpu_torch_observation(arrays: Mapping[str, np.ndarray]) -> dict[str, torch.Tensor]:
    """Convert a complete NumPy observation to CPU tensors at the adapter boundary."""

    result: dict[str, torch.Tensor] = {}
    for name, value in arrays.items():
        if not isinstance(value, np.ndarray):
            raise TypeError(f"observation field {name} must be a numpy.ndarray")
        result[name] = torch.from_numpy(value)
    return result


def collate_observations(
    observations: Sequence[Mapping[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Stack same-schema single-environment observations into a planner batch."""

    if not observations:
        raise ValueError("cannot collate an empty observation sequence")
    names = observations[0].keys()
    expected_names = set(names)
    result: dict[str, torch.Tensor] = {}
    for index, observation in enumerate(observations):
        if set(observation) != expected_names:
            raise ValueError(f"observation {index} does not match the first observation schema")
    for name in names:
        values = [observation[name] for observation in observations]
        if not all(isinstance(value, torch.Tensor) for value in values):
            raise TypeError(f"observation field {name} must contain only torch tensors")
        result[name] = torch.stack(values)
    return result
