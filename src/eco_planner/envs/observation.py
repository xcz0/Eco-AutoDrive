"""Single-environment observation collation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch


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
