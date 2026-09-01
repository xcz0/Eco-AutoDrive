"""Shared random sampling primitives for independently owned generators."""

from __future__ import annotations

from collections.abc import Sequence

import torch


def sample_batched_standard_normal(
    generators: Sequence[torch.Generator],
    sample_shape: tuple[int, ...],
    *,
    device: torch.device,
) -> torch.Tensor:
    """Draw one sample per generator without changing per-generator consumption order."""

    return torch.cat(
        [
            torch.randn(
                (1, *sample_shape),
                dtype=torch.float32,
                device=device,
                generator=generator,
            )
            for generator in generators
        ]
    )
