"""Normalization rules"""

from __future__ import annotations

from collections.abc import Mapping
from copy import copy

import torch


class StateNormalizer:
    """Normalize and invert the joint ego/neighbor trajectory state."""

    def __init__(self, mean: object, std: object) -> None:
        self.mean = torch.as_tensor(mean, dtype=torch.float32)
        self.std = torch.as_tensor(std, dtype=torch.float32)
        if self.mean.shape != (11, 1, 4) or self.std.shape != (11, 1, 4):
            raise ValueError("state normalizer tensors must have shape [11, 1, 4]")
        if not torch.isfinite(self.mean).all() or not torch.isfinite(self.std).all():
            raise ValueError("state normalizer values must be finite")
        if torch.any(self.std <= 0):
            raise ValueError("state normalizer standard deviations must be positive")

    def inverse(self, data: torch.Tensor) -> torch.Tensor:
        return data * self.std.to(data.device) + self.mean.to(data.device)


class ObservationNormalizer:
    """Apply checkpoint normalization while preserving all-zero padding."""

    def __init__(self, normalization: Mapping[str, Mapping[str, object]]) -> None:
        parsed: dict[str, dict[str, torch.Tensor]] = {}
        for name, values in normalization.items():
            if not isinstance(values, Mapping) or set(values) != {"mean", "std"}:
                raise ValueError(f"normalization for {name!r} must contain exactly mean and std")
            mean = torch.as_tensor(values["mean"], dtype=torch.float32)
            std = torch.as_tensor(values["std"], dtype=torch.float32)
            if mean.ndim != 1 or mean.numel() == 0:
                raise ValueError(f"normalization values for {name!r} must be non-empty vectors")
            if mean.shape != std.shape:
                raise ValueError(f"normalization mean/std shape mismatch for {name!r}")
            if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
                raise ValueError(f"normalization values for {name!r} must be finite")
            if torch.any(std <= 0):
                raise ValueError(f"normalization standard deviations for {name!r} must be positive")
            parsed[name] = {"mean": mean, "std": std}
        self._normalization = parsed

    def __call__(self, data: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        normalized = copy(dict(data))
        for name, values in self._normalization.items():
            if name not in data:
                continue
            tensor = data[name]
            padding = torch.sum(torch.ne(tensor, 0), dim=-1) == 0
            result = (tensor - values["mean"].to(tensor.device)) / values["std"].to(tensor.device)
            result[padding] = 0
            normalized[name] = result
        return normalized

    def feature_dimension(self, name: str) -> int:
        """Return the checkpoint-defined feature width for one normalized field."""

        try:
            return int(self._normalization[name]["mean"].numel())
        except KeyError as error:
            raise KeyError(f"no normalization is defined for {name!r}") from error

    @property
    def feature_dimensions(self) -> dict[str, int]:
        """Return feature widths indexed by their checkpoint observation field."""

        return {name: int(values["mean"].numel()) for name, values in self._normalization.items()}
