"""Small validation primitives for strict environment boundaries."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def is_real_scalar(value: object) -> bool:
    """Return whether value is a Python or NumPy real scalar, excluding bool."""

    return not isinstance(value, (bool, np.bool_)) and isinstance(
        value, (int, float, np.integer, np.floating)
    )


def require_finite_real_scalar(value: object, name: str) -> float:
    """Return a finite real scalar or raise a boundary-specific error."""

    if not is_real_scalar(value):
        raise TypeError(f"{name} must be a numeric scalar")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def require_finite_array(
    value: object,
    name: str,
    *,
    dtype: np.dtype[object],
    shape: Sequence[int],
) -> np.ndarray:
    """Return an exact NumPy array contract after type, shape, and finite-value checks."""

    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray")
    if value.dtype != dtype:
        raise TypeError(f"{name} must use numpy.{dtype.name}")
    expected_shape = tuple(shape)
    if value.shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")
    return value
