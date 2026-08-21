"""Small validation primitives for strict environment boundaries."""

from __future__ import annotations

import numpy as np


def is_real_scalar(value: object) -> bool:
    """Return whether value is a Python or NumPy real scalar, excluding bool."""

    return not isinstance(value, (bool, np.bool_)) and isinstance(
        value, (int, float, np.integer, np.floating)
    )
