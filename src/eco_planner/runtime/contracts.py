"""Cross-domain CPU execution contracts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HostExecutionResult:
    """Batched ego trajectories required before advancing simulator workers."""

    ego_trajectory: np.ndarray
