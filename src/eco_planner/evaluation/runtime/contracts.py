"""CPU-resident inference results shared by execution and trace recording."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HostGuidanceDiagnostics:
    lateral_target_offset_m: np.ndarray
    longitudinal_target_speed_fraction: np.ndarray
    longitudinal_target_speed_delta_mps: np.ndarray
    lateral_objective_delta: np.ndarray
    longitudinal_objective_delta: np.ndarray
    applied_gradient_l2: np.ndarray
    applied_gradient_max_abs: np.ndarray
    raw_neighbor_gradient_l2: np.ndarray
    zero_speed_count: np.ndarray


@dataclass(frozen=True)
class HostInferenceResult:
    """One planner result transferred to CPU exactly once."""

    initial_noise: np.ndarray
    prediction: np.ndarray
    reference_prediction: np.ndarray | None = None
    guidance_action: np.ndarray | None = None
    guidance_diagnostics: HostGuidanceDiagnostics | None = None

    @property
    def ego_trajectory(self) -> np.ndarray:
        """Return the batch-zero ego prediction as a view, not a copy."""

        return self.prediction[0, 0]
