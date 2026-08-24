from __future__ import annotations

import numpy as np
import pytest

from eco_planner.envs.execution import metadrive_fuel_proxy_step_energy_ml


def test_metadrive_fuel_proxy_matches_upstream_formula() -> None:
    start = np.array([0.0, 0.0], dtype=np.float64)
    end = np.array([0.5, 0.0], dtype=np.float64)
    speed_mps = 5.0
    expected_ml = 3.25 * np.exp(0.01 * 18.0) * 0.0005 / 100.0 * 1000.0

    assert metadrive_fuel_proxy_step_energy_ml(start, end, speed_mps) == pytest.approx(
        expected_ml
    )
