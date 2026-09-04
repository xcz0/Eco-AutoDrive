from __future__ import annotations

import numpy as np
import pytest

from eco_planner.envs.domain import EnergyTrace, FASTSimEnergyConfig, FASTSimEnergyProvider


def _fastsim_config() -> FASTSimEnergyConfig:
    return FASTSimEnergyConfig(
        vehicle_resource="2012_Ford_Fusion.yaml",
        grade=0.0,
        ambient_temperature_k=295.15,
        initial_elevation_m=121.92,
    )


def _short_trace() -> EnergyTrace:
    return EnergyTrace(
        time_s=np.arange(6, dtype=np.float64),
        speed_mps=np.arange(6, dtype=np.float64),
        step_distance_m=np.arange(1, 6, dtype=np.float64),
    )


def test_fastsim_conventional_vehicle_matches_locked_numeric_result() -> None:
    metrics = FASTSimEnergyProvider(_fastsim_config()).measure(_short_trace())

    assert metrics.metric == "fastsim_fuel_energy"
    assert metrics.distance_m == 15.0
    assert metrics.energy_j == pytest.approx(116_855.316236982)
    assert metrics.energy_wh == pytest.approx(metrics.energy_j / 3_600.0)
    assert metrics.energy_j_per_km == pytest.approx(metrics.energy_j * 1_000.0 / 15.0)
    assert metrics.fuel_ml is None


def test_fastsim_stationary_trace_keeps_idle_energy_without_intensity() -> None:
    trace = EnergyTrace(
        time_s=np.asarray([0.0, 1.0, 2.0]),
        speed_mps=np.zeros(3, dtype=np.float64),
        step_distance_m=np.zeros(2, dtype=np.float64),
    )

    metrics = FASTSimEnergyProvider(_fastsim_config()).measure(trace)

    assert metrics.energy_j is not None and metrics.energy_j > 0.0
    assert metrics.energy_j_per_km is None
    assert metrics.energy_wh_per_km is None
