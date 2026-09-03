from __future__ import annotations

import math

import numpy as np
import pytest

from eco_planner.energy import (
    EnergyTrace,
    FASTSimEnergyConfig,
    FASTSimEnergyProvider,
    MetaDriveFuelProxyProvider,
)


def _fastsim_config(vehicle_resource: str = "2012_Ford_Fusion.yaml") -> FASTSimEnergyConfig:
    return FASTSimEnergyConfig(
        vehicle_resource=vehicle_resource,
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


def test_metadrive_proxy_matches_upstream_formula() -> None:
    trace = EnergyTrace(
        time_s=np.asarray([0.0, 0.1]),
        speed_mps=np.asarray([0.0, 5.0]),
        step_distance_m=np.asarray([0.5]),
    )
    expected_ml = 3.25 * math.exp(0.01 * 18.0) * 0.0005 / 100.0 * 1_000.0

    metrics = MetaDriveFuelProxyProvider().measure(trace)

    assert metrics.metric == "metadrive_fuel_proxy"
    assert metrics.energy_j is None
    assert metrics.fuel_ml == pytest.approx(expected_ml)
    assert metrics.fuel_ml_per_km == pytest.approx(expected_ml * 2_000.0)


def test_metadrive_proxy_handles_zero_distance_and_low_speed() -> None:
    zero = MetaDriveFuelProxyProvider().measure(
        EnergyTrace(
            time_s=np.asarray([0.0, 0.1]),
            speed_mps=np.asarray([0.0, 0.0]),
            step_distance_m=np.asarray([0.0]),
        )
    )
    low_speed = MetaDriveFuelProxyProvider().measure(
        EnergyTrace(
            time_s=np.asarray([0.0, 0.1, 0.2]),
            speed_mps=np.asarray([0.0, 0.01, 0.02]),
            step_distance_m=np.asarray([0.001, 0.002]),
        )
    )

    assert zero.fuel_ml == 0.0
    assert zero.fuel_ml_per_km is None
    assert low_speed.fuel_ml is not None and low_speed.fuel_ml > 0.0
    assert low_speed.fuel_ml_per_km is not None


def test_metadrive_proxy_accumulates_complete_trace() -> None:
    trace = _short_trace()
    expected_ml = sum(
        3.25 * math.exp(0.01 * speed_mps * 3.6) * distance_m / 1_000.0 / 100.0 * 1_000.0
        for speed_mps, distance_m in zip(trace.speed_mps[1:], trace.step_distance_m, strict=True)
    )

    metrics = MetaDriveFuelProxyProvider().measure(trace)

    assert metrics.fuel_ml == pytest.approx(expected_ml)


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


def test_fastsim_rejects_non_conventional_vehicle() -> None:
    with pytest.raises(ValueError, match="conventional vehicle"):
        FASTSimEnergyProvider(_fastsim_config("2016 Nissan Leaf 30 kWh thrml.yaml"))


def test_fastsim_rejects_cycle_distance_disagreement() -> None:
    trace = EnergyTrace(
        time_s=np.asarray([0.0, 1.0]),
        speed_mps=np.asarray([0.0, 1.0]),
        step_distance_m=np.asarray([2.0]),
    )

    with pytest.raises(ValueError, match="cycle distance does not match"):
        FASTSimEnergyProvider(_fastsim_config()).measure(trace)


def test_proxy_and_fastsim_keep_sources_and_units_separate() -> None:
    trace = _short_trace()

    proxy = MetaDriveFuelProxyProvider().measure(trace)
    fastsim = FASTSimEnergyProvider(_fastsim_config()).measure(trace)

    assert proxy.distance_m == fastsim.distance_m
    assert proxy.metric != fastsim.metric
    assert proxy.fuel_ml is not None and proxy.energy_j is None
    assert fastsim.energy_j is not None and fastsim.fuel_ml is None
