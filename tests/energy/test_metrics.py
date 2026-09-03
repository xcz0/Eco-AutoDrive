from __future__ import annotations

import numpy as np
import pytest

from eco_planner.energy import EnergyMetrics, EnergyTrace


def test_energy_metrics_expose_consistent_units() -> None:
    metrics = EnergyMetrics(
        metric="fastsim_fuel_energy",
        distance_m=2_000.0,
        energy_j=7_200.0,
        fuel_ml=None,
    )

    assert metrics.energy_wh == 2.0
    assert metrics.energy_j_per_km == 3_600.0
    assert metrics.energy_wh_per_km == 1.0
    assert metrics.fuel_ml_per_km is None


def test_zero_distance_has_no_energy_intensity() -> None:
    metrics = EnergyMetrics(
        metric="metadrive_fuel_proxy",
        distance_m=0.0,
        energy_j=None,
        fuel_ml=0.0,
    )

    assert metrics.fuel_ml_per_km is None


@pytest.mark.parametrize(
    ("time_s", "speed_mps", "step_distance_m", "message"),
    [
        ([0.0], [0.0], [], "at least two"),
        ([0.0, 0.1], [0.0], [0.0], "identical shape"),
        ([0.0, 0.1], [0.0, 0.0], [], "align with transitions"),
        ([1.0, 1.1], [0.0, 0.0], [0.0], "start at zero"),
        ([0.0, 0.0], [0.0, 0.0], [0.0], "strictly increase"),
        ([0.0, 0.1], [0.0, -1.0], [0.0], "non-negative"),
        ([0.0, 0.1], [0.0, 0.0], [-1.0], "non-negative"),
        ([0.0, float("nan")], [0.0, 0.0], [0.0], "finite"),
    ],
)
def test_energy_trace_rejects_invalid_inputs(
    time_s: list[float],
    speed_mps: list[float],
    step_distance_m: list[float],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        EnergyTrace(
            time_s=np.asarray(time_s),
            speed_mps=np.asarray(speed_mps),
            step_distance_m=np.asarray(step_distance_m),
        )


@pytest.mark.parametrize(
    "updates",
    [
        {"distance_m": -1.0},
        {"energy_j": float("inf")},
        {"fuel_ml": -1.0},
        {"energy_j": None, "fuel_ml": None},
    ],
)
def test_energy_metrics_reject_invalid_outputs(updates: dict[str, object]) -> None:
    values: dict[str, object] = {
        "metric": "fastsim_fuel_energy",
        "distance_m": 1.0,
        "energy_j": 1.0,
        "fuel_ml": None,
    }
    values.update(updates)

    with pytest.raises(ValueError):
        EnergyMetrics(**values)  # type: ignore[arg-type]
