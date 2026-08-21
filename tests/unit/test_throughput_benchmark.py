from __future__ import annotations

import pytest
from benchmarking.common import ScalingBenchmarkConfig, measurement
from pydantic import ValidationError


def test_measurement_reports_statistical_median_and_extrema() -> None:
    odd = measurement([3.0, 1.0, 2.0])
    even = measurement([4.0, 1.0, 3.0, 2.0])

    assert odd == {
        "samples": [3.0, 1.0, 2.0],
        "median": 2.0,
        "minimum": 1.0,
        "maximum": 3.0,
    }
    assert even["median"] == 2.5


def test_measurement_rejects_empty_samples() -> None:
    with pytest.raises(ValueError, match="at least one sample"):
        measurement([])


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_measurement_rejects_non_finite_samples(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        measurement([value])


def test_measurement_rejects_negative_samples() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        measurement([-0.1])


@pytest.mark.parametrize(
    "update",
    [
        {"batch_sizes": []},
        {"batch_sizes": [1, 0]},
        {"batch_sizes": [1, 1]},
        {"worker_counts": []},
        {"warmup_cycles": 0},
        {"measured_cycles": 0},
        {"repeats": 0},
    ],
)
def test_scaling_config_rejects_invalid_controls(update: dict[str, object]) -> None:
    payload: dict[str, object] = {
        "batch_sizes": [1, 2],
        "worker_counts": [1, 2],
        "warmup_cycles": 1,
        "measured_cycles": 1,
        "repeats": 1,
    }
    payload.update(update)

    with pytest.raises(ValidationError):
        ScalingBenchmarkConfig.model_validate(payload)
