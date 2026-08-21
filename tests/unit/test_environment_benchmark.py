from __future__ import annotations

import pytest
from benchmarking.common import EnvironmentBenchmarkConfig
from benchmarking.environment import BenchmarkAcceptanceError, _validate_baselines


def _config(**update: object) -> EnvironmentBenchmarkConfig:
    payload: dict[str, object] = {
        "map": "S",
        "seed": 0,
        "map_query_radius_m": 100.0,
        "traffic_density": 0.05,
        "history_warmup_steps": 20,
        "timing_warmup_cycles": 1,
        "measured_cycles": 1,
        "repeats": 1,
        "traffic_baseline_ms": 20.0,
        "no_traffic_baseline_ms": 5.0,
        "traffic_required_improvement_fraction": 0.2,
        "no_traffic_allowed_regression_fraction": 0.05,
    }
    payload.update(update)
    return EnvironmentBenchmarkConfig.model_validate(payload)


def test_environment_baselines_use_configured_relative_thresholds() -> None:
    report: dict[str, object] = {
        "traffic": {"samples": [16.0], "median": 16.0, "minimum": 16.0, "maximum": 16.0},
        "no_traffic": {"samples": [5.25], "median": 5.25, "minimum": 5.25, "maximum": 5.25},
    }

    _validate_baselines(report, _config())

    with pytest.raises(BenchmarkAcceptanceError, match="traffic median"):
        _validate_baselines(report, _config(traffic_baseline_ms=19.0))
