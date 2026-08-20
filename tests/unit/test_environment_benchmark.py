from __future__ import annotations

from typing import cast

import pytest

from eco_planner.benchmarks import env
from eco_planner.models import OfficialDiffusionPlannerConfig


def test_environment_benchmark_is_importable_and_enforces_baselines(monkeypatch) -> None:
    def measure(_: OfficialDiffusionPlannerConfig, *, traffic: bool) -> env.BenchmarkMeasurement:
        median = 10.0 if traffic else 5.0
        return {
            "cycle_ms": [median],
            "median_cycle_ms": median,
            "minimum_cycle_ms": median,
            "maximum_cycle_ms": median,
        }

    monkeypatch.setattr(env, "_measure", measure)
    model_config = cast(OfficialDiffusionPlannerConfig, object())

    report = env.benchmark_environment(
        model_config, traffic_baseline_ms=20.0, no_traffic_baseline_ms=5.0
    )

    assert report["traffic"]["median_cycle_ms"] == 10.0
    assert report["no_traffic"]["median_cycle_ms"] == 5.0
    with pytest.raises(env.BenchmarkAcceptanceError, match="traffic median"):
        env.benchmark_environment(model_config, traffic_baseline_ms=10.0)
