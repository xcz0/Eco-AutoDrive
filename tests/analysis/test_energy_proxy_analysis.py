"""Synthetic traces test the proxy decomposition, counterfactuals and parity logic."""

from __future__ import annotations

import numpy as np

from eco_planner._repository import CONFIG_ROOT
from eco_planner.envs.domain.energy import EnergyTrace, MetaDriveFuelProxyProvider
from eco_planner.evaluation.artifacts.models import EnergySummary, SpeedSummary
from eco_planner.experiments.energy_proxy.analysis import (
    AnalysisConfig,
    CounterfactualConfig,
    ParityConfig,
    _pair_deltas,
    counterfactual_study,
    intensity_decomposition,
    kinematics_descriptives,
    load_analysis_config,
    parity_checks,
    proxy_step_fuel_ml,
)
from tests.evaluation.helpers import _episode

PARITY = ParityConfig(rtol=1e-10, atol=1e-12)


def _trace(speeds: np.ndarray, distances: np.ndarray, timestep_s: float = 0.1) -> EnergyTrace:
    points = np.concatenate([[speeds[0]], speeds])
    return EnergyTrace(
        time_s=np.arange(points.size, dtype=np.float64) * timestep_s,
        speed_mps=points,
        step_distance_m=distances,
    )


def test_frozen_analysis_config_loads():
    config = load_analysis_config(CONFIG_ROOT / "experiments/energy-proxy/analysis.yaml")
    assert isinstance(config, AnalysisConfig)
    assert config.checkpoint_labels == ["final"]
    assert config.contrast_primary == ["a1", "a2"]
    assert config.contrast_context == ["a0", "a1"]
    assert config.bootstrap.n_resamples == 10000
    assert config.bootstrap.bootstrap_seed == 0
    assert config.bootstrap.confidence_level == 0.95
    assert config.parity_include_baseline is True
    assert config.counterfactuals.step_count == 100


def test_folded_formula_matches_provider():
    speeds = np.array([5.0, 7.5, 0.0, 12.0, 3.25])
    distances = speeds * 0.1
    provider = MetaDriveFuelProxyProvider()
    metrics = provider.measure(_trace(speeds, distances))
    expected = proxy_step_fuel_ml(speeds, distances)
    assert metrics.fuel_ml is not None
    assert np.allclose(metrics.fuel_ml, float(expected.sum(dtype=np.float64)), rtol=0, atol=0)
    assert np.allclose(
        proxy_step_fuel_ml(speeds, distances),
        3.25 * np.exp(0.01 * speeds * 3.6) * (distances / 1_000.0) / 100.0 * 1_000.0,
        rtol=1e-12,
        atol=1e-15,
    )


def test_intensity_identity_is_exact():
    rng = np.random.default_rng(0)
    speeds = np.sort(rng.uniform(0.0, 14.0, size=250))
    distances = speeds * 0.1 + 1e-3
    decomposition = intensity_decomposition(speeds, distances)
    assert abs(decomposition["identity_residual"]) <= 1e-12
    intensity = decomposition["intensity_ml_per_km"]
    assert np.isclose(
        decomposition["ln_intensity"],
        np.log(1000.0 * decomposition["total_ml"] / decomposition["distance_m"]),
        rtol=0,
        atol=1e-12,
    )
    weights = distances / distances.sum()
    assert np.isclose(decomposition["mu_d"], float((weights * speeds).sum()), rtol=0, atol=1e-15)
    assert intensity is not None and intensity > 32.5


def test_reordering_preserves_intensity():
    speeds = np.array([4.0, 11.0, 7.0, 13.5, 2.0, 9.5])
    distances = speeds * 0.1
    original = intensity_decomposition(speeds, distances)
    reordered = intensity_decomposition(speeds[::-1].copy(), distances[::-1].copy())
    for key in ("mu_d", "speed_shape_ln", "intensity_ml_per_km", "total_ml"):
        assert np.isclose(original[key], reordered[key], rtol=1e-12, atol=1e-15)


def test_same_mean_speed_different_fluctuation_changes_intensity():
    constant = np.full(40, 10.0)
    fluctuating = 10.0 + 3.0 * np.array([-1.0, 0.0, 1.0, 0.0])[np.arange(40) % 4]
    left = intensity_decomposition(constant, constant * 0.1)
    right = intensity_decomposition(fluctuating, fluctuating * 0.1)
    assert np.isclose(constant.mean(), fluctuating.mean(), rtol=0, atol=1e-12)
    assert right["intensity_ml_per_km"] > left["intensity_ml_per_km"]


def test_zero_distance_is_undefined_intensity():
    speeds = np.zeros(10)
    distances = np.zeros(10)
    decomposition = intensity_decomposition(speeds, distances)
    assert decomposition["total_ml"] == 0.0
    assert decomposition["intensity_ml_per_km"] is None
    assert decomposition["mu_d"] is None
    metrics = MetaDriveFuelProxyProvider().measure(_trace(speeds, distances))
    assert metrics.fuel_ml == 0.0
    assert metrics.fuel_ml_per_km is None


def test_stationary_tail_is_invisible_to_proxy_totals():
    speeds = np.full(10, 10.0)
    with_tail = np.concatenate([speeds, np.zeros(5)])
    base = intensity_decomposition(speeds, speeds * 0.1)
    tail = intensity_decomposition(with_tail, with_tail * 0.1)
    assert tail["total_ml"] == base["total_ml"]
    assert tail["intensity_ml_per_km"] == base["intensity_ml_per_km"]
    assert with_tail.mean() < speeds.mean()


def test_counterfactual_study_invariants():
    config = CounterfactualConfig(
        timestep_s=0.1,
        step_count=100,
        base_speed_mps=10.0,
        fluctuation_amplitude_mps=3.0,
        stationary_step_count=50,
    )
    result = counterfactual_study(config)
    invariants = result["invariants"]
    assert invariants["reordering_invariance"]["invariant"]
    assert invariants["fluctuation_sensitivity"]["time_mean_speed_equal"]
    assert invariants["fluctuation_sensitivity"]["intensity_strictly_increases"]
    assert invariants["stationary_tail_invariance"]["total_ml_equal"]
    assert invariants["stationary_tail_invariance"]["intensity_equal"]
    assert invariants["stationary_tail_invariance"]["time_mean_speed_changes"]
    assert invariants["stationary_undefined_intensity"]["undefined_intensity_confirmed"]
    assert invariants["stationary_undefined_intensity"]["total_ml"] == 0.0
    for profile in result["profiles"].values():
        assert profile["provider_matches_independent_formula"]


def test_pair_deltas_identity_residual_is_float_noise():
    left = intensity_decomposition(np.full(20, 10.0), np.full(20, 1.0))
    right = intensity_decomposition(np.full(20, 9.6), np.full(20, 0.96))
    facts = []
    for row in (left, right):
        facts.append(
            {
                "intensity_ml_per_km": row["intensity_ml_per_km"],
                "total_ml": row["total_ml"],
                "distance_m": row["distance_m"],
                "route_completion": 0.5,
                "time_mean_speed_mps": 10.0,
                "mu_d": row["mu_d"],
                "stopped_fraction": 0.0,
                "accel_rms_mps2": 0.1,
                "jerk_rms_mps3": 0.2,
                "accel_decel_fraction": 0.3,
                "ln_intensity": row["ln_intensity"],
                "speed_shape_ln": row["speed_shape_ln"],
            }
        )
    deltas = _pair_deltas({"facts": facts[0]}, {"facts": facts[1]})
    assert abs(deltas["identity_residual"]) <= 1e-12
    assert np.isclose(
        deltas["ln_intensity"],
        deltas["speed_level_term"] + deltas["speed_shape_term"],
        rtol=0,
        atol=1e-12,
    )
    assert np.isclose(deltas["speed_level_term"], 0.036 * deltas["mu_d"], rtol=0, atol=1e-15)


def test_kinematics_descriptives():
    constant = kinematics_descriptives(np.full(10, 8.0))
    assert constant["accel_rms_mps2"] == 0.0
    assert constant["accel_decel_fraction"] == 0.0
    ramp = kinematics_descriptives(np.arange(10, dtype=np.float64))
    assert np.isclose(ramp["accel_mean_mps2"], 10.0)
    assert np.isclose(ramp["accel_decel_fraction"], 0.0)
    single = kinematics_descriptives(np.array([5.0]))
    assert single["accel_rms_mps2"] is None
    assert single["jerk_rms_mps3"] is None


def _summary_for(speeds: np.ndarray, distances: np.ndarray):
    fuel = proxy_step_fuel_ml(speeds, distances)
    total = float(fuel.sum(dtype=np.float64))
    distance = float(distances.sum(dtype=np.float64))
    episode = _episode(seed=0, distance_m=distance, energy_ml=total)
    metrics = episode.metrics.model_copy(
        update={
            "speed_mps": SpeedSummary(
                minimum=float(speeds.min()), mean=float(speeds.mean()), maximum=float(speeds.max())
            ),
            "distance_m": distance,
            "energy": EnergySummary(
                metric="metadrive_fuel_proxy",
                total_ml=total,
                distance_m=distance,
                ml_per_km=1000.0 * total / distance,
            ),
        }
    )
    return episode.model_copy(
        update={
            "metrics": metrics,
            "simulator_steps": int(speeds.size),
            "environment_steps_including_warmup": int(speeds.size),
        }
    )


def test_parity_checks_pass_on_consistent_stream():
    speeds = np.array([5.0, 6.0, 4.5, 7.0, 7.0, 3.0, 8.0, 9.0, 6.5, 5.5])
    distances = speeds * 0.1
    stored = proxy_step_fuel_ml(speeds, distances)
    record = parity_checks(
        speed=speeds,
        step_distance=distances,
        stored_fuel=stored,
        state_speed=speeds,
        state_distance=distances,
        stopped_fraction=0.0,
        warmup_steps=0,
        summary=_summary_for(speeds, distances),
        parity=PARITY,
    )
    assert record["passed"]
    assert record["step_mismatch_indices"] == []
    assert record["max_step_relative_error"] <= 1e-12


def test_parity_checks_detect_tampering():
    speeds = np.array([5.0, 6.0, 4.5, 7.0, 7.0, 3.0, 8.0, 9.0, 6.5, 5.5])
    distances = speeds * 0.1
    stored = proxy_step_fuel_ml(speeds, distances)
    stored[3] += 1e-6
    record = parity_checks(
        speed=speeds,
        step_distance=distances,
        stored_fuel=stored,
        state_speed=speeds,
        state_distance=distances,
        stopped_fraction=0.0,
        warmup_steps=0,
        summary=_summary_for(speeds, distances),
        parity=PARITY,
    )
    assert not record["passed"]
    assert 3 in record["step_mismatch_indices"]
    assert not record["step_formula_matches_stored_stream"]


def test_parity_checks_reject_wrong_speed_endpoint():
    speeds = np.array([5.0, 6.0, 4.5, 7.0, 7.0, 3.0, 8.0, 9.0, 6.5, 5.5])
    distances = speeds * 0.1
    stored = proxy_step_fuel_ml(speeds, distances)
    shifted = np.concatenate([[speeds[0]], speeds[:-1]])
    record = parity_checks(
        speed=speeds,
        step_distance=distances,
        stored_fuel=stored,
        state_speed=shifted,
        state_distance=distances,
        stopped_fraction=0.0,
        warmup_steps=0,
        summary=_summary_for(speeds, distances),
        parity=PARITY,
    )
    assert not record["speed_endpoint_matches_state_velocity"]
    assert not record["passed"]
