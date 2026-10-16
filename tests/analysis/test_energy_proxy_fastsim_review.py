"""Window reconstruction, physics proxy, FASTSim replay and diagnostic tests."""

from __future__ import annotations

import math

import numpy as np
import pytest

from eco_planner._repository import CONFIG_ROOT
from eco_planner.experiments.energy_proxy.fastsim_review import (
    FastSimWindowMeasurer,
    _paired_provider_contrast,
    build_diagnostic_profiles,
    diagnostic_study,
    distance_preserving_windows,
    load_review_config,
)
from eco_planner.experiments.energy_proxy.physics_proxy import measure_physics_proxy

FROZEN = load_review_config(CONFIG_ROOT / "experiments/energy-proxy/fastsim-review.yaml")


def test_frozen_review_config_loads():
    assert FROZEN.study_name == "issue113_task2_fastsim_fixed_trace_review"
    assert FROZEN.checkpoint_labels == ["final"]
    assert FROZEN.contrast_primary == ["a1", "a2"]
    assert FROZEN.contrast_context == ["a0", "a1"]
    assert FROZEN.include_baseline is True
    assert FROZEN.bootstrap.confidence_level == 0.95
    assert FROZEN.bootstrap.n_resamples == 10000
    assert FROZEN.bootstrap.bootstrap_seed == 0
    cycle = FROZEN.fastsim
    assert cycle.vehicle_resource == "2012_Ford_Fusion.yaml"
    assert cycle.grade == 0.0
    assert cycle.ambient_temperature_k == 295.15
    assert cycle.initial_elevation_m == 121.92
    assert cycle.warm_engine is True
    assert cycle.primary_window_s == 1.0
    assert cycle.secondary_window_s == 0.5
    assert cycle.ramp_accel_mps2 == 0.9
    assert cycle.ramp_timestep_s == 0.5
    assert cycle.hold_window_count == 4


def test_review_config_rejects_invalid_windows(tmp_path):
    from omegaconf import OmegaConf

    base = FROZEN.model_dump()
    bad_window = {**base, "fastsim": {**base["fastsim"], "primary_window_s": 1.05}}
    OmegaConf.save(bad_window, tmp_path / "bad-window.yaml")
    with pytest.raises(ValueError):
        load_review_config(tmp_path / "bad-window.yaml")
    equal_windows = {**base, "fastsim": {**base["fastsim"], "secondary_window_s": 1.0}}
    OmegaConf.save(equal_windows, tmp_path / "equal-windows.yaml")
    with pytest.raises(ValueError):
        load_review_config(tmp_path / "equal-windows.yaml")


def test_distance_preserving_windows_are_exact():
    rng = np.random.default_rng(0)
    speeds = rng.uniform(0.0, 14.0, size=137)
    distances = speeds * 0.1
    window_speeds, durations = distance_preserving_windows(speeds, distances, 1.0)
    # 13 full windows; the trailing 7 substeps merge into the last one
    assert window_speeds.size == 13
    assert durations[-1] == pytest.approx(1.7)
    assert float((window_speeds * durations).sum(dtype=np.float64)) == pytest.approx(
        float(distances.sum(dtype=np.float64)), rel=1e-12
    )
    assert window_speeds[0] == pytest.approx(float(distances[:10].sum()) / 1.0)
    merged_distance = float(distances[120:].sum())
    assert window_speeds[-1] == pytest.approx(merged_distance / 1.7)
    # an exact multiple of the window leaves no partial tail to merge
    exact_speeds, exact_durations = distance_preserving_windows(speeds[:130], distances[:130], 1.0)
    assert exact_speeds.size == 13
    assert np.all(exact_durations == 1.0)
    assert exact_speeds[-1] == pytest.approx(float(distances[120:130].sum()) / 1.0)


def test_window_reconstruction_rejects_misaligned_input():
    with pytest.raises(ValueError):
        distance_preserving_windows(np.ones(5), np.ones(4), 1.0)
    with pytest.raises(ValueError):
        distance_preserving_windows(np.ones(5), np.ones(5), 0.35)


def test_physics_proxy_constant_cruise_is_hand_computable():
    config = FROZEN.physics_proxy
    speeds = np.full(30, 10.0)
    durations = np.full(30, 1.0)
    result = measure_physics_proxy(speeds, durations, initial_speed_mps=10.0, config=config)
    aero = (
        0.5
        * config.air_density_kg_m3
        * config.drag_coefficient
        * config.frontal_area_m2
        * 10.0**3
        * 30.0
    )
    rolling = (
        config.rolling_resistance_coefficient
        * config.vehicle_mass_kg
        * config.gravity_mps2
        * 10.0
        * 30.0
    )
    efficiency = config.driveline_efficiency * config.engine_peak_efficiency
    expected_fuel = (
        (aero + rolling) / efficiency
        + config.auxiliary_power_w * 30.0 / efficiency
        + config.idle_fuel_power_w * 30.0 / efficiency
    )
    assert result.aero_energy_j == pytest.approx(aero, rel=1e-12)
    assert result.rolling_energy_j == pytest.approx(rolling, rel=1e-12)
    assert result.inertial_energy_j == pytest.approx(0.0, abs=1e-12)
    assert result.braking_dissipated_j == pytest.approx(0.0, abs=1e-12)
    assert result.fuel_energy_j == pytest.approx(expected_fuel, rel=1e-12)
    assert result.distance_m == pytest.approx(300.0)
    assert result.fuel_energy_wh_per_km == pytest.approx(
        result.fuel_energy_j / 3_600.0 / 0.3, rel=1e-12
    )


def test_physics_proxy_inertial_term_is_kinetic_energy():
    config = FROZEN.physics_proxy
    speeds = np.asarray([8.0, 12.0, 6.0])
    durations = np.full(3, 1.0)
    result = measure_physics_proxy(speeds, durations, initial_speed_mps=5.0, config=config)
    previous = np.concatenate([[5.0], speeds[:-1]])
    kinetic = (
        0.5 * config.effective_mass_kg * float((speeds**2 - previous**2).sum(dtype=np.float64))
    )
    assert result.inertial_energy_j == pytest.approx(kinetic, rel=1e-12)


def test_physics_proxy_dissipates_braking_without_fuel():
    config = FROZEN.physics_proxy
    speeds = np.asarray([10.0, 8.0])
    durations = np.full(2, 1.0)
    result = measure_physics_proxy(speeds, durations, initial_speed_mps=10.0, config=config)
    assert result.braking_dissipated_j > 0.0
    efficiency = config.driveline_efficiency * config.engine_peak_efficiency
    # window 0 is steady 10 m/s (aero + rolling fuel); window 1 brakes hard
    # enough that the net tractive request is negative, so it burns fuel only
    # for the auxiliary load while the kinetic energy is dissipated
    steady = (
        0.5 * config.air_density_kg_m3 * config.drag_coefficient * config.frontal_area_m2 * 10.0**3
        + config.rolling_resistance_coefficient
        * config.vehicle_mass_kg
        * config.gravity_mps2
        * 10.0
    )
    expected = (steady + config.auxiliary_power_w * 2.0) / efficiency
    assert result.fuel_energy_j == pytest.approx(expected, rel=1e-12)
    # the dissipated energy equals the kinetic-energy drop minus the road-load
    # work the braking window still covers
    mid = 9.0
    kinetic_drop = config.effective_mass_kg * 2.0 * mid
    road_load = (
        0.5 * config.air_density_kg_m3 * config.drag_coefficient * config.frontal_area_m2 * mid**3
        + config.rolling_resistance_coefficient * config.vehicle_mass_kg * config.gravity_mps2 * mid
    )
    assert result.braking_dissipated_j == pytest.approx(kinetic_drop - road_load, rel=1e-12)


def test_ramp_prefix_starts_from_rest_and_reaches_window_speed():
    measurer = FastSimWindowMeasurer(FROZEN.fastsim)
    speeds, durations = measurer._prefix(10.625)
    assert speeds[0] == 0.0
    assert speeds[-1] == pytest.approx(10.625)
    assert np.all(np.diff(speeds) >= 0.0)
    assert durations.size == speeds.size
    assert (
        durations[-FROZEN.fastsim.hold_window_count :].tolist()
        == [0.5] * FROZEN.fastsim.hold_window_count
    )
    stationary, stationary_durations = measurer._prefix(0.0)
    assert stationary[0] == 0.0
    assert np.all(stationary == 0.0)
    assert stationary_durations.size == stationary.size


def test_fastsim_window_measurement_excludes_prefix_and_is_deterministic():
    measurer = FastSimWindowMeasurer(FROZEN.fastsim)
    speeds = np.full(10, 10.0)
    durations = np.full(10, 1.0)
    first = measurer.measure(speeds, durations)
    second = measurer.measure(speeds, durations)
    assert first["followed"]
    assert first["fuel_j"] > 0.0
    assert first["prefix_fuel_j"] > 0.0
    assert first["fuel_j"] == second["fuel_j"]
    single = measurer.measure(np.asarray([10.0]), np.asarray([1.0]))
    assert single["followed"]
    # steady-state windows burn nearly identical fuel, so ten windows cost
    # roughly ten times one window; the prefix is excluded either way
    assert first["fuel_j"] == pytest.approx(10.0 * single["fuel_j"], rel=0.05)
    assert first["prefix_fuel_j"] == pytest.approx(single["prefix_fuel_j"], rel=1e-9)


def test_fastsim_window_measurement_fails_closed_on_unfollowable_trace():
    measurer = FastSimWindowMeasurer(FROZEN.fastsim)
    speeds = np.asarray([10.0, 26.0, 10.0])
    durations = np.full(3, 1.0)
    result = measurer.measure(speeds, durations)
    assert result["followed"] is False
    assert result["fuel_j"] is None
    assert result["failure_message"] is not None
    assert "failed to meet speed trace" in result["failure_message"]


def test_diagnostic_profiles_are_frozen_and_feasible():
    profiles = build_diagnostic_profiles(FROZEN.diagnostics)
    assert set(profiles) == {
        "cruise",
        "fluctuation_fast",
        "fluctuation_slow",
        "stop_and_go",
        "stop_and_go_no_dwell",
        "hard_brake",
        "gentle_brake",
    }
    dt = FROZEN.diagnostics.timestep_s
    limit = max(
        FROZEN.diagnostics.launch_accel_mps2,
        abs(FROZEN.diagnostics.stop_and_go_brake_accel_mps2),
        abs(FROZEN.diagnostics.hard_brake_accel_mps2),
        abs(FROZEN.diagnostics.gentle_brake_accel_mps2),
    )
    for name, speeds in profiles.items():
        assert np.all(speeds >= 0.0), name
        assert np.all(np.isfinite(speeds)), name
        assert speeds[0] <= FROZEN.diagnostics.launch_accel_mps2 * dt + 1e-12, name
        accel = np.abs(np.diff(np.concatenate([[speeds[0]], speeds]))) / dt
        assert accel.max() <= limit + 1e-9, name
    fast, slow = profiles["fluctuation_fast"], profiles["fluctuation_slow"]
    assert fast.size == slow.size
    assert float((fast * dt).sum()) == pytest.approx(float((slow * dt).sum()), rel=1e-9)
    dwell = FROZEN.diagnostics.stop_and_go_dwell_steps
    assert profiles["stop_and_go"].size - profiles["stop_and_go_no_dwell"].size == dwell


def test_diagnostic_study_reports_provider_invariants():
    result = diagnostic_study(FROZEN)
    for name, row in result["profiles"].items():
        assert row["proxy"]["total_ml"] > 0.0, name
        assert row["fastsim"]["fuel_j"] > 0.0, name
        assert row["physics"]["fuel_j"] > 0.0, name
    invariants = result["invariants"]
    idle = invariants["idle_visibility"]
    assert idle["proxy_total_ml_absolute_difference"] == 0.0
    assert idle["fuel_absolute_difference"]["fastsim"] > 0.0
    assert idle["fuel_absolute_difference"]["physics"] > 0.0
    fluctuation = invariants["fluctuation_rate_sensitivity"]
    assert fluctuation["time_mean_speed_gap_mps"] < 0.05
    assert abs(fluctuation["intensity_relative_difference"]["proxy"]) < 1e-9
    assert fluctuation["intensity_relative_difference"]["fastsim"] > 0.0
    assert fluctuation["intensity_relative_difference"]["physics"] > 0.0


def test_physics_proxy_config_matches_fusion_resource():
    import fastsim

    vehicle = fastsim.Vehicle.from_resource(FROZEN.fastsim.vehicle_resource)
    payload = vehicle.to_pydict()
    chassis = payload["chassis"]
    conv = payload["pt_type"]["Conv"]
    config = FROZEN.physics_proxy
    assert config.vehicle_mass_kg == payload["mass_kilograms"]
    assert config.drag_coefficient == chassis["drag_coef"]
    assert config.frontal_area_m2 == chassis["frontal_area_square_meters"]
    assert config.rolling_resistance_coefficient == chassis["wheel_rr_coef"]
    assert config.wheel_inertia_kg_m2 == chassis["wheel_inertia_kilogram_square_meters"]
    assert config.wheel_count == chassis["num_wheels"]
    assert config.wheel_radius_m == chassis["wheel_radius_meters"]
    assert config.auxiliary_power_w == payload["pwr_aux_base_watts"]
    assert config.driveline_efficiency == conv["transmission"]["eff_interp"]
    efficiency_values = conv["fc"]["eff_interp_from_pwr_out"]["data"]["values"]["data"]
    assert config.engine_peak_efficiency == max(efficiency_values)


def _fake_episode(
    arm: str,
    label: str,
    seed: int | None,
    scenario: str,
    family: str,
    *,
    proxy_ml_per_km: float,
    fastsim_wh_per_km: float | None,
    physics_wh_per_km: float,
    followed: bool = True,
) -> dict:
    return {
        "arm": arm,
        "checkpoint_label": label,
        "training_seed": seed,
        "scenario": scenario,
        "map_sequence": family,
        "noise_seed": 760025,
        "facts": {
            "distance_m": 100.0,
            "time_mean_speed_mps": 10.0,
            "resolutions": {
                "1": {
                    "proxy": {"ml_per_km": proxy_ml_per_km},
                    "fastsim": {
                        "followed": followed,
                        "wh_per_km": fastsim_wh_per_km if followed else None,
                    },
                    "physics": {"wh_per_km": physics_wh_per_km},
                }
            },
        },
    }


def test_paired_provider_contrast_respects_fail_closed_availability():
    episodes = []
    for scenario, family in (("s16", "S"), ("s17", "S"), ("sc16", "SC"), ("sc17", "SC")):
        episodes.append(
            _fake_episode(
                "a1",
                "final",
                0,
                scenario,
                family,
                proxy_ml_per_km=50.0,
                fastsim_wh_per_km=1000.0,
                physics_wh_per_km=900.0,
            )
        )
        followed = scenario != "sc17"
        episodes.append(
            _fake_episode(
                "a2",
                "final",
                0,
                scenario,
                family,
                proxy_ml_per_km=49.0,
                fastsim_wh_per_km=970.0 if followed else None,
                physics_wh_per_km=880.0,
                followed=followed,
            )
        )
    contrast = _paired_provider_contrast(
        episodes, "a1", "a2", "final", FROZEN, "1", descriptive=False
    )
    seed_row = contrast["per_seed"]["0"]
    assert seed_row["pair_count"] == 4
    assert seed_row["providers"]["proxy"]["available_pair_count"] == 4
    assert seed_row["providers"]["physics"]["available_pair_count"] == 4
    assert seed_row["providers"]["fastsim"]["available_pair_count"] == 3
    unavailable = seed_row["providers"]["fastsim"]["unavailable_pairs"]
    assert unavailable == [{"scenario": "sc17", "reason": "fastsim_not_followed"}]
    proxy_effect = seed_row["providers"]["proxy"]["effects"]["ln_intensity"]
    assert proxy_effect["estimate"] == pytest.approx(math.log(49.0 / 50.0), rel=1e-12)
    fastsim_effect = seed_row["providers"]["fastsim"]["effects"]["ln_intensity"]
    assert fastsim_effect["estimate"] == pytest.approx(math.log(970.0 / 1000.0), rel=1e-12)
    physics_effect = seed_row["providers"]["physics"]["effects"]["ln_intensity"]
    assert physics_effect["estimate"] == pytest.approx(math.log(880.0 / 900.0), rel=1e-12)
    shared = seed_row["shared"]["effects"]["distance_m"]
    assert shared["estimate"] == 0.0
    assert contrast["across_seeds"]["proxy_ln_intensity"]["all_seeds_same_nonzero_sign"]


def test_paired_provider_contrast_pairs_baseline_against_every_seed():
    episodes = []
    for scenario, family in (("s16", "S"), ("sc16", "SC")):
        episodes.append(
            _fake_episode(
                "a0",
                "baseline",
                None,
                scenario,
                family,
                proxy_ml_per_km=48.0,
                fastsim_wh_per_km=1010.0,
                physics_wh_per_km=910.0,
            )
        )
        for seed in (0, 1):
            episodes.append(
                _fake_episode(
                    "a1",
                    "final",
                    seed,
                    scenario,
                    family,
                    proxy_ml_per_km=50.0 + seed,
                    fastsim_wh_per_km=1000.0 + 10.0 * seed,
                    physics_wh_per_km=900.0 + 10.0 * seed,
                )
            )
    contrast = _paired_provider_contrast(
        episodes, "a0", "a1", "final", FROZEN, "1", descriptive=False
    )
    assert contrast["reference_has_training_seed"] is False
    assert sorted(contrast["per_seed"]) == ["0", "1"]
    for seed, row in contrast["per_seed"].items():
        assert row["providers"]["proxy"]["available_pair_count"] == 2
        effect = row["providers"]["proxy"]["effects"]["ln_intensity"]
        assert effect["estimate"] == pytest.approx(math.log((50.0 + int(seed)) / 48.0), rel=1e-12)
