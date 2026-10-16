"""Issue #113 Task 2: offline physics-model review on fixed E-060 traces.

The study replays the executed E-060 speed/distance traces through the
existing FASTSim conventional-vehicle model and through a minimal
longitudinal-dynamics power proxy, and compares every provider's paired
A2-A1 (and A1-A0 context) direction against the MetaDrive fuel proxy on the
identical reconstructed input series.

Because the raw 0.1 s execution traces are kinematic waypoint outputs
(sub-step acceleration artifacts up to ~47 m/s^2) while FASTSim is a forward
powertrain simulation that starts from rest with a cold-start power limit,
the replayed cycle is a frozen reconstruction of the same trace:

* distance-preserving window-mean speeds (primary 1.0 s, secondary 0.5 s,
  trailing partial window kept with its own duration), so the cycle travels
  exactly the executed distance;
* a study-local warm-engine copy of the pinned vehicle (the traces are
  mid-trip segments of an already-cruising vehicle);
* a physical ramp prefix from rest to the first window speed plus a short
  hold, excluded from every reported window through per-step cumulative
  fuel accounting.

FASTSim speed-trace failures are expected and recorded per episode
(fail-closed): unfollowable episodes are unavailable for that provider's
paired statistics and are never silently replaced. Provider intensities are
never compared across units; only signs and within-provider paired relative
differences are compared.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictFloat, StrictInt

from eco_planner.contracts import SIMULATOR_STEP_S
from eco_planner.envs.domain.energy import EnergyTrace, MetaDriveFuelProxyProvider
from eco_planner.envs.domain.fastsim import FASTSimEnergyConfig, FASTSimEnergyProvider
from eco_planner.evaluation.artifacts import (
    CompletedEpisodeSummary,
    EpisodeSummary,
    load_episode_summary,
    load_trace_artifact,
)
from eco_planner.statistics import ScenarioBootstrapConfig, scenario_effect

from .analysis import _load_source, _read_json_mapping, proxy_step_fuel_ml
from .physics_proxy import PhysicsProxyConfig, measure_physics_proxy

_FUEL_HISTORY_PATH = "veh.pt_type.Conv.fc.history.energy_fuel_joules"
_FUEL_FINAL_STATE_PATH = "veh.pt_type.Conv.fc.state.energy_fuel_joules"
_WINDOW_DISTANCE_RESIDUAL_LIMIT = 1e-9
_PROVIDER_KEYS = ("proxy", "fastsim", "physics")
_SHARED_METRICS = ("distance_m", "time_mean_speed_mps")


class FastSimCycleConfig(BaseModel):
    """Frozen FASTSim vehicle, environment and cycle-reconstruction conditions."""

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid", allow_inf_nan=False)

    vehicle_resource: str = Field(min_length=1)
    grade: StrictFloat
    ambient_temperature_k: StrictFloat = Field(gt=0.0)
    initial_elevation_m: StrictFloat
    warm_engine: StrictBool
    primary_window_s: StrictFloat = Field(gt=0.0)
    secondary_window_s: StrictFloat = Field(gt=0.0)
    ramp_accel_mps2: StrictFloat = Field(gt=0.0)
    ramp_timestep_s: StrictFloat = Field(gt=0.0)
    hold_window_count: StrictInt = Field(gt=0)


class DiagnosticProfileConfig(BaseModel):
    """Frozen synthetic provider-response profiles (from rest, feasible)."""

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid", allow_inf_nan=False)

    timestep_s: StrictFloat = Field(gt=0.0)
    cruise_speed_mps: StrictFloat = Field(gt=0.0)
    launch_accel_mps2: StrictFloat = Field(gt=0.0)
    cruise_hold_steps: StrictInt = Field(gt=0)
    fluctuation_amplitude_mps: StrictFloat = Field(gt=0.0)
    fluctuation_period_fast_s: StrictFloat = Field(gt=0.0)
    fluctuation_period_slow_s: StrictFloat = Field(gt=0.0)
    fluctuation_duration_s: StrictFloat = Field(gt=0.0)
    stop_and_go_cruise_steps: StrictInt = Field(gt=0)
    stop_and_go_brake_accel_mps2: StrictFloat = Field(lt=0.0)
    stop_and_go_dwell_steps: StrictInt = Field(gt=0)
    stop_and_go_second_cruise_steps: StrictInt = Field(gt=0)
    brake_approach_speed_mps: StrictFloat = Field(gt=0.0)
    brake_cruise_steps: StrictInt = Field(gt=0)
    hard_brake_accel_mps2: StrictFloat = Field(lt=0.0)
    gentle_brake_accel_mps2: StrictFloat = Field(lt=0.0)
    brake_target_speed_mps: StrictFloat = Field(gt=0.0)
    brake_after_cruise_steps: StrictInt = Field(gt=0)


class ReviewConfig(BaseModel):
    """Frozen machine-readable pre-registration of the E-062 study."""

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid", allow_inf_nan=False)

    version: StrictInt
    study_name: str = Field(min_length=1)
    checkpoint_labels: list[str] = Field(min_length=1)
    contrast_primary: list[str] = Field(min_length=2, max_length=2)
    contrast_context: list[str] = Field(min_length=2, max_length=2)
    include_baseline: StrictBool
    bootstrap: ScenarioBootstrapConfig
    fastsim: FastSimCycleConfig
    physics_proxy: PhysicsProxyConfig
    diagnostics: DiagnosticProfileConfig

    @property
    def resolutions(self) -> tuple[float, float]:
        cfg = self.fastsim
        return (cfg.primary_window_s, cfg.secondary_window_s)


def load_review_config(path: Path) -> ReviewConfig:
    config = ReviewConfig.model_validate(OmegaConf.to_container(OmegaConf.load(path)))
    step = float(SIMULATOR_STEP_S)
    for window_s in config.resolutions:
        if abs(window_s / step - round(window_s / step)) > 1e-12:
            raise ValueError(f"analysis window {window_s} must be a multiple of {step} s")
    if config.fastsim.primary_window_s == config.fastsim.secondary_window_s:
        raise ValueError("primary and secondary analysis windows must differ")
    return config


def _window_key(window_s: float) -> str:
    return f"{window_s:g}"


def distance_preserving_windows(
    speed_mps: np.ndarray, step_distance_m: np.ndarray, window_s: float
) -> tuple[np.ndarray, np.ndarray]:
    """Distance-preserving window-mean reconstruction of one executed trace.

    Consecutive simulator substeps are grouped into windows of ``window_s``.
    A trailing partial group is merged into the last full window (the merged
    window keeps the summed distance and the summed real duration), so no
    sub-window-duration point can reimport the raw trace's kinematic
    quantization as an artificial acceleration spike. Every window mean is
    the executed distance of the window divided by its real duration, so
    ``sum(speeds * durations)`` reproduces the executed distance exactly up
    to floating-point summation.
    """

    speeds = np.asarray(speed_mps, dtype=np.float64)
    distance = np.asarray(step_distance_m, dtype=np.float64)
    if speeds.ndim != 1 or distance.ndim != 1 or distance.size != speeds.size or speeds.size == 0:
        raise ValueError("window reconstruction requires aligned non-empty trace arrays")
    step = float(SIMULATOR_STEP_S)
    substeps_per_window = int(round(window_s / step))
    if substeps_per_window <= 0 or abs(window_s / step - substeps_per_window) > 1e-12:
        raise ValueError(f"window {window_s} is not an integer number of {step} s substeps")
    window_speeds: list[float] = []
    window_durations: list[float] = []
    for start in range(0, speeds.size, substeps_per_window):
        end = min(start + substeps_per_window, speeds.size)
        group_distance = float(distance[start:end].sum(dtype=np.float64))
        group_duration = step * (end - start)
        partial = end - start < substeps_per_window
        if partial and window_speeds:
            merged_distance = window_speeds[-1] * window_durations[-1] + group_distance
            merged_duration = window_durations[-1] + group_duration
            window_speeds[-1] = merged_distance / merged_duration
            window_durations[-1] = merged_duration
        else:
            window_speeds.append(group_distance / group_duration)
            window_durations.append(group_duration)
    return (
        np.asarray(window_speeds, dtype=np.float64),
        np.asarray(window_durations, dtype=np.float64),
    )


class FastSimWindowMeasurer:
    """Replay window-mean speed series through FASTSim with ramp prefix.

    The vehicle is a study-local copy of the pinned conventional resource;
    with ``warm_engine`` the cold-start power ramp is replaced by the full
    engine power (mid-trip assumption). A fresh vehicle/simulation is built
    per measurement so no state leaks between episodes. Failures to follow
    the speed trace are returned, not raised.
    """

    def __init__(self, config: FastSimCycleConfig) -> None:
        import fastsim

        base = fastsim.Vehicle.from_resource(config.vehicle_resource)
        if base.veh_type() != "Conv":
            raise ValueError("FASTSim review requires a conventional vehicle")
        payload = base.to_pydict()
        if config.warm_engine:
            converter = payload["pt_type"]["Conv"]["fc"]
            converter["pwr_out_max_init_watts"] = converter["pwr_out_max_watts"]
        self._config = config
        self._vehicle_payload = payload

    def _prefix(self, target_speed_mps: float) -> tuple[np.ndarray, np.ndarray]:
        cfg = self._config
        timestep = cfg.ramp_timestep_s
        if target_speed_mps <= 0.0:
            ramp = np.zeros(0, dtype=np.float64)
        else:
            count = int(math.ceil(target_speed_mps / (cfg.ramp_accel_mps2 * timestep)))
            ramp = np.minimum(
                np.arange(1, count + 1, dtype=np.float64) * cfg.ramp_accel_mps2 * timestep,
                target_speed_mps,
            )
        hold = np.full(cfg.hold_window_count, target_speed_mps, dtype=np.float64)
        speeds = np.concatenate([[0.0], ramp, hold])
        # per-point durations: entry i is the duration of the cycle interval
        # ending at point i (entry 0 is unused because time starts at zero)
        durations = np.concatenate(
            [[0.0], np.full(ramp.size, timestep), np.full(hold.size, timestep)]
        )
        return speeds, durations

    def measure(self, window_speeds: np.ndarray, window_durations: np.ndarray) -> dict[str, Any]:
        """Run one full replay; the reported fuel excludes the ramp prefix."""

        import fastsim

        cfg = self._config
        speeds = np.asarray(window_speeds, dtype=np.float64)
        durations = np.asarray(window_durations, dtype=np.float64)
        if speeds.ndim != 1 or durations.ndim != 1 or speeds.size != durations.size:
            raise ValueError("FASTSim replay requires aligned window series")
        prefix_speeds, prefix_durations = self._prefix(float(speeds[0]) if speeds.size else 0.0)
        cycle_speeds = np.concatenate([prefix_speeds, speeds])
        per_point = np.concatenate([prefix_durations, durations])
        if per_point.size != cycle_speeds.size:
            raise ValueError("cycle point/duration misalignment")
        time_s = np.concatenate([[0.0], np.cumsum(per_point[1:])])
        zeros = [0.0] * cycle_speeds.size
        payload: dict[str, Any] = {
            "init_elev_meters": float(cfg.initial_elevation_m),
            "time_seconds": time_s.tolist(),
            "speed_meters_per_second": cycle_speeds.tolist(),
            "dist_meters": zeros,
            "grade": [float(cfg.grade)] * cycle_speeds.size,
            "elev_meters": [float(cfg.initial_elevation_m)] * cycle_speeds.size,
            "pwr_max_chrg_watts": list(zeros),
            "temp_amb_air_kelvin": [float(cfg.ambient_temperature_k)] * cycle_speeds.size,
            "pwr_solar_load_watts": list(zeros),
            "grade_interp": 0.0,
            "elev_interp": float(cfg.initial_elevation_m),
        }
        vehicle = fastsim.Vehicle.from_dict(self._vehicle_payload)
        vehicle.set_save_interval(1)
        simulation = fastsim.SimDrive(vehicle, fastsim.Cycle.from_dict(payload))
        started = time.perf_counter()
        try:
            simulation.walk()
        except RuntimeError as error:
            return {
                "followed": False,
                "fuel_j": None,
                "prefix_fuel_j": None,
                "failure_message": str(error)[:400],
                "elapsed_ms": (time.perf_counter() - started) * 1_000.0,
            }
        elapsed_ms = (time.perf_counter() - started) * 1_000.0
        frame = simulation.to_dataframe()
        history = frame[_FUEL_HISTORY_PATH].to_numpy().astype(np.float64)
        final = simulation.to_pydict(flatten=True)
        final_fuel = float(final[_FUEL_FINAL_STATE_PATH])
        if not np.isclose(history[-1], final_fuel, rtol=1e-9, atol=1e-6):
            raise ValueError("FASTSim fuel history is not cumulative; window accounting is invalid")
        prefix_point_count = prefix_speeds.size
        if history.size != cycle_speeds.size or prefix_point_count < 1:
            raise ValueError("FASTSim history does not align with the cycle points")
        prefix_fuel = float(history[prefix_point_count - 1])
        window_fuel = float(history[-1]) - prefix_fuel
        return {
            "followed": True,
            "fuel_j": window_fuel,
            "prefix_fuel_j": prefix_fuel,
            "failure_message": None,
            "elapsed_ms": elapsed_ms,
        }


def _episode_identity(
    arm: str, checkpoint_label: str, training_seed: int | None, episode_dir: Path
) -> tuple[dict[str, Any], EpisodeSummary]:
    summary = load_episode_summary(episode_dir / "summary.json")
    record: dict[str, Any] = {
        "arm": arm,
        "checkpoint_label": checkpoint_label,
        "training_seed": training_seed,
        "scenario": summary.scenario.name,
        "map_sequence": summary.scenario.map_sequence,
        "scenario_seed": summary.scenario.seed,
        "noise_seed": summary.noise_seed,
        "status": summary.status,
        "episode_dir": str(episode_dir),
    }
    return record, summary


def measure_episode(
    arm: str,
    checkpoint_label: str,
    training_seed: int | None,
    episode_dir: Path,
    config: ReviewConfig,
    measurer: FastSimWindowMeasurer,
) -> dict[str, Any]:
    """Measure every provider over one executed episode at each resolution."""

    record, summary = _episode_identity(arm, checkpoint_label, training_seed, episode_dir)
    trace = load_trace_artifact(episode_dir / "trace.npz")
    record["trace_status"] = trace.trace_status
    if not isinstance(summary, CompletedEpisodeSummary) or trace.trace_status != "complete":
        record["facts"] = None
        record["unavailable_reason"] = (
            f"status_{summary.status}"
            if not isinstance(summary, CompletedEpisodeSummary)
            else f"trace_status_{trace.trace_status}"
        )
        return record
    arrays = trace.arrays
    speed = arrays["executed_speed_mps"]
    step_distance = arrays["executed_step_distance_m"]
    distance_m = float(step_distance.sum(dtype=np.float64))
    raw_total_ml = float(proxy_step_fuel_ml(speed, step_distance).sum(dtype=np.float64))
    record["facts"] = {
        "step_count": int(speed.size),
        "distance_m": distance_m,
        "time_mean_speed_mps": float(speed.mean()),
        "arrive_dest": summary.metrics.arrive_dest,
        "terminated": summary.terminated,
        "truncated": summary.truncated,
        "terminal_reason": summary.terminal_reason,
        "raw_proxy": {
            "total_ml": raw_total_ml,
            "ml_per_km": None if distance_m == 0.0 else raw_total_ml * 1_000.0 / distance_m,
        },
        "resolutions": {},
    }
    for window_s in config.resolutions:
        key = _window_key(window_s)
        window_speeds, window_durations = distance_preserving_windows(
            speed, step_distance, window_s
        )
        window_distance = float((window_speeds * window_durations).sum(dtype=np.float64))
        residual = None if distance_m == 0.0 else abs(window_distance - distance_m) / distance_m
        if residual is not None and residual > _WINDOW_DISTANCE_RESIDUAL_LIMIT:
            raise ValueError(f"window reconstruction lost distance: relative residual {residual}")
        window_proxy_total = float(
            proxy_step_fuel_ml(window_speeds, window_speeds * window_durations).sum(
                dtype=np.float64
            )
        )
        fastsim_result = measurer.measure(window_speeds, window_durations)
        physics = measure_physics_proxy(
            window_speeds,
            window_durations,
            initial_speed_mps=float(window_speeds[0]),
            config=config.physics_proxy,
        )
        fastsim_fuel = fastsim_result["fuel_j"]
        record["facts"]["resolutions"][key] = {
            "window_count": int(window_speeds.size),
            "window_distance_residual_rel": residual,
            "window_speed_mean_mps": float(window_speeds.mean()),
            "proxy": {
                "total_ml": window_proxy_total,
                "ml_per_km": (
                    None if distance_m == 0.0 else window_proxy_total * 1_000.0 / distance_m
                ),
            },
            "fastsim": {
                "followed": fastsim_result["followed"],
                "fuel_j": fastsim_fuel,
                "wh_per_km": (
                    None
                    if fastsim_fuel is None or distance_m == 0.0
                    else fastsim_fuel / 3_600.0 * 1_000.0 / distance_m
                ),
                "prefix_fuel_j": fastsim_result["prefix_fuel_j"],
                "failure_message": fastsim_result["failure_message"],
                "elapsed_ms": fastsim_result["elapsed_ms"],
            },
            "physics": {
                "fuel_j": physics.fuel_energy_j,
                "wh_per_km": (
                    None
                    if distance_m == 0.0
                    else physics.fuel_energy_j / 3_600.0 * 1_000.0 / distance_m
                ),
                "tractive_j": physics.tractive_energy_j,
                "aero_j": physics.aero_energy_j,
                "rolling_j": physics.rolling_energy_j,
                "inertial_j": physics.inertial_energy_j,
                "braking_dissipated_j": physics.braking_dissipated_j,
                "auxiliary_fuel_j": physics.auxiliary_fuel_energy_j,
                "idle_fuel_j": physics.idle_fuel_energy_j,
            },
        }
    return record


def _measure_evaluation_directory(
    arm: str,
    checkpoint_label: str,
    training_seed: int | None,
    directory: Path,
    config: ReviewConfig,
    measurer: FastSimWindowMeasurer,
) -> list[dict[str, Any]]:
    summary = _read_json_mapping(directory / "summary.json")
    records = []
    for scenario in summary["workload"]["scenarios"]:
        episode_dir = directory / str(scenario["name"])
        if (
            not (episode_dir / "trace.npz").is_file()
            or not (episode_dir / "summary.json").is_file()
        ):
            raise ValueError(f"evaluation episode artifact is missing: {episode_dir}")
        records.append(
            measure_episode(arm, checkpoint_label, training_seed, episode_dir, config, measurer)
        )
    return records


def _segment_hold(speed: float, steps: int) -> np.ndarray:
    return np.full(steps, speed, dtype=np.float64)


def _segment_ramp(from_speed: float, to_speed: float, accel: float, timestep: float) -> np.ndarray:
    if accel == 0.0:
        raise ValueError("ramp segment requires a non-zero acceleration")
    delta = to_speed - from_speed
    if delta == 0.0:
        return np.zeros(0, dtype=np.float64)
    if delta * accel <= 0.0:
        raise ValueError("ramp segment acceleration must point toward the target speed")
    count = int(math.ceil(abs(delta) / (abs(accel) * timestep)))
    step_speeds = from_speed + np.sign(delta) * np.minimum(
        np.arange(1, count + 1, dtype=np.float64) * abs(accel) * timestep, abs(delta)
    )
    return step_speeds


def _segment_sine(
    base_speed: float, amplitude: float, period_s: float, duration_s: float, timestep: float
) -> np.ndarray:
    count = int(round(duration_s / timestep))
    times = np.arange(1, count + 1, dtype=np.float64) * timestep
    return base_speed + amplitude * np.sin(2.0 * math.pi * times / period_s)


def build_diagnostic_profiles(config: DiagnosticProfileConfig) -> dict[str, np.ndarray]:
    """Frozen synthetic step-speed profiles, all launched from rest."""

    dt = config.timestep_s
    launch = _segment_ramp(0.0, config.cruise_speed_mps, config.launch_accel_mps2, dt)
    profiles: dict[str, np.ndarray] = {
        "cruise": np.concatenate(
            [launch, _segment_hold(config.cruise_speed_mps, config.cruise_hold_steps)]
        ),
        "fluctuation_fast": np.concatenate(
            [
                launch,
                _segment_sine(
                    config.cruise_speed_mps,
                    config.fluctuation_amplitude_mps,
                    config.fluctuation_period_fast_s,
                    config.fluctuation_duration_s,
                    dt,
                ),
            ]
        ),
        "fluctuation_slow": np.concatenate(
            [
                launch,
                _segment_sine(
                    config.cruise_speed_mps,
                    config.fluctuation_amplitude_mps,
                    config.fluctuation_period_slow_s,
                    config.fluctuation_duration_s,
                    dt,
                ),
            ]
        ),
    }
    stop_and_go = np.concatenate(
        [
            launch,
            _segment_hold(config.cruise_speed_mps, config.stop_and_go_cruise_steps),
            _segment_ramp(config.cruise_speed_mps, 0.0, config.stop_and_go_brake_accel_mps2, dt),
            _segment_hold(0.0, config.stop_and_go_dwell_steps),
            _segment_ramp(0.0, config.cruise_speed_mps, config.launch_accel_mps2, dt),
            _segment_hold(config.cruise_speed_mps, config.stop_and_go_second_cruise_steps),
        ]
    )
    stop_and_go_no_dwell = np.concatenate(
        [
            launch,
            _segment_hold(config.cruise_speed_mps, config.stop_and_go_cruise_steps),
            _segment_ramp(config.cruise_speed_mps, 0.0, config.stop_and_go_brake_accel_mps2, dt),
            _segment_ramp(0.0, config.cruise_speed_mps, config.launch_accel_mps2, dt),
            _segment_hold(config.cruise_speed_mps, config.stop_and_go_second_cruise_steps),
        ]
    )
    profiles["stop_and_go"] = stop_and_go
    profiles["stop_and_go_no_dwell"] = stop_and_go_no_dwell
    approach_launch = _segment_ramp(
        0.0, config.brake_approach_speed_mps, config.launch_accel_mps2, dt
    )
    profiles["hard_brake"] = np.concatenate(
        [
            approach_launch,
            _segment_hold(config.brake_approach_speed_mps, config.brake_cruise_steps),
            _segment_ramp(
                config.brake_approach_speed_mps,
                config.brake_target_speed_mps,
                config.hard_brake_accel_mps2,
                dt,
            ),
            _segment_hold(config.brake_target_speed_mps, config.brake_after_cruise_steps),
        ]
    )
    profiles["gentle_brake"] = np.concatenate(
        [
            approach_launch,
            _segment_hold(config.brake_approach_speed_mps, config.brake_cruise_steps),
            _segment_ramp(
                config.brake_approach_speed_mps,
                config.brake_target_speed_mps,
                config.gentle_brake_accel_mps2,
                dt,
            ),
            _segment_hold(config.brake_target_speed_mps, config.brake_after_cruise_steps),
        ]
    )
    return profiles


def _step_trace(step_speeds: np.ndarray, timestep_s: float) -> EnergyTrace:
    speeds = np.asarray(step_speeds, dtype=np.float64)
    points = np.concatenate([[speeds[0]], speeds])
    times = np.arange(points.size, dtype=np.float64) * timestep_s
    return EnergyTrace(time_s=times, speed_mps=points, step_distance_m=speeds * timestep_s)


def diagnostic_study(config: ReviewConfig) -> dict[str, Any]:
    """Measure every frozen synthetic profile through all three providers.

    The synthetic profiles start from rest with feasible accelerations, so
    they replay through the unmodified domain provider (stock cold-start
    vehicle) with exact trace parity; they are reported separately from the
    real held-out traces.
    """

    proxy_provider = MetaDriveFuelProxyProvider()
    fastsim_provider = FASTSimEnergyProvider(
        FASTSimEnergyConfig(
            vehicle_resource=config.fastsim.vehicle_resource,
            grade=config.fastsim.grade,
            ambient_temperature_k=config.fastsim.ambient_temperature_k,
            initial_elevation_m=config.fastsim.initial_elevation_m,
        )
    )
    dt = config.diagnostics.timestep_s
    results: dict[str, Any] = {}
    for name, step_speeds in build_diagnostic_profiles(config.diagnostics).items():
        trace = _step_trace(step_speeds, dt)
        proxy = proxy_provider.measure(trace)
        fastsim = fastsim_provider.measure(trace)
        physics = measure_physics_proxy(
            step_speeds,
            np.full(step_speeds.size, dt, dtype=np.float64),
            initial_speed_mps=0.0,
            config=config.physics_proxy,
        )
        distance = float(trace.distance_m)
        proxy_fuel_ml = proxy.fuel_ml
        fastsim_energy_j = fastsim.energy_j
        if proxy_fuel_ml is None or fastsim_energy_j is None:
            raise ValueError("diagnostic providers must report non-optional energy metrics")
        results[name] = {
            "step_count": int(step_speeds.size),
            "duration_s": float(step_speeds.size * dt),
            "time_mean_speed_mps": float(step_speeds.mean()),
            "distance_m": distance,
            "proxy": {
                "total_ml": proxy_fuel_ml,
                "ml_per_km": None if distance == 0.0 else proxy_fuel_ml * 1_000.0 / distance,
            },
            "fastsim": {
                "fuel_j": fastsim_energy_j,
                "wh_per_km": (
                    None if distance == 0.0 else fastsim_energy_j / 3_600.0 * 1_000.0 / distance
                ),
            },
            "physics": {
                "fuel_j": physics.fuel_energy_j,
                "wh_per_km": (
                    None
                    if distance == 0.0
                    else physics.fuel_energy_j / 3_600.0 * 1_000.0 / distance
                ),
                "braking_dissipated_j": physics.braking_dissipated_j,
                "auxiliary_fuel_j": physics.auxiliary_fuel_energy_j,
            },
        }

    def _intensity(provider: str, profile: str) -> float:
        value = results[profile][provider]["ml_per_km" if provider == "proxy" else "wh_per_km"]
        if value is None:
            raise ValueError(f"profile {profile!r} has undefined {provider} intensity")
        return float(value)

    def _total(provider: str, profile: str) -> float:
        return float(results[profile][provider]["total_ml" if provider == "proxy" else "fuel_j"])

    fast_fluctuation = build_diagnostic_profiles(config.diagnostics)["fluctuation_fast"]
    slow_fluctuation = build_diagnostic_profiles(config.diagnostics)["fluctuation_slow"]
    quantiles = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0])
    distribution_gap = float(
        np.max(
            np.abs(
                np.quantile(fast_fluctuation, quantiles) - np.quantile(slow_fluctuation, quantiles)
            )
        )
    )
    invariants = {
        "fluctuation_rate_sensitivity": {
            "profiles": ["fluctuation_fast", "fluctuation_slow"],
            "note": "identical duration, mean speed and (up to sampling) marginal speed"
            " distribution; the fast profile cycles kinetic energy twice as often"
            " per kilometre, which only acceleration-sensitive providers can see",
            "speed_quantile_gap_mps": distribution_gap,
            "time_mean_speed_gap_mps": abs(
                results["fluctuation_fast"]["time_mean_speed_mps"]
                - results["fluctuation_slow"]["time_mean_speed_mps"]
            ),
            "intensity_relative_difference": {
                provider: (
                    _intensity(provider, "fluctuation_fast")
                    - _intensity(provider, "fluctuation_slow")
                )
                / _intensity(provider, "fluctuation_slow")
                for provider in _PROVIDER_KEYS
            },
        },
        "idle_visibility": {
            "profiles": ["stop_and_go", "stop_and_go_no_dwell"],
            "note": "removing the 0 m/s dwell preserves speed continuity; the speed-only"
            " proxy totals must be identical while physics providers keep burning"
            " idle/auxiliary fuel during the dwell",
            "proxy_total_ml_absolute_difference": abs(
                _total("proxy", "stop_and_go") - _total("proxy", "stop_and_go_no_dwell")
            ),
            "fuel_absolute_difference": {
                provider: _total(provider, "stop_and_go") - _total(provider, "stop_and_go_no_dwell")
                for provider in ("fastsim", "physics")
            },
            "dwell_seconds": float(config.diagnostics.stop_and_go_dwell_steps * dt),
        },
        "brake_intensity_response": {
            "profiles": ["hard_brake", "gentle_brake"],
            "note": "same launch, cruise and braking endpoints with different braking"
            " intensity; descriptive response per provider",
            "intensity": {
                provider: {
                    "hard_brake": _intensity(provider, "hard_brake"),
                    "gentle_brake": _intensity(provider, "gentle_brake"),
                }
                for provider in _PROVIDER_KEYS
            },
        },
    }
    return {"profiles": results, "invariants": invariants}


def _provider_ln_intensity(
    facts: dict[str, Any], resolution_key: str, provider: str
) -> float | None:
    row = facts["resolutions"][resolution_key]
    if provider == "proxy":
        value = row["proxy"]["ml_per_km"]
    elif provider == "fastsim":
        value = None if not row["fastsim"]["followed"] else row["fastsim"]["wh_per_km"]
    elif provider == "physics":
        value = row["physics"]["wh_per_km"]
    else:
        raise ValueError(f"unknown provider: {provider!r}")
    if value is None or value <= 0.0:
        return None
    return math.log(value)


def _leave_one_out(values: np.ndarray) -> dict[str, Any] | None:
    if values.size < 2:
        return None
    total = float(values.sum())
    full_mean = total / values.size
    loo_means = (total - values) / (values.size - 1)
    return {
        "sample_count": int(values.size),
        "full_mean": full_mean,
        "loo_mean_min": float(np.min(loo_means)),
        "loo_mean_max": float(np.max(loo_means)),
        "sign_change_count": int(sum(np.sign(mean) != np.sign(full_mean) for mean in loo_means)),
    }


def _paired_provider_contrast(
    episodes: list[dict[str, Any]],
    reference_arm: str,
    comparison_arm: str,
    checkpoint_label: str,
    config: ReviewConfig,
    resolution_key: str,
    *,
    descriptive: bool,
) -> dict[str, Any]:
    def _rows(arm: str) -> list[dict[str, Any]]:
        rows = [
            e for e in episodes if e["arm"] == arm and e["checkpoint_label"] == checkpoint_label
        ]
        if not rows:
            rows = [e for e in episodes if e["arm"] == arm and e["checkpoint_label"] == "baseline"]
        return rows

    reference_rows, comparison_rows = _rows(reference_arm), _rows(comparison_arm)
    if not reference_rows or not comparison_rows:
        raise ValueError(
            f"contrast {reference_arm}->{comparison_arm} has no measured episodes at"
            f" label {checkpoint_label!r}"
        )
    noise_seeds = {e["noise_seed"] for e in reference_rows + comparison_rows}
    if len(noise_seeds) != 1:
        raise ValueError("contrast episodes do not share a single noise seed")
    scenarios = sorted({e["scenario"] for e in comparison_rows})
    for arm, rows in ((reference_arm, reference_rows), (comparison_arm, comparison_rows)):
        if sorted({e["scenario"] for e in rows}) != scenarios:
            raise ValueError(f"arm {arm} scenario set differs from the comparison arm")
    reference_index = {(e["scenario"], e["training_seed"]): e for e in reference_rows}
    comparison_index = {(e["scenario"], e["training_seed"]): e for e in comparison_rows}
    if len(reference_index) != len(reference_rows) or len(comparison_index) != len(comparison_rows):
        raise ValueError("duplicate episode identity within a contrast arm")
    reference_seeds = sorted(
        {e["training_seed"] for e in reference_rows}, key=lambda s: (s is None, s)
    )
    comparison_seeds = sorted(
        {e["training_seed"] for e in comparison_rows}, key=lambda s: (s is None, s)
    )
    if reference_seeds != comparison_seeds and reference_seeds != [None]:
        raise ValueError("contrast arms do not share matched training seeds")

    per_seed: dict[str, Any] = {}
    for comparison_seed in comparison_seeds:
        reference_seed = (
            comparison_seed if (scenarios[0], comparison_seed) in reference_index else None
        )
        pairs = []
        for scenario in scenarios:
            reference = reference_index[(scenario, reference_seed)]
            comparison = comparison_index[(scenario, comparison_seed)]
            if reference["map_sequence"] != comparison["map_sequence"]:
                raise ValueError(f"map family mismatch for scenario {scenario}")
            shared = reference["facts"] is not None and comparison["facts"] is not None
            available: dict[str, bool] = {}
            reasons: dict[str, str | None] = {}
            deltas: dict[str, float | None] = {}
            for provider in _PROVIDER_KEYS:
                reference_ln = (
                    None
                    if reference["facts"] is None
                    else _provider_ln_intensity(reference["facts"], resolution_key, provider)
                )
                comparison_ln = (
                    None
                    if comparison["facts"] is None
                    else _provider_ln_intensity(comparison["facts"], resolution_key, provider)
                )
                if reference_ln is None or comparison_ln is None:
                    available[provider] = False
                    if reference["facts"] is None or comparison["facts"] is None:
                        reasons[provider] = "incomplete_episode"
                    elif provider == "fastsim":
                        reasons[provider] = "fastsim_not_followed"
                    else:
                        reasons[provider] = "undefined_intensity"
                    deltas[f"{provider}_ln_intensity"] = None
                else:
                    available[provider] = True
                    reasons[provider] = None
                    deltas[f"{provider}_ln_intensity"] = comparison_ln - reference_ln
            if shared:
                comparison_facts = comparison["facts"]
                reference_facts = reference["facts"]
                if comparison_facts is None or reference_facts is None:
                    raise ValueError("shared pair requires complete episode facts")
                deltas["distance_m"] = (
                    comparison_facts["distance_m"] - reference_facts["distance_m"]
                )
                deltas["time_mean_speed_mps"] = (
                    comparison_facts["time_mean_speed_mps"] - reference_facts["time_mean_speed_mps"]
                )
            else:
                deltas["distance_m"] = None
                deltas["time_mean_speed_mps"] = None
            pairs.append(
                {
                    "scenario": scenario,
                    "map_sequence": comparison["map_sequence"],
                    "training_seed": comparison_seed,
                    "available": available,
                    "unavailable_reasons": reasons,
                    "reference": reference["facts"],
                    "comparison": comparison["facts"],
                    "deltas": deltas,
                }
            )

        seed_row: dict[str, Any] = {"pair_count": len(pairs), "pairs": pairs}
        providers: dict[str, Any] = {}
        for provider in _PROVIDER_KEYS:
            metric = f"{provider}_ln_intensity"
            valid = [p for p in pairs if p["available"][provider]]
            values = np.asarray([p["deltas"][metric] for p in valid], dtype=np.float64)
            if descriptive:
                effects = {
                    "estimate": float(values.mean()) if values.size else None,
                    "sample_count": int(values.size),
                }
            else:
                effects = scenario_effect(values, config.bootstrap)
            providers[provider] = {
                "available_pair_count": int(values.size),
                "effects": {"ln_intensity": effects},
                "unavailable_pairs": [
                    {"scenario": p["scenario"], "reason": p["unavailable_reasons"][provider]}
                    for p in pairs
                    if not p["available"][provider]
                ],
                "family_split": {
                    family: {
                        "mean": float(
                            np.mean(
                                [p["deltas"][metric] for p in valid if p["map_sequence"] == family]
                            )
                        ),
                        "count": sum(1 for p in valid if p["map_sequence"] == family),
                    }
                    for family in sorted({p["map_sequence"] for p in valid})
                },
            }
            if not descriptive:
                providers[provider]["leave_one_out"] = _leave_one_out(values)
        shared_valid = [p for p in pairs if p["deltas"]["distance_m"] is not None]
        shared_effects = {}
        for metric in _SHARED_METRICS:
            values = np.asarray([p["deltas"][metric] for p in shared_valid], dtype=np.float64)
            if descriptive:
                shared_effects[metric] = {
                    "estimate": float(values.mean()) if values.size else None,
                    "sample_count": int(values.size),
                }
            else:
                shared_effects[metric] = scenario_effect(values, config.bootstrap)
        seed_row["providers"] = providers
        seed_row["shared"] = {
            "available_pair_count": len(shared_valid),
            "effects": shared_effects,
        }
        per_seed[str(comparison_seed)] = seed_row

    across_seeds: dict[str, Any] = {}
    for provider in _PROVIDER_KEYS:
        metric = f"{provider}_ln_intensity"
        estimates = {
            seed: row["providers"][provider]["effects"]["ln_intensity"]["estimate"]
            for seed, row in per_seed.items()
            if row["providers"][provider]["effects"]["ln_intensity"]["estimate"] is not None
        }
        signs = {seed: int(np.sign(value)) for seed, value in estimates.items()}
        nonzero = [sign for sign in signs.values() if sign != 0]
        across_seeds[metric] = {
            "per_seed_estimates": estimates,
            "mean_of_estimates": float(np.mean(list(estimates.values()))) if estimates else None,
            "per_seed_ci95": {
                seed: row["providers"][provider]["effects"]["ln_intensity"].get("ci95")
                for seed, row in per_seed.items()
            },
            "per_seed_signs": signs,
            "all_seeds_same_nonzero_sign": bool(nonzero) and len(set(nonzero)) == 1,
        }
    for metric in _SHARED_METRICS:
        estimates = {
            seed: row["shared"]["effects"][metric]["estimate"]
            for seed, row in per_seed.items()
            if row["shared"]["effects"][metric]["estimate"] is not None
        }
        across_seeds[metric] = {
            "per_seed_estimates": estimates,
            "mean_of_estimates": float(np.mean(list(estimates.values()))) if estimates else None,
        }
    return {
        "reference_arm": reference_arm,
        "comparison_arm": comparison_arm,
        "difference_direction": "comparison - reference",
        "checkpoint_label": checkpoint_label,
        "resolution_key": resolution_key,
        "descriptive_only": descriptive,
        "noise_seed": noise_seeds.pop(),
        "reference_has_training_seed": reference_seeds != [None],
        "per_seed": per_seed,
        "across_seeds": across_seeds,
    }


def _replay_census(episodes: list[dict[str, Any]], config: ReviewConfig) -> dict[str, Any]:
    """Fail-closed feasibility accounting per resolution, arm and seed."""

    census: dict[str, Any] = {}
    for window_s in config.resolutions:
        key = _window_key(window_s)
        rows: dict[str, Any] = {}
        residuals = []
        elapsed = []
        for episode in episodes:
            facts = episode["facts"]
            arm_key = f"{episode['arm']}/seed_{episode['training_seed']}"
            row = rows.setdefault(
                arm_key,
                {"measured": 0, "followed": 0, "unfollowed": 0, "unavailable": 0},
            )
            if facts is None:
                row["unavailable"] += 1
                continue
            row["measured"] += 1
            resolution = facts["resolutions"][key]
            if resolution["window_distance_residual_rel"] is not None:
                residuals.append(resolution["window_distance_residual_rel"])
            if resolution["fastsim"]["followed"]:
                row["followed"] += 1
                elapsed.append(resolution["fastsim"]["elapsed_ms"])
            else:
                row["unfollowed"] += 1
        unfollowed = [
            {
                "arm": e["arm"],
                "training_seed": e["training_seed"],
                "scenario": e["scenario"],
                "failure_message": e["facts"]["resolutions"][key]["fastsim"]["failure_message"],
            }
            for e in episodes
            if e["facts"] is not None and not e["facts"]["resolutions"][key]["fastsim"]["followed"]
        ]
        census[key] = {
            "window_s": window_s,
            "per_arm_seed": rows,
            "measured_episode_count": sum(r["measured"] for r in rows.values()),
            "followed_episode_count": sum(r["followed"] for r in rows.values()),
            "unfollowed_episode_count": sum(r["unfollowed"] for r in rows.values()),
            "unavailable_episode_count": sum(r["unavailable"] for r in rows.values()),
            "unfollowed_episodes": unfollowed,
            "max_window_distance_relative_residual": max(residuals) if residuals else None,
            "mean_fastsim_elapsed_ms": float(np.mean(elapsed)) if elapsed else None,
        }
    return census


def _gate_inputs(
    census: dict[str, Any],
    contrasts: dict[str, Any],
    diagnostics: dict[str, Any],
    config: ReviewConfig,
) -> dict[str, Any]:
    primary_key = _window_key(config.fastsim.primary_window_s)
    secondary_key = _window_key(config.fastsim.secondary_window_s)
    primary_census = census[primary_key]
    label = config.checkpoint_labels[0]
    primary = contrasts[label]["primary"]
    context = contrasts[label]["context"]

    def _direction(contrast: dict[str, Any]) -> dict[str, Any]:
        rows = {}
        for seed, row in contrast["per_seed"].items():
            signs = {
                provider: row["providers"][provider]["effects"]["ln_intensity"]["estimate"]
                for provider in _PROVIDER_KEYS
            }
            values = [value for value in signs.values() if value is not None]
            rows[seed] = {
                "per_provider_mean_delta_ln": signs,
                "all_available_providers_same_sign": bool(values)
                and all(np.sign(value) == np.sign(values[0]) for value in values),
            }
        return rows

    capabilities = [
        {
            "provider": "metadrive_fuel_proxy",
            "units": "mL and mL/km (proxy units, never compared to J/Wh)",
            "inputs": "end-of-substep speed norm and executed displacement only",
            "zero_distance_intensity": "undefined (None)",
            "idle_visibility": "none (dwell steps contribute exactly zero)",
            "braking_regeneration": "none",
            "offline_recomputable": "exact closed-form formula",
        },
        {
            "provider": "fastsim_fuel_energy (fixed-trace window replay)",
            "units": "J and Wh/km",
            "inputs": "distance-preserving window-mean cycle, warm-engine 2012_Ford_Fusion"
            " copy, ramp prefix excluded by cumulative accounting",
            "zero_distance_intensity": "idle energy defined, per-km undefined",
            "idle_visibility": "yes (engine idle/auxiliary model)",
            "braking_regeneration": "conventional vehicle, no regeneration; follow failures"
            " are fail-closed",
            "offline_recomputable": "deterministic forward simulation, fastsim 3.1.0",
            "mean_elapsed_ms_per_episode": primary_census["mean_fastsim_elapsed_ms"],
        },
        {
            "provider": "physics_proxy (minimal longitudinal dynamics)",
            "units": "J and Wh/km",
            "inputs": "same window series; mass/aero/rolling parameters from the pinned"
            " vehicle resource, constant path efficiency, no regeneration",
            "zero_distance_intensity": "auxiliary fuel defined, per-km undefined",
            "idle_visibility": "yes (explicit auxiliary and idle terms)",
            "braking_regeneration": "none (braking fully dissipated)",
            "offline_recomputable": "exact closed-form integration",
        },
    ]
    return {
        "verdict_1_replay_reliability": {
            "primary_window_s": config.fastsim.primary_window_s,
            "measured_episode_count": primary_census["measured_episode_count"],
            "followed_episode_count": primary_census["followed_episode_count"],
            "unfollowed_episode_count": primary_census["unfollowed_episode_count"],
            "unfollowed_episodes": primary_census["unfollowed_episodes"],
            "max_window_distance_relative_residual": primary_census[
                "max_window_distance_relative_residual"
            ],
            "vehicle_conditions": {
                "vehicle_resource": config.fastsim.vehicle_resource,
                "grade": config.fastsim.grade,
                "ambient_temperature_k": config.fastsim.ambient_temperature_k,
                "initial_elevation_m": config.fastsim.initial_elevation_m,
                "warm_engine_modification": config.fastsim.warm_engine,
                "note": "study-local vehicle copy; the domain provider is unmodified",
            },
            "secondary_resolution": {
                "window_s": config.fastsim.secondary_window_s,
                "followed_episode_count": census[secondary_key]["followed_episode_count"],
                "unfollowed_episode_count": census[secondary_key]["unfollowed_episode_count"],
                "role": "descriptive only; never enters Gate P2",
            },
        },
        "verdict_2_direction_agreement": {
            "primary_contrast": {
                "arms": f"{primary['reference_arm']}->{primary['comparison_arm']}",
                "resolution_key": primary_key,
                "per_seed": _direction(primary),
                "per_provider_cross_seed_consistency": {
                    metric: {
                        "all_seeds_same_nonzero_sign": row["all_seeds_same_nonzero_sign"],
                        "per_seed_estimates": row["per_seed_estimates"],
                    }
                    for metric, row in primary["across_seeds"].items()
                    if metric.endswith("_ln_intensity")
                },
            },
            "context_contrast": {
                "arms": f"{context['reference_arm']}->{context['comparison_arm']}",
                "per_seed": _direction(context),
            },
        },
        "verdict_3_provider_capabilities": capabilities,
        "diagnostic_invariants": diagnostics["invariants"],
    }


def run(source: Path, config_path: Path, output: Path, *, figures: bool = True) -> dict[str, Any]:
    """Execute the frozen fixed-trace physics review over one E-060 source."""

    from eco_planner.reporting.artifacts import separate_output

    source, output = separate_output(source, output)
    config = load_review_config(config_path)
    study = _load_source(source)
    measurer = FastSimWindowMeasurer(config.fastsim)

    provenance = {
        "source_manifest": str((source / "comparison.yaml").resolve()),
        "matched_scenarios": [list(key) for key in study.scenario_keys],
        "noise_seed": study.noise_seed,
        "runtime_seed": study.runtime_seed,
        "runs": [
            {
                "arm": run.arm,
                "checkpoint_label": run.checkpoint_label,
                "training_seed": run.training_seed,
                "reward_profile": run.training.reward_profile,
                "policy_hash": run.policy_hash,
                "evaluation_dir": str(run.evaluation_dir),
            }
            for run in study.runs
        ],
        "baseline": {
            "arm": study.baseline_arm,
            "evaluation_dir": str(study.baseline_dir),
        },
        "cycle_reconstruction": {
            "primary_window_s": config.fastsim.primary_window_s,
            "secondary_window_s": config.fastsim.secondary_window_s,
            "ramp_accel_mps2": config.fastsim.ramp_accel_mps2,
            "ramp_timestep_s": config.fastsim.ramp_timestep_s,
            "hold_window_count": config.fastsim.hold_window_count,
            "warm_engine": config.fastsim.warm_engine,
            "note": "raw 0.1 s traces are kinematic waypoint outputs and are physically"
            " unfollowable; every provider consumes the same distance-preserving"
            " window-mean reconstruction of the identical executed trace",
        },
    }

    episodes: list[dict[str, Any]] = []
    for run in study.runs:
        if run.checkpoint_label in config.checkpoint_labels:
            episodes.extend(
                _measure_evaluation_directory(
                    run.arm,
                    run.checkpoint_label,
                    run.training_seed,
                    run.evaluation_dir,
                    config,
                    measurer,
                )
            )
    if config.include_baseline and study.baseline_dir is not None:
        episodes.extend(
            _measure_evaluation_directory(
                str(study.baseline_arm), "baseline", None, study.baseline_dir, config, measurer
            )
        )
    census = _replay_census(episodes, config)
    diagnostics = diagnostic_study(config)

    contrasts: dict[str, Any] = {}
    primary_key = _window_key(config.fastsim.primary_window_s)
    secondary_key = _window_key(config.fastsim.secondary_window_s)
    for label in config.checkpoint_labels:
        contrasts[label] = {
            "primary": _paired_provider_contrast(
                episodes,
                config.contrast_primary[0],
                config.contrast_primary[1],
                label,
                config,
                primary_key,
                descriptive=False,
            ),
            "context": _paired_provider_contrast(
                episodes,
                config.contrast_context[0],
                config.contrast_context[1],
                label,
                config,
                primary_key,
                descriptive=False,
            ),
            "secondary_descriptive": _paired_provider_contrast(
                episodes,
                config.contrast_primary[0],
                config.contrast_primary[1],
                label,
                config,
                secondary_key,
                descriptive=True,
            ),
        }
    result = {
        "status": "completed",
        "study_name": config.study_name,
        "source": str(source),
        "config": config.model_dump(mode="json"),
        "provenance": provenance,
        "census": census,
        "diagnostics": diagnostics,
        "contrasts": contrasts,
        "gate_p2_inputs": _gate_inputs(census, contrasts, diagnostics, config),
    }
    _publish(result, output, figures=figures)
    return {
        "status": result["status"],
        "output_dir": str(output),
        "gate_p2_inputs": result["gate_p2_inputs"],
    }


def _publish(result: dict[str, Any], output: Path, *, figures: bool) -> None:
    from eco_planner.artifacts import write_json
    from eco_planner.reporting.artifacts import write_analysis

    from .fastsim_report import write_report

    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "summary.json", result)
    files: list[str] = []
    if figures:
        from .fastsim_report import plot

        files = plot(result, output)
    write_analysis(
        output,
        result,
        files,
        experiment="energy-proxy-fastsim-fixed-trace-review",
        source=Path(result["source"]),
    )
    write_report(result, output, files)


__all__ = [
    "FastSimCycleConfig",
    "FastSimWindowMeasurer",
    "ReviewConfig",
    "build_diagnostic_profiles",
    "diagnostic_study",
    "distance_preserving_windows",
    "load_review_config",
    "run",
]
