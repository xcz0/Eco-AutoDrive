"""Typed episode metric aggregation from executed facts, independent of persistence."""

from collections.abc import Mapping

import numpy as np

from eco_planner.contracts import SIMULATOR_STEP_S
from eco_planner.envs import TrajectoryExecutionRecord
from eco_planner.envs.domain import EnergyMetrics

from .artifacts.models import EnergySummary, EpisodeMetrics, SpeedSummary


def compute_episode_metrics(
    trace_arrays: Mapping[str, np.ndarray],
    final_execution: TrajectoryExecutionRecord,
) -> EpisodeMetrics:
    """Compute one evaluation episode's common metrics from its execution trace."""

    states = trace_arrays["executed_states"]
    if states.ndim != 2 or states.shape[1] != 7 or not states.shape[0]:
        raise ValueError("completed evaluation metrics require non-empty [N, 7] executed states")
    if not np.isfinite(states).all():
        raise ValueError("completed evaluation metrics require finite executed states")
    positions = np.vstack((trace_arrays["initial_state"][None, :2], states[:, :2]))
    distance_m = float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())
    speeds = trace_arrays["executed_speed_mps"]
    stopped_steps = trace_arrays["executed_stopped"]
    wrong_direction_steps = trace_arrays["executed_wrong_direction"]
    for name, values in (
        ("executed_speed_mps", speeds),
        ("executed_stopped", stopped_steps),
        ("executed_wrong_direction", wrong_direction_steps),
    ):
        if values.shape != (states.shape[0],):
            raise ValueError(f"completed evaluation metrics require state-aligned {name}")
    if not np.isfinite(speeds).all() or np.any(speeds < 0.0):
        raise ValueError("completed evaluation metrics require finite non-negative speeds")
    energy = compute_trace_energy(trace_arrays)
    if energy is None:
        raise ValueError("completed evaluation metrics require execution energy arrays")
    return EpisodeMetrics(
        simulated_seconds=float(states.shape[0] * SIMULATOR_STEP_S),
        distance_m=distance_m,
        speed_mps=SpeedSummary(
            minimum=float(speeds.min()),
            mean=float(speeds.mean()),
            maximum=float(speeds.max()),
        ),
        stopped_fraction=float(np.mean(stopped_steps)),
        route_completion=final_execution.route_completion,
        energy=energy,
        arrive_dest=final_execution.arrive_dest,
        collision=final_execution.collision,
        out_of_road=final_execution.out_of_road,
        wrong_direction=bool(wrong_direction_steps.any()),
        wrong_direction_fraction=float(wrong_direction_steps.mean()),
    )


def compute_trace_energy(trace_arrays: Mapping[str, np.ndarray]) -> EnergySummary | None:
    """Aggregate only the execution-recomputed fuel-proxy trace flow."""

    states = trace_arrays["executed_states"]
    if not states.size:
        return None
    expected_shape = (states.shape[0],)
    fields = (
        "executed_native_step_energy_ml",
        "executed_native_episode_energy_ml",
        "executed_fuel_proxy_step_energy_ml",
        "executed_step_distance_m",
    )
    values = {name: trace_arrays[name] for name in fields}
    for name, value in values.items():
        if value.shape != expected_shape or not np.isfinite(value).all() or np.any(value < 0.0):
            raise ValueError(f"trace {name} must be finite, non-negative, and state-aligned")
    metrics = EnergyMetrics(
        metric="metadrive_fuel_proxy",
        distance_m=float(values["executed_step_distance_m"].sum(dtype=np.float64)),
        energy_j=None,
        fuel_ml=float(values["executed_fuel_proxy_step_energy_ml"].sum(dtype=np.float64)),
    )
    if metrics.fuel_ml is None:
        raise RuntimeError("fuel-proxy aggregation did not produce a fuel-volume metric")
    return EnergySummary(
        metric="metadrive_fuel_proxy",
        total_ml=metrics.fuel_ml,
        distance_m=metrics.distance_m,
        ml_per_km=metrics.fuel_ml_per_km,
    )
