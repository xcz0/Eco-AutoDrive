"""Single trace-to-episode metric definitions for closed-loop evaluation."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from eco_planner.envs import TrajectoryExecutionRecord
from eco_planner.evaluation.models import EnergySummary, EpisodeMetrics, SpeedSummary
from eco_planner.execution_contracts import SIMULATOR_STEP_S

STOPPED_SPEED_THRESHOLD_MPS = 0.1


def compute_episode_metrics(
    trace_arrays: Mapping[str, np.ndarray],
    final_execution: TrajectoryExecutionRecord,
    *,
    total_reward: float,
) -> EpisodeMetrics:
    """Compute one evaluation episode's common metrics from its execution trace."""

    states = trace_arrays["executed_states"]
    if states.ndim != 2 or states.shape[1] != 7 or not states.shape[0]:
        raise ValueError("completed evaluation metrics require non-empty [N, 7] executed states")
    if not np.isfinite(states).all():
        raise ValueError("completed evaluation metrics require finite executed states")
    positions = np.vstack((trace_arrays["initial_state"][None, :2], states[:, :2]))
    distance_m = float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())
    speeds = states[:, 5]
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
        stopped_fraction=float(np.mean(speeds < STOPPED_SPEED_THRESHOLD_MPS)),
        route_completion=final_execution.route_completion,
        energy=energy,
        total_reward=total_reward,
        arrive_dest=final_execution.arrive_dest,
        collision=(
            final_execution.crash_vehicle
            or final_execution.crash_object
            or final_execution.crash_building
            or final_execution.crash_human
            or final_execution.crash_sidewalk
        ),
        out_of_road=final_execution.out_of_road,
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
    total_ml = float(values["executed_fuel_proxy_step_energy_ml"].sum(dtype=np.float64))
    distance_m = float(values["executed_step_distance_m"].sum(dtype=np.float64))
    return EnergySummary(
        metric="metadrive_fuel_proxy",
        total_ml=total_ml,
        distance_m=distance_m,
        ml_per_km=None if distance_m == 0.0 else total_ml * 1_000.0 / distance_m,
    )
