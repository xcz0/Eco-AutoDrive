"""Extract objective-neutral transition facts from MetaDrive state."""

from __future__ import annotations

from typing import Any

import numpy as np

from eco_planner.contracts import SIMULATOR_STEP_S
from eco_planner.envs.domain.metrics import (
    EnergyMetricProvider,
    TransitionMetricInput,
    TransitionMetrics,
    derive_transition_metrics,
)
from eco_planner.envs.domain.traffic import TrafficFrame
from eco_planner.envs.metadrive.lane_speed import model_lane_speed_limit_mps
from eco_planner.envs.metadrive.simulator import MetaDriveBackend, MetaDriveStepResult
from eco_planner.envs.metadrive.snapshot import capture_traffic_frame


class TransitionExtractor:
    """Own the previous motion state required to derive consecutive transition metrics."""

    def __init__(self, energy_provider: EnergyMetricProvider) -> None:
        self._energy_provider = energy_provider
        self._previous_position: np.ndarray | None = None
        self._previous_velocity: np.ndarray | None = None
        self._previous_acceleration: np.ndarray | None = None

    def reset(self, backend: MetaDriveBackend) -> TrafficFrame:
        """Initialize metric history and return the reset-time traffic frame."""

        self._previous_position = np.asarray(backend.agent.position, dtype=np.float64).copy()
        self._previous_velocity = np.asarray(backend.agent.velocity, dtype=np.float64).copy()
        self._previous_acceleration = np.zeros(2, dtype=np.float64)
        return capture_traffic_frame(backend)

    def extract(
        self,
        backend: MetaDriveBackend,
        step: MetaDriveStepResult,
        yaw_rate_radps: float,
        target_position_xy_m: tuple[float, float],
        target_heading_rad: float,
    ) -> TransitionMetrics:
        """Capture traffic and derive metrics for one completed simulator transition."""

        if (
            self._previous_position is None
            or self._previous_velocity is None
            or self._previous_acceleration is None
        ):
            raise RuntimeError("transition extractor is unavailable before reset")
        vehicle = backend.agent
        reference_lanes = vehicle.navigation.current_ref_lanes
        if not reference_lanes:
            raise RuntimeError("navigation did not expose current reference lanes")
        lane = vehicle.lane if vehicle.lane in reference_lanes else reference_lanes[0]
        previous_longitudinal, _ = lane.local_coordinates(self._previous_position)
        current_longitudinal, _ = lane.local_coordinates(vehicle.position)
        lane_length = float(lane.length)
        route_heading = float(
            lane.heading_theta_at(float(np.clip(current_longitudinal, 0.0, lane_length)))
        )
        speed_limit_mps, has_speed_limit = model_lane_speed_limit_mps(lane)
        if not has_speed_limit:
            raise RuntimeError("current route lane does not expose a configured speed limit")
        metric_input = TransitionMetricInput(
            previous_position_xy_m=_finite_pair(
                self._previous_position, "previous metric position"
            ),
            position_xy_m=_finite_pair(vehicle.position, "vehicle position"),
            previous_velocity_xy_mps=_finite_pair(
                self._previous_velocity, "previous metric velocity"
            ),
            velocity_xy_mps=_finite_pair(vehicle.velocity, "vehicle velocity"),
            previous_acceleration_xy_mps2=_finite_pair(
                self._previous_acceleration, "previous metric acceleration"
            ),
            heading_rad=float(vehicle.heading_theta),
            yaw_rate_radps=yaw_rate_radps,
            route_progress_delta_m=float(current_longitudinal - previous_longitudinal),
            route_heading_rad=route_heading,
            speed_limit_mps=speed_limit_mps,
            ego_width_m=float(vehicle.WIDTH),
            ego_length_m=float(vehicle.LENGTH),
            traffic_frame=capture_traffic_frame(backend),
            target_position_xy_m=target_position_xy_m,
            target_heading_rad=target_heading_rad,
            crash_vehicle=step.crash_vehicle,
            crash_object=step.crash_object,
            crash_building=step.crash_building,
            crash_human=step.crash_human,
            crash_sidewalk=step.crash_sidewalk,
            out_of_road=step.out_of_road,
            native_step_energy_ml=step.native_step_energy_ml,
            native_episode_energy_ml=step.native_episode_energy_ml,
            timestep_s=SIMULATOR_STEP_S,
        )
        metrics = derive_transition_metrics(metric_input, self._energy_provider)
        self._advance(metric_input)
        return metrics

    def _advance(self, metric_input: TransitionMetricInput) -> None:
        previous_velocity = np.asarray(metric_input.previous_velocity_xy_mps, dtype=np.float64)
        velocity = np.asarray(metric_input.velocity_xy_mps, dtype=np.float64)
        self._previous_position = np.asarray(metric_input.position_xy_m, dtype=np.float64)
        self._previous_velocity = velocity
        self._previous_acceleration = (velocity - previous_velocity) / metric_input.timestep_s


def _finite_pair(value: Any, name: str) -> tuple[float, float]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (2,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite 2D vector")
    return float(array[0]), float(array[1])
