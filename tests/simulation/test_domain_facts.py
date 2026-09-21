"""Objective-neutral domain facts consumed by reward and evaluation."""

from __future__ import annotations

import math

import pytest

from eco_planner.envs.domain import (
    STOPPED_SPEED_THRESHOLD_MPS,
    WRONG_DIRECTION_MAX_HEADING_ERROR_RAD,
    MetaDriveFuelProxyProvider,
    StaticTrafficObjectState,
    TrafficFrame,
    TrafficParticipantState,
    TransitionMetricInput,
    derive_transition_metrics,
)
from eco_planner.reward import evaluate_plannerrft_energy_step
from tests.training.test_reward import _config


def _frame(
    *,
    participants: tuple[TrafficParticipantState, ...] = (),
    static_objects: tuple[StaticTrafficObjectState, ...] = (),
) -> TrafficFrame:
    return TrafficFrame(
        simulator_step=1,
        ego_center_xy_m=(1.0, 0.0),
        ego_heading_rad=0.0,
        ego_rear_wheelbase_m=1.0,
        participants=participants,
        static_objects=static_objects,
    )


def _input(**updates: object) -> TransitionMetricInput:
    values: dict[str, object] = {
        "previous_position_xy_m": (0.0, 0.0),
        "position_xy_m": (1.0, 0.0),
        "previous_velocity_xy_mps": (10.0, 0.0),
        "velocity_xy_mps": (10.0, 0.0),
        "previous_acceleration_xy_mps2": (0.0, 0.0),
        "heading_rad": 0.0,
        "yaw_rate_radps": 0.0,
        "route_progress_delta_m": 1.0,
        "route_heading_rad": 0.0,
        "speed_limit_mps": 10.0,
        "ego_width_m": 2.0,
        "ego_length_m": 4.0,
        "traffic_frame": _frame(),
        "target_position_xy_m": (1.0, 0.0),
        "target_heading_rad": 0.0,
        "crash_vehicle": False,
        "crash_object": False,
        "crash_building": False,
        "crash_human": False,
        "crash_sidewalk": False,
        "out_of_road": False,
        "native_step_energy_ml": 0.0,
        "native_episode_energy_ml": 0.0,
        "timestep_s": 0.1,
    }
    values.update(updates)
    return TransitionMetricInput(**values)  # type: ignore[arg-type]


def _metrics(**updates: object):
    return derive_transition_metrics(_input(**updates), MetaDriveFuelProxyProvider())


def test_route_heading_error_is_the_wrapped_behavior_geometry() -> None:
    metrics = _metrics(
        heading_rad=math.pi - 0.1,
        target_heading_rad=math.pi - 0.1,
        route_heading_rad=-math.pi + 0.1,
    )

    assert metrics.route_heading_error_rad == pytest.approx(0.2, abs=1e-9)
    assert metrics.heading_error_rad == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize(
    "heading_rad,wrong_direction",
    [
        (WRONG_DIRECTION_MAX_HEADING_ERROR_RAD, False),
        (WRONG_DIRECTION_MAX_HEADING_ERROR_RAD + 1e-6, True),
        (math.pi, True),
    ],
    ids=["half-pi-boundary", "beyond-half-pi", "opposite"],
)
def test_wrong_direction_fact_uses_the_canonical_threshold(
    heading_rad: float, wrong_direction: bool
) -> None:
    metrics = _metrics(heading_rad=heading_rad, route_heading_rad=0.0)

    assert metrics.wrong_direction is wrong_direction


@pytest.mark.parametrize(
    "speed_mps,stopped",
    [
        (STOPPED_SPEED_THRESHOLD_MPS, False),
        (STOPPED_SPEED_THRESHOLD_MPS - 1e-3, True),
        (0.0, True),
    ],
    ids=["threshold-boundary", "below-threshold", "stationary"],
)
def test_stopped_fact_uses_the_canonical_speed_threshold(speed_mps: float, stopped: bool) -> None:
    metrics = _metrics(
        previous_velocity_xy_mps=(speed_mps, 0.0),
        velocity_xy_mps=(speed_mps, 0.0),
    )

    assert metrics.speed_mps == pytest.approx(speed_mps, abs=1e-9)
    assert metrics.stopped is stopped


@pytest.mark.parametrize(
    "flag",
    ["crash_vehicle", "crash_object", "crash_building", "crash_human", "crash_sidewalk"],
)
def test_collision_fact_is_any_raw_crash_flag(flag: str) -> None:
    assert _metrics().collision is False
    assert _metrics(**{flag: True}).collision is True


def test_reward_safety_gate_consumes_the_domain_wrong_direction_fact() -> None:
    aligned = evaluate_plannerrft_energy_step(
        _config(), _metrics(heading_rad=0.1, route_heading_rad=0.0)
    )
    opposite = evaluate_plannerrft_energy_step(
        _config(), _metrics(heading_rad=math.pi, route_heading_rad=0.0)
    )

    assert aligned.diagnostics.wrong_direction_score == 1.0
    assert aligned.safety_gate == 1.0
    assert opposite.diagnostics.wrong_direction_score == 0.0
    assert opposite.safety_gate == 0.0
