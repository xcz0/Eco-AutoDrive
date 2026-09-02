from __future__ import annotations

import math

import numpy as np
import pytest

from eco_planner.envs.domain.metrics import TransitionMetricInput
from eco_planner.envs.domain.traffic import (
    StaticTrafficObjectState,
    TrafficFrame,
    TrafficParticipantState,
)
from eco_planner.rl.reward import PlannerRFTEnergyRewardConfig, score_plannerrft_energy_step


def _config() -> PlannerRFTEnergyRewardConfig:
    return PlannerRFTEnergyRewardConfig.model_validate(
        {
            "name": "plannerrft_energy_v1",
            "weights": {
                "ttc": 5.0,
                "progress": 5.0,
                "comfort": 2.0,
                "speed": 4.0,
                "energy": 1.0,
            },
            "gates": {
                "collision_vehicle": True,
                "collision_object": True,
                "collision_building": True,
                "collision_human": True,
                "collision_sidewalk": True,
                "wrong_direction_max_heading_error_rad": math.pi / 2,
            },
            "ttc": {
                "critical_ttc_s": 1.0,
                "safe_ttc_s": 4.0,
                "maximum_ttc_s": 10.0,
                "minimum_closing_speed_mps": 0.1,
                "lateral_margin_m": 0.5,
                "longitudinal_margin_m": 0.5,
            },
            "progress": {"full_score_delta_m": 1.0},
            "comfort": {
                "longitudinal_acceleration_limit_mps2": 3.0,
                "lateral_acceleration_limit_mps2": 3.0,
                "jerk_limit_mps3": 5.0,
                "yaw_rate_limit_radps": 0.5,
            },
            "speed": {"overspeed_margin_mps": 0.0, "zero_score_overspeed_mps": 5.0},
            "energy": {"reference_ml_per_km": 50.0, "minimum_step_distance_m": 0.01},
        }
    )


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


@pytest.mark.smoke
def test_plannerrft_energy_reward_matches_the_worked_no_traffic_example() -> None:
    audit = score_plannerrft_energy_step(_config(), _input())

    assert audit.reward_gate == 1.0
    assert audit.ttc_score == 1.0
    assert audit.progress_score == 1.0
    assert audit.comfort_score == 1.0
    assert audit.speed_score == 1.0
    assert audit.executed_fuel_proxy_ml_per_km == pytest.approx(32.5 * math.exp(0.36))
    assert audit.energy_score == pytest.approx(math.exp(-(32.5 * math.exp(0.36)) / 50.0))
    assert audit.reward_total == pytest.approx((5.0 + 5.0 + 2.0 + 4.0 + audit.energy_score) / 17.0)


def test_energy_score_does_not_reward_a_stationary_transition() -> None:
    audit = score_plannerrft_energy_step(
        _config(),
        _input(
            position_xy_m=(0.0, 0.0),
            velocity_xy_mps=(0.0, 0.0),
            route_progress_delta_m=0.0,
            traffic_frame=_frame(),
        ),
    )

    assert not audit.energy_distance_valid
    assert audit.step_distance_m == 0.0
    assert audit.energy_score == 0.0
    assert audit.progress_score == 0.0


def test_ttc_and_terminal_gates_are_independent_auditable_components() -> None:
    lead = TrafficParticipantState(
        object_id="lead",
        kind="vehicle",
        position_xy_m=(8.0, 0.0),
        heading_rad=0.0,
        velocity_xy_mps=(5.0, 0.0),
        width_m=2.0,
        length_m=4.0,
    )
    approaching = score_plannerrft_energy_step(
        _config(), _input(traffic_frame=_frame(participants=(lead,)))
    )
    collision = score_plannerrft_energy_step(
        _config(), _input(traffic_frame=_frame(participants=(lead,)), crash_vehicle=True)
    )

    assert approaching.min_ttc_s == pytest.approx(0.5)
    assert approaching.ttc_score == 0.0
    assert approaching.reward_gate == 1.0
    assert collision.collision_score == 0.0
    assert collision.reward_gate == 0.0
    assert collision.reward_total == 0.0


def test_ttc_ignores_non_closing_lead_traffic_and_scores_static_corridor_objects() -> None:
    following = TrafficParticipantState(
        object_id="lead",
        kind="vehicle",
        position_xy_m=(20.0, 0.0),
        heading_rad=0.0,
        velocity_xy_mps=(10.0, 0.0),
        width_m=2.0,
        length_m=4.0,
    )
    barrier = StaticTrafficObjectState(
        object_id="barrier",
        kind="barrier",
        position_xy_m=(10.0, 0.0),
        heading_rad=math.pi / 2,
        width_m=1.0,
        length_m=4.0,
    )

    non_closing = score_plannerrft_energy_step(
        _config(), _input(traffic_frame=_frame(participants=(following,)))
    )
    static = score_plannerrft_energy_step(
        _config(), _input(traffic_frame=_frame(static_objects=(barrier,)))
    )

    assert not non_closing.has_ttc_candidate
    assert non_closing.min_ttc_s == 10.0
    assert non_closing.ttc_score == 1.0
    assert static.has_ttc_candidate
    assert static.min_ttc_s == pytest.approx(0.6)
    assert static.ttc_score == 0.0


def test_wrong_direction_speed_and_comfort_scores_follow_configured_bounds() -> None:
    audit = score_plannerrft_energy_step(
        _config(),
        _input(
            velocity_xy_mps=(14.0, 0.0),
            previous_velocity_xy_mps=(10.0, 0.0),
            previous_acceleration_xy_mps2=(0.0, 0.0),
            route_heading_rad=math.pi,
        ),
    )

    assert audit.wrong_direction_score == 0.0
    assert audit.reward_gate == 0.0
    assert audit.speed_score == pytest.approx(0.2)
    assert audit.longitudinal_acceleration_mps2 == pytest.approx(40.0)
    assert audit.jerk_mps3 == pytest.approx(400.0)
    assert audit.comfort_score == 0.0


@pytest.mark.parametrize(
    "updates",
    [
        {"out_of_road": True},
        {"crash_object": True},
        {"crash_building": True},
        {"crash_human": True},
        {"crash_sidewalk": True},
    ],
)
def test_every_configured_terminal_gate_zeroes_reward(updates: dict[str, bool]) -> None:
    audit = score_plannerrft_energy_step(_config(), _input(**updates))

    assert audit.reward_gate == 0.0
    assert audit.reward_total == 0.0


@pytest.mark.parametrize(
    "updates",
    [
        {},
        {"position_xy_m": (0.005, 0.0), "route_progress_delta_m": 0.005},
        {"velocity_xy_mps": (15.0, 0.0)},
        {"yaw_rate_radps": 1.0},
        {"route_progress_delta_m": -1.0},
    ],
)
def test_all_plannerrft_component_scores_are_finite_unit_interval(
    updates: dict[str, object],
) -> None:
    audit = score_plannerrft_energy_step(_config(), _input(**updates))

    scores = np.asarray(
        [
            audit.reward_total,
            audit.reward_ungated,
            audit.reward_gate,
            audit.collision_score,
            audit.drivable_score,
            audit.wrong_direction_score,
            audit.ttc_score,
            audit.progress_score,
            audit.comfort_score,
            audit.speed_score,
            audit.energy_score,
        ]
    )
    assert np.isfinite(scores).all()
    assert ((0.0 <= scores) & (scores <= 1.0)).all()
