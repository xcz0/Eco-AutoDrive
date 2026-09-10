from __future__ import annotations

import math
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from eco_planner.envs.domain import (
    MetaDriveFuelProxyProvider,
    StaticTrafficObjectState,
    TrafficFrame,
    TrafficParticipantState,
    TransitionMetricInput,
    derive_transition_metrics,
)
from eco_planner.rl.reward import (
    PlannerRFTEnergyRewardConfig,
    PlannerRFTNoEnergyRewardConfig,
    evaluate_plannerrft_energy_step,
    evaluate_plannerrft_no_energy_step,
)
from eco_planner.rl.reward.components import calibrated_band_score


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


def _no_energy_config() -> PlannerRFTNoEnergyRewardConfig:
    payload = _config().model_dump(mode="python")
    payload["name"] = "plannerrft_no_energy_v1"
    payload["weights"] = {
        key: value for key, value in payload["weights"].items() if key != "energy"
    }
    return PlannerRFTNoEnergyRewardConfig.model_validate(payload)


def _energy_config_with_weight(energy_weight: float) -> PlannerRFTEnergyRewardConfig:
    payload = _config().model_dump(mode="python")
    payload["weights"]["energy"] = energy_weight
    return PlannerRFTEnergyRewardConfig.model_validate(payload)


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


@pytest.mark.smoke
def test_plannerrft_energy_reward_matches_the_worked_no_traffic_example() -> None:
    result = evaluate_plannerrft_energy_step(_config(), _metrics())

    assert result.safety_gate == 1.0
    assert result.components.ttc == 1.0
    assert result.components.progress == 1.0
    assert result.components.comfort == 1.0
    assert result.components.speed == 1.0
    assert result.diagnostics.executed_fuel_proxy_ml_per_km == pytest.approx(32.5 * math.exp(0.36))
    assert result.components.energy == pytest.approx(math.exp(-(32.5 * math.exp(0.36)) / 50.0))
    assert result.total == pytest.approx((5.0 + 5.0 + 2.0 + 4.0 + result.components.energy) / 17.0)


def test_energy_score_does_not_reward_a_stationary_transition() -> None:
    result = evaluate_plannerrft_energy_step(
        _config(),
        _metrics(
            position_xy_m=(0.0, 0.0),
            velocity_xy_mps=(0.0, 0.0),
            route_progress_delta_m=0.0,
            traffic_frame=_frame(),
        ),
    )

    assert not result.diagnostics.energy_distance_valid
    assert result.diagnostics.step_distance_m == 0.0
    assert result.components.energy == 0.0
    assert result.components.progress == 0.0


def _band_config() -> PlannerRFTEnergyRewardConfig:
    payload = _config().model_dump(mode="python")
    payload["energy"] = {
        "mode": "calibrated_band",
        "reference_ml_per_km": 50.0,
        "minimum_step_distance_m": 0.01,
        "band_full_score_ml_per_km": 44.0,
        "band_zero_score_ml_per_km": 50.0,
    }
    return PlannerRFTEnergyRewardConfig.model_validate(payload)


def test_calibrated_band_score_direction_and_saturation() -> None:
    assert calibrated_band_score(44.0, 44.0, 50.0) == 1.0
    assert calibrated_band_score(43.0, 44.0, 50.0) == 1.0
    assert calibrated_band_score(50.0, 44.0, 50.0) == 0.0
    assert calibrated_band_score(51.0, 44.0, 50.0) == 0.0
    assert calibrated_band_score(47.0, 44.0, 50.0) == pytest.approx(0.5)
    assert calibrated_band_score(45.0, 44.0, 50.0) > calibrated_band_score(49.0, 44.0, 50.0)


def test_plannerrft_band_energy_reward_scores_intensity_in_the_band() -> None:
    result = evaluate_plannerrft_energy_step(_band_config(), _metrics())

    intensity = 32.5 * math.exp(0.36)
    assert result.diagnostics.energy_distance_valid
    assert result.diagnostics.executed_fuel_proxy_ml_per_km == pytest.approx(intensity)
    assert result.components.energy == pytest.approx(calibrated_band_score(intensity, 44.0, 50.0))
    assert 0.0 < result.components.energy < 1.0


def test_band_energy_score_does_not_reward_a_stationary_transition() -> None:
    result = evaluate_plannerrft_energy_step(
        _band_config(),
        _metrics(
            position_xy_m=(0.0, 0.0),
            velocity_xy_mps=(0.0, 0.0),
            route_progress_delta_m=0.0,
            traffic_frame=_frame(),
        ),
    )

    assert not result.diagnostics.energy_distance_valid
    assert result.components.energy == 0.0


def test_energy_band_config_validation() -> None:
    payload = _band_config().model_dump(mode="python")
    del payload["energy"]["band_zero_score_ml_per_km"]
    with pytest.raises(ValueError, match="calibrated_band energy mode requires"):
        PlannerRFTEnergyRewardConfig.model_validate(payload)

    inverted = _band_config().model_dump(mode="python")
    inverted["energy"]["band_full_score_ml_per_km"] = 50.0
    inverted["energy"]["band_zero_score_ml_per_km"] = 44.0
    with pytest.raises(ValueError, match="zero_score above full_score"):
        PlannerRFTEnergyRewardConfig.model_validate(inverted)

    leaked = _config().model_dump(mode="python")
    leaked["energy"]["band_full_score_ml_per_km"] = 44.0
    with pytest.raises(ValueError, match="only allowed in calibrated_band mode"):
        PlannerRFTEnergyRewardConfig.model_validate(leaked)


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
    approaching = evaluate_plannerrft_energy_step(
        _config(), _metrics(traffic_frame=_frame(participants=(lead,)))
    )
    collision = evaluate_plannerrft_energy_step(
        _config(), _metrics(traffic_frame=_frame(participants=(lead,)), crash_vehicle=True)
    )

    assert approaching.diagnostics.min_ttc_s == pytest.approx(0.5)
    assert approaching.components.ttc == 0.0
    assert approaching.safety_gate == 1.0
    assert collision.diagnostics.collision_score == 0.0
    assert collision.safety_gate == 0.0
    assert collision.total == 0.0


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

    non_closing = evaluate_plannerrft_energy_step(
        _config(), _metrics(traffic_frame=_frame(participants=(following,)))
    )
    static = evaluate_plannerrft_energy_step(
        _config(), _metrics(traffic_frame=_frame(static_objects=(barrier,)))
    )

    assert not non_closing.diagnostics.has_ttc_candidate
    assert non_closing.diagnostics.min_ttc_s == 10.0
    assert non_closing.components.ttc == 1.0
    assert static.diagnostics.has_ttc_candidate
    assert static.diagnostics.min_ttc_s == pytest.approx(0.6)
    assert static.components.ttc == 0.0


def test_wrong_direction_speed_and_comfort_scores_follow_configured_bounds() -> None:
    result = evaluate_plannerrft_energy_step(
        _config(),
        _metrics(
            velocity_xy_mps=(14.0, 0.0),
            previous_velocity_xy_mps=(10.0, 0.0),
            previous_acceleration_xy_mps2=(0.0, 0.0),
            route_heading_rad=math.pi,
        ),
    )

    assert result.diagnostics.wrong_direction_score == 0.0
    assert result.safety_gate == 0.0
    assert result.components.speed == pytest.approx(0.2)
    assert result.diagnostics.longitudinal_acceleration_mps2 == pytest.approx(40.0)
    assert result.diagnostics.jerk_mps3 == pytest.approx(400.0)
    assert result.components.comfort == 0.0


@pytest.mark.smoke
def test_plannerrft_no_energy_reward_matches_the_worked_no_traffic_example() -> None:
    result = evaluate_plannerrft_no_energy_step(_no_energy_config(), _metrics())

    assert result.profile_name == "plannerrft_no_energy_v1"
    assert result.safety_gate == 1.0
    assert result.components.ttc == 1.0
    assert result.components.progress == 1.0
    assert result.components.comfort == 1.0
    assert result.components.speed == 1.0
    # Energy stays an audited, unweighted diagnostic with identical normalization.
    assert result.components.energy == pytest.approx(math.exp(-(32.5 * math.exp(0.36)) / 50.0))
    assert result.diagnostics.executed_fuel_proxy_ml_per_km == pytest.approx(32.5 * math.exp(0.36))
    assert result.base_total == 1.0
    assert result.total == 1.0


def test_no_energy_reward_keeps_gate_semantics_and_stationary_progress_at_zero() -> None:
    stationary = evaluate_plannerrft_no_energy_step(
        _no_energy_config(),
        _metrics(
            position_xy_m=(0.0, 0.0),
            previous_velocity_xy_mps=(0.0, 0.0),
            velocity_xy_mps=(0.0, 0.0),
            route_progress_delta_m=0.0,
            traffic_frame=_frame(),
        ),
    )
    tiny = evaluate_plannerrft_no_energy_step(
        _no_energy_config(),
        _metrics(
            position_xy_m=(0.005, 0.0),
            previous_velocity_xy_mps=(0.05, 0.0),
            velocity_xy_mps=(0.05, 0.0),
            route_progress_delta_m=0.005,
        ),
    )

    assert stationary.components.progress == 0.0
    assert not stationary.diagnostics.energy_distance_valid
    assert stationary.total == pytest.approx((5.0 + 2.0 + 4.0) / 16.0)
    assert tiny.components.progress == pytest.approx(0.005)
    assert not tiny.diagnostics.energy_distance_valid


def test_no_energy_reward_total_is_zeroed_by_terminal_gates() -> None:
    collision = evaluate_plannerrft_no_energy_step(
        _no_energy_config(), _metrics(crash_vehicle=True)
    )
    off_road = evaluate_plannerrft_no_energy_step(_no_energy_config(), _metrics(out_of_road=True))
    wrong_direction = evaluate_plannerrft_no_energy_step(
        _no_energy_config(), _metrics(route_heading_rad=math.pi)
    )

    assert collision.diagnostics.collision_score == 0.0
    assert collision.safety_gate == 0.0
    assert collision.total == 0.0
    assert off_road.diagnostics.drivable_score == 0.0
    assert off_road.safety_gate == 0.0
    assert off_road.total == 0.0
    assert wrong_direction.diagnostics.wrong_direction_score == 0.0
    assert wrong_direction.safety_gate == 0.0
    assert wrong_direction.total == 0.0


@pytest.mark.parametrize("energy_weight", [0.5, 1.0, 2.0, 4.0])
def test_energy_weight_lambda_changes_only_the_energy_term_and_denominator(
    energy_weight: float,
) -> None:
    result = evaluate_plannerrft_energy_step(_energy_config_with_weight(energy_weight), _metrics())
    reference = evaluate_plannerrft_no_energy_step(_no_energy_config(), _metrics())

    assert result.profile_name == "plannerrft_energy_v1"
    assert result.components == reference.components
    assert result.diagnostics == reference.diagnostics
    expected = (
        5.0 * result.components.ttc
        + 5.0 * result.components.progress
        + 2.0 * result.components.comfort
        + 4.0 * result.components.speed
        + energy_weight * result.components.energy
    ) / (16.0 + energy_weight)
    assert result.base_total == pytest.approx(expected)
    assert result.total == pytest.approx(result.safety_gate * expected)


def test_energy_reward_converges_to_the_no_energy_profile_as_lambda_approaches_zero() -> None:
    result = evaluate_plannerrft_energy_step(_energy_config_with_weight(1.0e-9), _metrics())
    reference = evaluate_plannerrft_no_energy_step(_no_energy_config(), _metrics())

    assert result.total == pytest.approx(reference.total, abs=1.0e-9)


def test_no_energy_and_energy_profiles_differ_only_in_the_energy_objective_term() -> None:
    config_root = Path(__file__).resolve().parents[2] / "configs" / "components" / "reward"
    profiles = {
        name: OmegaConf.to_container(OmegaConf.load(config_root / f"{name}.yaml"), resolve=True)
        for name in ("plannerrft_energy_v1", "plannerrft_no_energy_v1")
    }
    energy = PlannerRFTEnergyRewardConfig.model_validate(profiles["plannerrft_energy_v1"])
    no_energy = PlannerRFTNoEnergyRewardConfig.model_validate(profiles["plannerrft_no_energy_v1"])

    assert energy.gates == no_energy.gates
    assert energy.ttc == no_energy.ttc
    assert energy.progress == no_energy.progress
    assert energy.comfort == no_energy.comfort
    assert energy.speed == no_energy.speed
    assert energy.energy == no_energy.energy
    assert (
        energy.weights.ttc,
        energy.weights.progress,
        energy.weights.comfort,
        energy.weights.speed,
    ) == (
        no_energy.weights.ttc,
        no_energy.weights.progress,
        no_energy.weights.comfort,
        no_energy.weights.speed,
    )
    assert no_energy.weights.total == 16.0
    assert energy.weights.total == pytest.approx(16.0 + energy.weights.energy)
