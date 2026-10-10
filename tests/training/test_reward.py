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
from eco_planner.reward import (
    FrozenEnergyBand,
    PlannerRFTEnergyRewardConfig,
    PlannerRFTNoEnergyRewardConfig,
    PlannerRFTRewardResult,
    RewardComponents,
    RewardDiagnostics,
    aggregate_transition_reward,
    apply_frozen_energy_band,
    evaluate_plannerrft_energy_step,
    evaluate_plannerrft_no_energy_step,
)
from eco_planner.reward.components import calibrated_band_score


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


def _task_g_profile_payload(name: str) -> dict:
    config_root = Path(__file__).resolve().parents[2] / "configs" / "components" / "reward"
    payload = OmegaConf.to_container(OmegaConf.load(config_root / f"{name}.yaml"), resolve=True)
    assert isinstance(payload, dict)
    return payload


def test_task_g_profiles_freeze_calibration_band_and_lambda_64() -> None:
    r0 = PlannerRFTNoEnergyRewardConfig.model_validate(
        _task_g_profile_payload("plannerrft_no_energy_calibrated_v1")
    )
    rstress = PlannerRFTEnergyRewardConfig.model_validate(
        _task_g_profile_payload("plannerrft_energy_band_lam64_v1")
    )

    assert r0.name == "plannerrft_no_energy_calibrated_v1"
    assert rstress.name == "plannerrft_energy_band_lam64_v1"
    assert r0.weights.total == 16.0
    assert rstress.weights.energy == 64.0
    assert rstress.weights.total == 80.0
    # E-034 frozen calibration on Progress and Comfort.
    assert r0.progress.full_score_delta_m == pytest.approx(1.7813475926717124)
    assert r0.comfort.longitudinal_acceleration_limit_mps2 == pytest.approx(4.157548461641585)
    assert r0.comfort.lateral_acceleration_limit_mps2 == pytest.approx(3.0)
    assert r0.comfort.jerk_limit_mps3 == pytest.approx(111.70486995152065)
    assert r0.comfort.yaw_rate_limit_radps == pytest.approx(0.5)
    # E-038 frozen efficiency-band thresholds on the stress arm only.
    assert r0.energy.mode == "reference_exponential"
    assert r0.energy.band_full_score_ml_per_km is None
    assert r0.energy.band_zero_score_ml_per_km is None
    assert rstress.energy.mode == "calibrated_band"
    assert rstress.energy.band_full_score_ml_per_km == pytest.approx(46.37086372375488)
    assert rstress.energy.band_zero_score_ml_per_km == pytest.approx(48.7514030456543)
    # The paired arms differ only in the energy objective term.
    assert r0.gates == rstress.gates
    assert r0.ttc == rstress.ttc
    assert r0.progress == rstress.progress
    assert r0.comfort == rstress.comfort
    assert r0.speed == rstress.speed
    assert (r0.weights.ttc, r0.weights.progress, r0.weights.comfort, r0.weights.speed) == (
        rstress.weights.ttc,
        rstress.weights.progress,
        rstress.weights.comfort,
        rstress.weights.speed,
    )


def test_frozen_energy_band_switches_a_profile_to_the_e038_thresholds() -> None:
    r0 = PlannerRFTNoEnergyRewardConfig.model_validate(
        _task_g_profile_payload("plannerrft_no_energy_calibrated_v1")
    )
    band = FrozenEnergyBand(
        full_score_ml_per_km=46.37086372375488,
        zero_score_ml_per_km=48.7514030456543,
    )

    applied = apply_frozen_energy_band(r0, band)

    assert applied.energy.mode == "calibrated_band"
    assert applied.energy.band_full_score_ml_per_km == pytest.approx(46.37086372375488)
    assert applied.energy.band_zero_score_ml_per_km == pytest.approx(48.7514030456543)
    # Only the energy representation changes; Progress/Comfort calibration, the
    # shared weights, and the no-energy profile identity are untouched.
    assert applied.name == r0.name
    assert applied.progress == r0.progress
    assert applied.comfort == r0.comfort
    assert applied.weights == r0.weights


def test_frozen_energy_band_rejects_inverted_thresholds() -> None:
    with pytest.raises(ValueError, match="zero_score above full_score"):
        FrozenEnergyBand(full_score_ml_per_km=48.0, zero_score_ml_per_km=47.0)

    with pytest.raises(ValueError):
        FrozenEnergyBand(full_score_ml_per_km=0.0, zero_score_ml_per_km=48.0)


def test_task_g_rstress_reward_uses_band_energy_at_lambda_64() -> None:
    rstress = PlannerRFTEnergyRewardConfig.model_validate(
        _task_g_profile_payload("plannerrft_energy_band_lam64_v1")
    )
    r0 = PlannerRFTNoEnergyRewardConfig.model_validate(
        _task_g_profile_payload("plannerrft_no_energy_calibrated_v1")
    )
    rstress_result = evaluate_plannerrft_energy_step(rstress, _metrics())
    r0_result = evaluate_plannerrft_no_energy_step(r0, _metrics())

    intensity = 32.5 * math.exp(0.36)
    band_full = 46.37086372375488
    band_zero = 48.7514030456543
    assert (
        rstress_result.components.ttc,
        rstress_result.components.progress,
        rstress_result.components.comfort,
        rstress_result.components.speed,
    ) == (
        r0_result.components.ttc,
        r0_result.components.progress,
        r0_result.components.comfort,
        r0_result.components.speed,
    )
    assert rstress_result.diagnostics == r0_result.diagnostics
    assert r0_result.components.energy == pytest.approx(math.exp(-intensity / 50.0))
    assert rstress_result.components.energy == pytest.approx(
        calibrated_band_score(intensity, band_full, band_zero)
    )
    expected_r0 = (
        5.0 * r0_result.components.ttc
        + 5.0 * r0_result.components.progress
        + 2.0 * r0_result.components.comfort
        + 4.0 * r0_result.components.speed
    ) / 16.0
    expected_rstress = (
        5.0 * rstress_result.components.ttc
        + 5.0 * rstress_result.components.progress
        + 2.0 * rstress_result.components.comfort
        + 4.0 * rstress_result.components.speed
        + 64.0 * rstress_result.components.energy
    ) / 80.0
    assert r0_result.total == pytest.approx(expected_r0)
    assert rstress_result.total == pytest.approx(expected_rstress)


def test_reward_results_and_audit_schema_carry_the_configured_profile_name() -> None:
    from eco_planner.rl.rollout import rollout_audit_keys

    r0 = PlannerRFTNoEnergyRewardConfig.model_validate(
        _task_g_profile_payload("plannerrft_no_energy_calibrated_v1")
    )
    rstress = PlannerRFTEnergyRewardConfig.model_validate(
        _task_g_profile_payload("plannerrft_energy_band_lam64_v1")
    )

    assert (
        evaluate_plannerrft_no_energy_step(r0, _metrics()).profile_name
        == "plannerrft_no_energy_calibrated_v1"
    )
    assert (
        evaluate_plannerrft_energy_step(rstress, _metrics()).profile_name
        == "plannerrft_energy_band_lam64_v1"
    )
    assert (
        evaluate_plannerrft_no_energy_step(_no_energy_config(), _metrics()).profile_name
        == "plannerrft_no_energy_v1"
    )
    assert (
        evaluate_plannerrft_energy_step(_config(), _metrics()).profile_name
        == "plannerrft_energy_v1"
    )
    # Every profile shares one rollout audit schema, including the Task G arms.
    assert rollout_audit_keys("plannerrft_energy_band_lam64_v1") == rollout_audit_keys(
        "plannerrft_energy_v1"
    )
    assert rollout_audit_keys("plannerrft_no_energy_calibrated_v1") == rollout_audit_keys(
        "plannerrft_no_energy_v1"
    )


ISSUE83_LAMBDA_PROFILES = (
    "plannerrft_energy_band_lam1_v1",
    "plannerrft_energy_band_lam2_v1",
    "plannerrft_energy_band_lam4_v1",
    "plannerrft_energy_band_lam8_v1",
)


def test_issue83_lambda_arms_freeze_a_matched_calibrated_band_objective() -> None:
    r0 = PlannerRFTNoEnergyRewardConfig.model_validate(
        _task_g_profile_payload("plannerrft_no_energy_calibrated_v1")
    )
    arms = {
        name: PlannerRFTEnergyRewardConfig.model_validate(_task_g_profile_payload(name))
        for name in ISSUE83_LAMBDA_PROFILES
    }

    for name, arm in arms.items():
        # The resolved config states lambda twice and both statements agree:
        # the profile name and weights.energy encode the same value.
        lam = float(name.removeprefix("plannerrft_energy_band_lam").removesuffix("_v1"))
        assert arm.weights.energy == lam
        assert arm.weights.total == 16.0 + lam
        # Every lambda arm uses the E-038 frozen calibrated-band representation.
        assert arm.energy.mode == "calibrated_band"
        assert arm.energy.band_full_score_ml_per_km == pytest.approx(46.37086372375488)
        assert arm.energy.band_zero_score_ml_per_km == pytest.approx(48.7514030456543)

    # Gate I: all lambda arms share Progress/Comfort/TTC/Speed, the safety gate,
    # and the energy-band thresholds exactly; they differ only in the profile
    # identity and weights.energy.
    reference = arms["plannerrft_energy_band_lam1_v1"]
    for name, arm in arms.items():
        if name == "plannerrft_energy_band_lam1_v1":
            continue
        assert arm.gates == reference.gates
        assert arm.ttc == reference.ttc
        assert arm.progress == reference.progress
        assert arm.comfort == reference.comfort
        assert arm.speed == reference.speed
        assert arm.energy == reference.energy
        assert (arm.weights.ttc, arm.weights.progress, arm.weights.comfort, arm.weights.speed) == (
            reference.weights.ttc,
            reference.weights.progress,
            reference.weights.comfort,
            reference.weights.speed,
        )

    # The lambda arms also match the calibrated R0 anchor and the Task G
    # stress arm on every shared component.
    rstress = PlannerRFTEnergyRewardConfig.model_validate(
        _task_g_profile_payload("plannerrft_energy_band_lam64_v1")
    )
    for shared in (r0, rstress):
        assert reference.gates == shared.gates
        assert reference.ttc == shared.ttc
        assert reference.progress == shared.progress
        assert reference.comfort == shared.comfort
        assert reference.speed == shared.speed
        assert reference.energy.band_full_score_ml_per_km == pytest.approx(
            rstress.energy.band_full_score_ml_per_km
        )
    assert (r0.weights.ttc, r0.weights.progress, r0.weights.comfort, r0.weights.speed) == (
        reference.weights.ttc,
        reference.weights.progress,
        reference.weights.comfort,
        reference.weights.speed,
    )

    # Gate I: lambda=0 is the calibrated R0 profile itself, not an energy
    # weight of zero; its objective denominator stays at the four shared weights.
    assert "energy" not in type(r0.weights).model_fields
    assert r0.weights.total == 16.0


def test_issue83_lambda_arms_score_only_through_the_energy_weight() -> None:
    # With identical metrics, every lambda arm shares components with the
    # calibrated R0 anchor and combines them with its own 16+lambda denominator.
    r0 = PlannerRFTNoEnergyRewardConfig.model_validate(
        _task_g_profile_payload("plannerrft_no_energy_calibrated_v1")
    )
    for name in ISSUE83_LAMBDA_PROFILES:
        arm = PlannerRFTEnergyRewardConfig.model_validate(_task_g_profile_payload(name))
        result = evaluate_plannerrft_energy_step(arm, _metrics())
        reference = evaluate_plannerrft_no_energy_step(r0, _metrics())
        assert result.profile_name == name
        assert (
            result.components.ttc,
            result.components.progress,
            result.components.comfort,
            result.components.speed,
        ) == (
            reference.components.ttc,
            reference.components.progress,
            reference.components.comfort,
            reference.components.speed,
        )
        assert result.diagnostics == reference.diagnostics
        expected = (
            5.0 * result.components.ttc
            + 5.0 * result.components.progress
            + 2.0 * result.components.comfort
            + 4.0 * result.components.speed
            + arm.weights.energy * result.components.energy
        ) / arm.weights.total
        assert result.base_total == pytest.approx(expected)
        assert result.total == pytest.approx(result.safety_gate * expected)


def test_issue83_band_profile_rejects_a_name_weight_lambda_mismatch() -> None:
    payload = _task_g_profile_payload("plannerrft_energy_band_lam8_v1")
    payload["weights"]["energy"] = 4.0

    with pytest.raises(ValueError, match="plannerrft_energy_band_lam8_v1 requires"):
        PlannerRFTEnergyRewardConfig.model_validate(payload)


def _components(**updates: object) -> RewardComponents:
    values: dict[str, object] = {
        "ttc": 1.0,
        "progress": 2.0,
        "comfort": 3.0,
        "speed": 4.0,
        "energy": 5.0,
    }
    values.update(updates)
    return RewardComponents(**values)  # type: ignore[arg-type]


def _diagnostics(**updates: object) -> RewardDiagnostics:
    values: dict[str, object] = {
        "collision_score": 1.0,
        "drivable_score": 1.0,
        "wrong_direction_score": 1.0,
        "has_ttc_candidate": True,
        "min_ttc_s": 4.0,
        "route_progress_delta_m": 1.0,
        "speed_mps": 2.0,
        "speed_limit_mps": 3.0,
        "overspeed_mps": 4.0,
        "longitudinal_acceleration_mps2": 5.0,
        "lateral_acceleration_mps2": 6.0,
        "jerk_mps3": 7.0,
        "yaw_rate_radps": 8.0,
        "step_distance_m": 0.5,
        "native_step_energy_ml": 0.1,
        "native_episode_energy_ml": 1.0,
        "executed_fuel_proxy_step_energy_ml": 0.3,
        "executed_fuel_proxy_ml_per_km": 8.0,
        "energy_distance_valid": True,
    }
    values.update(updates)
    return RewardDiagnostics(**values)  # type: ignore[arg-type]


def _result(**updates: object) -> PlannerRFTRewardResult:
    values: dict[str, object] = {
        "profile_name": "plannerrft_energy_v1",
        "total": 1.0,
        "base_total": 1.0,
        "safety_gate": 1.0,
        "components": _components(),
        "diagnostics": _diagnostics(),
    }
    values.update(updates)
    return PlannerRFTRewardResult(**values)  # type: ignore[arg-type]


def test_transition_aggregation_of_one_substep_is_identical_to_that_substep() -> None:
    single = _result(
        total=0.75,
        base_total=0.9,
        safety_gate=0.5,
        components=_components(ttc=0.5),
        diagnostics=_diagnostics(min_ttc_s=2.5, has_ttc_candidate=False),
    )

    assert aggregate_transition_reward([single]) == single


def test_transition_aggregation_uses_explicit_sum_mean_any_all_min_rules() -> None:
    first = _result(
        total=1.0,
        base_total=1.0,
        safety_gate=1.0,
        components=_components(ttc=1.0, progress=2.0, comfort=3.0, speed=4.0, energy=5.0),
        diagnostics=_diagnostics(
            collision_score=1.0,
            drivable_score=1.0,
            wrong_direction_score=1.0,
            has_ttc_candidate=True,
            min_ttc_s=4.0,
            route_progress_delta_m=1.0,
            speed_mps=2.0,
            speed_limit_mps=3.0,
            overspeed_mps=4.0,
            longitudinal_acceleration_mps2=5.0,
            lateral_acceleration_mps2=6.0,
            jerk_mps3=7.0,
            yaw_rate_radps=8.0,
            step_distance_m=0.5,
            native_step_energy_ml=0.1,
            native_episode_energy_ml=1.0,
            executed_fuel_proxy_step_energy_ml=0.3,
            executed_fuel_proxy_ml_per_km=8.0,
            energy_distance_valid=True,
        ),
    )
    second = _result(
        total=2.0,
        base_total=3.0,
        safety_gate=0.5,
        components=_components(ttc=10.0, progress=20.0, comfort=30.0, speed=40.0, energy=50.0),
        diagnostics=_diagnostics(
            collision_score=0.5,
            drivable_score=0.25,
            wrong_direction_score=0.75,
            has_ttc_candidate=False,
            min_ttc_s=2.0,
            route_progress_delta_m=3.0,
            speed_mps=4.0,
            speed_limit_mps=5.0,
            overspeed_mps=6.0,
            longitudinal_acceleration_mps2=7.0,
            lateral_acceleration_mps2=8.0,
            jerk_mps3=9.0,
            yaw_rate_radps=10.0,
            step_distance_m=0.25,
            native_step_energy_ml=0.2,
            native_episode_energy_ml=2.0,
            executed_fuel_proxy_step_energy_ml=0.4,
            executed_fuel_proxy_ml_per_km=10.0,
            energy_distance_valid=False,
        ),
    )

    aggregated = aggregate_transition_reward([first, second])

    assert aggregated.profile_name == "plannerrft_energy_v1"
    assert aggregated.total == pytest.approx(3.0)
    assert aggregated.base_total == pytest.approx(4.0)
    assert aggregated.safety_gate == pytest.approx(0.5)
    assert aggregated.components == RewardComponents(
        ttc=11.0, progress=22.0, comfort=33.0, speed=44.0, energy=55.0
    )
    # sum for additive diagnostics
    assert aggregated.diagnostics.route_progress_delta_m == pytest.approx(4.0)
    assert aggregated.diagnostics.step_distance_m == pytest.approx(0.75)
    assert aggregated.diagnostics.native_step_energy_ml == pytest.approx(0.3)
    # MetaDrive exposes episode energy as a running cumulative value: keep the last substep.
    assert aggregated.diagnostics.native_episode_energy_ml == pytest.approx(2.0)
    assert aggregated.diagnostics.executed_fuel_proxy_step_energy_ml == pytest.approx(0.7)
    # mean for intensive diagnostics
    assert aggregated.diagnostics.min_ttc_s == pytest.approx(3.0)
    assert aggregated.diagnostics.speed_mps == pytest.approx(3.0)
    assert aggregated.diagnostics.speed_limit_mps == pytest.approx(4.0)
    assert aggregated.diagnostics.overspeed_mps == pytest.approx(5.0)
    assert aggregated.diagnostics.longitudinal_acceleration_mps2 == pytest.approx(6.0)
    assert aggregated.diagnostics.lateral_acceleration_mps2 == pytest.approx(7.0)
    assert aggregated.diagnostics.jerk_mps3 == pytest.approx(8.0)
    assert aggregated.diagnostics.yaw_rate_radps == pytest.approx(9.0)
    # `ml/km` is a ratio: distance-weighted, not a plain mean.
    assert aggregated.diagnostics.executed_fuel_proxy_ml_per_km == pytest.approx(
        (8.0 * 0.5 + 10.0 * 0.25) / 0.75
    )
    # any / all for boolean flags
    assert aggregated.diagnostics.has_ttc_candidate is True
    assert aggregated.diagnostics.energy_distance_valid is False
    # min for gate-like scores and the safety gate
    assert aggregated.diagnostics.collision_score == pytest.approx(0.5)
    assert aggregated.diagnostics.drivable_score == pytest.approx(0.25)
    assert aggregated.diagnostics.wrong_direction_score == pytest.approx(0.75)


def test_transition_total_is_not_base_total_times_safety_gate_for_multiple_substeps() -> None:
    results = [
        _result(total=1.0, base_total=1.0, safety_gate=1.0),
        _result(total=0.0, base_total=1.0, safety_gate=0.0),
    ]

    aggregated = aggregate_transition_reward(results)

    assert aggregated.total == pytest.approx(1.0)
    assert aggregated.base_total == pytest.approx(2.0)
    assert aggregated.safety_gate == pytest.approx(0.0)
    assert aggregated.total != pytest.approx(aggregated.base_total * aggregated.safety_gate)


def test_transition_aggregation_rejects_empty_and_mixed_profile_inputs() -> None:
    with pytest.raises(ValueError, match="at least one"):
        aggregate_transition_reward([])

    with pytest.raises(ValueError, match="different reward profiles"):
        aggregate_transition_reward(
            [
                _result(profile_name="plannerrft_energy_v1"),
                _result(profile_name="plannerrft_no_energy_v1"),
            ]
        )


@pytest.mark.parametrize("count", [0, -1, 4])
def test_recomposition_rejects_invalid_prefix(count):
    from eco_planner.reward import recompose_reward_prefix

    with pytest.raises(ValueError, match="valid substep count"):
        recompose_reward_prefix({"energy": 1.0}, [_components()] * 3, [1.0] * 3, count)


def test_recomposition_excludes_unexecuted_padding_from_gate_and_scores():
    from eco_planner.reward import energy_only_prefix

    valid = _components(energy=0.25)
    padded = _components(energy=float("nan"))
    result = energy_only_prefix([valid, padded], [0.5, 0.0], 1)
    assert result.total == 0.125
    assert result.base_total == 0.25
    assert result.safety_gate == 0.5
    assert result.components == valid
