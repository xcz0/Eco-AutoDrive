"""Standalone tracker geometry, feedback memory, and strict input contracts."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf
from pydantic import ValidationError

from eco_planner.contracts import METADRIVE_PHYSICS_STEP_S, PLANNER_HORIZON, SIMULATOR_STEP_S
from eco_planner.envs.domain.controller import (
    TrackingVehicleParameters,
    TrackingVehicleState,
    TrajectoryTrackingConfig,
    TrajectoryTrackingController,
)
from eco_planner.envs.domain.geometry import local_points_to_world
from eco_planner.envs.domain.trajectory import WorldTrajectory

_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "configs/components/controller/pure_pursuit_pid.yaml"
)
_VEHICLE = TrackingVehicleParameters(
    front_wheelbase_m=1.5,
    rear_wheelbase_m=1.0,
    max_steering_deg=35.0,
    enable_reverse=False,
)


def _config(*, lookahead: dict | None = None, **longitudinal: float) -> TrajectoryTrackingConfig:
    config = TrajectoryTrackingConfig.model_validate(
        OmegaConf.to_container(OmegaConf.load(_CONFIG_PATH), resolve=True)
    ).model_dump()
    config["lookahead"].update(lookahead or {})
    config["longitudinal"].update(longitudinal)
    return TrajectoryTrackingConfig.model_validate(config)


def _reference(speed: float = 5.0) -> WorldTrajectory:
    times = np.arange(PLANNER_HORIZON + 1) * SIMULATOR_STEP_S
    centers = np.column_stack((times * speed + _VEHICLE.rear_wheelbase_m, np.zeros_like(times)))
    velocities = np.column_stack((np.full(PLANNER_HORIZON, speed), np.zeros(PLANNER_HORIZON)))
    return WorldTrajectory(centers, np.zeros_like(times), velocities, np.zeros(PLANNER_HORIZON))


def _state(
    *, rear_x: float = 0.0, rear_y: float = 0.0, heading: float = 0.0, speed: float = 0.0
) -> TrackingVehicleState:
    direction = np.array([np.cos(heading), np.sin(heading)])
    return TrackingVehicleState(
        np.array([rear_x, rear_y]) + _VEHICLE.rear_wheelbase_m * direction,
        heading,
        speed * direction,
    )


def _controller(
    reference: WorldTrajectory | None = None, config: TrajectoryTrackingConfig | None = None
) -> TrajectoryTrackingController:
    controller = TrajectoryTrackingController(config or _config(), _VEHICLE)
    controller.set_reference(reference if reference is not None else _reference())
    return controller


def _compute(
    controller: TrajectoryTrackingController, state: TrackingVehicleState, time: float = 0.0
):
    return controller.compute(state, time, METADRIVE_PHYSICS_STEP_S)


@pytest.mark.smoke
def test_straight_reference_has_zero_steering_and_uses_actual_speed_feedback() -> None:
    controller = _controller(config=_config(kp=0.1, ki=0.0))
    accelerating = _compute(controller, _state(speed=2.0))
    braking = _compute(controller, _state(speed=8.0))
    assert accelerating.steering == pytest.approx(0.0, abs=1e-12)
    assert accelerating.throttle_brake == pytest.approx(0.3)
    assert braking.throttle_brake == pytest.approx(-0.3)


@pytest.mark.parametrize(("offset", "sign"), [(1.0, -1), (-1.0, 1)])
def test_lateral_offset_steers_toward_reference(offset: float, sign: int) -> None:
    assert sign * _compute(_controller(), _state(rear_y=offset)).steering > 0.0


def test_pursuit_interpolates_circle_intersection_and_normalizes_degrees() -> None:
    reference = _reference(0.0)
    # Rear-axle path goes diagonally from (0,0) to (4,4); Ld=2 crosses inside it.
    reference.centers[1:] = [5.0, 4.0]
    config = _config(lookahead={"base_m": 2.0, "min_m": 2.0, "max_m": 2.0})
    action = _compute(_controller(reference, config), _state())
    expected = np.arctan(2.5 * (2.0 * np.sqrt(2.0) / 4.0)) / np.deg2rad(35.0)
    assert action.steering == pytest.approx(min(expected, 1.0))
    # A wider steering range avoids saturation and distinguishes units and centre tracking.
    vehicle = _VEHICLE.model_copy(update={"front_wheelbase_m": 0.1, "max_steering_deg": 60.0})
    controller = TrajectoryTrackingController(config, vehicle)
    controller.set_reference(reference)
    assert _compute(controller, _state()).steering == pytest.approx(
        np.arctan(1.1 * np.sqrt(2.0) / 2.0) / np.deg2rad(60.0)
    )


@pytest.mark.parametrize(("speed", "distance"), [(-2.0, 2.0), (0.0, 2.0), (4.0, 4.0), (20.0, 6.0)])
def test_speed_dependent_lookahead_has_explicit_bounds(speed: float, distance: float) -> None:
    config = _config(lookahead={"base_m": 2.0, "speed_gain_s": 0.5, "min_m": 2.0, "max_m": 6.0})
    action = _compute(_controller(config=config), _state(rear_y=1.0, speed=speed))
    expected = np.arctan(2.5 * -2.0 / distance**2) / np.deg2rad(35.0)
    assert action.steering == pytest.approx(np.clip(expected, -1.0, 1.0))


def test_current_reference_center_and_shortest_heading_are_interpolated() -> None:
    reference = _reference(0.0)
    reference.centers[:] = [0.0, 0.0]
    reference.centers[1:] = [2.0, 0.0]
    reference.headings[:] = -np.pi + 0.1
    reference.headings[0] = np.pi - 0.1
    config = _config(lookahead={"base_m": 3.0, "min_m": 3.0, "max_m": 3.0})
    # At t=.05, interpolated centre=(1,0), heading=pi, reference rear=(2,0).
    state = _state(rear_x=5.0, heading=np.pi / 2.0)
    vehicle = _VEHICLE.model_copy(update={"max_steering_deg": 90.0})
    controller = TrajectoryTrackingController(config, vehicle)
    controller.set_reference(reference)
    steering = _compute(controller, state, 0.05).steering
    expected = np.arctan(2.5 * 2.0 / 3.0) / np.deg2rad(90.0)
    assert steering == pytest.approx(expected)


@pytest.mark.parametrize(
    ("time", "expected"),
    [(0.0, 1.0), (0.02, 1.0), (0.1, 1.0), (0.15, 1.5), (8.0, 80.0), (100.0, 80.0)],
)
def test_speed_uses_interval_endpoints_and_holds_terminal_value(
    time: float, expected: float
) -> None:
    reference = _reference()
    reference.velocities[:, 0] = -np.arange(1, PLANNER_HORIZON + 1, dtype=np.float64)
    controller = _controller(reference, _config(kp=0.001, ki=0.0))
    assert _compute(controller, _state(), time).throttle_brake == pytest.approx(expected * 0.001)


@pytest.mark.parametrize("time", [0.0, 0.02, 7.99, 8.0, 8.02, 1e308])
def test_repeated_stationary_points_and_terminal_queries_are_bounded(time: float) -> None:
    action = _compute(_controller(_reference(0.0)), _state(), time)
    assert action.steering == pytest.approx(0.0)
    assert action.throttle_brake == pytest.approx(0.0)


def test_remaining_path_shorter_than_lookahead_uses_terminal_rear_axle() -> None:
    reference = _reference(0.0)
    reference.centers[1:] = [2.0, 0.5]
    config = _config(lookahead={"base_m": 3.0, "min_m": 3.0, "max_m": 3.0})
    controller = _controller(reference, config)
    expected = np.arctan(2.5 / 9.0) / np.deg2rad(35.0)
    for time in (0.0, 8.0, 9.0):
        assert _compute(controller, _state(), time).steering == pytest.approx(expected)


def test_start_already_outside_circle_is_used_without_searching_ahead() -> None:
    action = _compute(_controller(), _state(rear_y=5.0))
    assert action.steering == -1.0


@pytest.mark.parametrize("speed", [0.0, 2.0, -2.0])
def test_zero_target_never_throttles_even_with_positive_integral(speed: float) -> None:
    controller = _controller(_reference(0.5), _config(kp=0.1, ki=1.0))
    for _ in range(20):
        _compute(controller, _state())
    controller.set_reference(_reference(0.0))
    action = _compute(controller, _state(speed=speed))
    assert action.throttle_brake <= 0.0
    if speed > 0.0:
        # Once the existing integral unwinds, moving ego brakes.
        for _ in range(20):
            action = _compute(controller, _state(speed=speed))
        assert action.throttle_brake < 0.0


@pytest.mark.parametrize("speed", [-100.0, 100.0])
def test_actuator_saturation_and_conditional_integration_prevent_windup(speed: float) -> None:
    config = _config(kp=1.0, ki=1.0)
    controller = _controller(_reference(5.0), config)
    for _ in range(100):
        action = _compute(controller, _state(rear_y=-10.0, speed=speed))
        assert action.throttle_brake == (1.0 if speed < 0.0 else -1.0)
        assert -1.0 <= action.steering <= 1.0
    assert _compute(controller, _state(speed=5.0)).throttle_brake == pytest.approx(0.0)


def test_integral_bound_and_reverse_error_unwind_existing_integral() -> None:
    controller = _controller(_reference(0.5), _config(kp=0.0, ki=1.0, integral_limit=0.1))
    for _ in range(100):
        action = _compute(controller, _state())
    assert action.throttle_brake == pytest.approx(0.1)
    assert _compute(controller, _state(speed=1.0)).throttle_brake == pytest.approx(0.09)


def test_saturated_existing_integral_can_unwind() -> None:
    controller = _controller(_reference(0.5), _config(kp=1.0, ki=1.0))
    for _ in range(20):
        _compute(controller, _state())
    assert _compute(controller, _state(speed=-5.0)).throttle_brake == 1.0
    # Opposite error must reduce the retained integral on the next tick.
    assert _compute(controller, _state(speed=0.6)).throttle_brake == pytest.approx(0.098)


def test_replan_preserves_integral_and_episode_reset_clears_it_and_reference() -> None:
    controller = _controller(_reference(0.5), _config(kp=0.0, ki=1.0))
    first = _compute(controller, _state())
    controller.set_reference(_reference(0.5))
    second = _compute(controller, _state())
    assert second.throttle_brake == pytest.approx(2.0 * first.throttle_brake)
    controller.reset()
    with pytest.raises(RuntimeError, match="requires a reference"):
        _compute(controller, _state())
    controller.set_reference(_reference(0.5))
    assert _compute(controller, _state()) == first


def test_derivative_on_measurement_survives_replan_without_target_kick() -> None:
    controller = _controller(_reference(5.0), _config(kp=0.0, ki=0.0, kd=0.01))
    assert _compute(controller, _state(speed=2.0)).throttle_brake == 0.0
    controller.set_reference(_reference(20.0))
    assert _compute(controller, _state(speed=2.0)).throttle_brake == 0.0
    assert _compute(controller, _state(speed=3.0)).throttle_brake == pytest.approx(-0.5)
    controller.reset()
    controller.set_reference(_reference())
    assert _compute(controller, _state(speed=4.0)).throttle_brake == 0.0


def test_longitudinal_feedback_projects_actual_world_velocity() -> None:
    state = TrackingVehicleState(np.array([0.0, 1.0]), np.pi / 2, np.array([20.0, 3.0]))
    action = _compute(_controller(config=_config(kp=0.1, ki=0.0)), state)
    assert action.throttle_brake == pytest.approx(0.2)


def test_rigid_transform_preserves_control() -> None:
    reference = _reference()
    state = _state(rear_y=0.7, heading=0.2, speed=3.0)
    angle = 2.8
    translation = np.array([130.0, -29.0])
    transformed = replace(
        reference,
        centers=local_points_to_world(reference.centers, translation, angle),
        headings=reference.headings + angle,
        velocities=local_points_to_world(reference.velocities, np.zeros(2), angle),
    )
    rotated_state = TrackingVehicleState(
        local_points_to_world(state.center_position[None], translation, angle)[0],
        state.heading_rad + angle,
        local_points_to_world(state.velocity[None], np.zeros(2), angle)[0],
    )
    original_action = _compute(_controller(reference), state, 0.12)
    rotated_action = _compute(_controller(transformed), rotated_state, 0.12)
    assert rotated_action.steering == pytest.approx(original_action.steering)
    assert rotated_action.throttle_brake == pytest.approx(original_action.throttle_brake)


def test_inputs_are_not_modified_and_reference_and_state_are_copied() -> None:
    reference = _reference()
    snapshots = {name: getattr(reference, name).copy() for name in reference.__dataclass_fields__}
    state = _state(speed=2.0)
    controller = _controller(reference, _config(ki=0.0))
    action = _compute(controller, state)
    for name, snapshot in snapshots.items():
        np.testing.assert_array_equal(getattr(reference, name), snapshot)
        getattr(reference, name)[:] = 123.0
    assert _compute(controller, state) == action
    np.testing.assert_array_equal(state.center_position, [1.0, 0.0])
    np.testing.assert_array_equal(state.velocity, [2.0, 0.0])
    center = np.array([1.0, 0.0])
    velocity = np.array([2.0, 0.0])
    copied_state = TrackingVehicleState(center, 0.0, velocity)
    center[:] = 99.0
    velocity[:] = 99.0
    np.testing.assert_array_equal(copied_state.center_position, [1.0, 0.0])
    np.testing.assert_array_equal(copied_state.velocity, [2.0, 0.0])
    with pytest.raises(ValueError):
        copied_state.velocity[0] = 0.0


@pytest.mark.parametrize("field", ["centers", "headings", "velocities", "angular_velocities"])
@pytest.mark.parametrize("problem", ["shape", "dtype", "nan", "inf"])
def test_invalid_reference_is_rejected_without_replacing_cache(field: str, problem: str) -> None:
    reference = _reference()
    array = getattr(reference, field).copy()
    if problem == "shape":
        array = array[:-1]
    elif problem == "dtype":
        array = array.astype(np.float32)
    else:
        array.flat[0] = np.nan if problem == "nan" else np.inf
    controller = _controller(reference, _config(ki=0.0))
    before = _compute(controller, _state())
    with pytest.raises(ValueError, match=field):
        controller.set_reference(replace(reference, **{field: array}))
    assert _compute(controller, _state()) == before


@pytest.mark.parametrize("time", [-0.02, np.nan, np.inf, True, "0"])
def test_invalid_reference_time_fails(time) -> None:
    with pytest.raises(ValueError, match="reference_time_s"):
        _compute(_controller(), _state(), time)


@pytest.mark.parametrize("dt", [0.0, -0.02, np.nan, np.inf, False, "0.02"])
def test_invalid_dt_fails(dt) -> None:
    with pytest.raises(ValueError, match="dt_s"):
        _controller().compute(_state(), 0.0, dt)


@pytest.mark.parametrize("field", ["center_position", "velocity", "heading_rad"])
def test_nonfinite_actual_state_fails(field: str) -> None:
    state = _state()
    value = np.nan if field == "heading_rad" else np.array([np.inf, 0.0])
    with pytest.raises(ValueError, match=field):
        replace(state, **{field: value})


@pytest.mark.parametrize("value", [np.zeros(3), np.zeros(2, dtype=np.float32), [0.0, 0.0]])
def test_invalid_actual_velocity_shape_or_dtype_fails(value) -> None:
    with pytest.raises(ValueError, match="velocity"):
        replace(_state(), velocity=value)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("enable_reverse", True),
        ("enable_reverse", 0),
        ("front_wheelbase_m", 0.0),
        ("rear_wheelbase_m", -1.0),
        ("max_steering_deg", 0.0),
        ("max_steering_deg", np.inf),
    ],
)
def test_invalid_vehicle_parameters_fail(field: str, value) -> None:
    with pytest.raises(ValidationError):
        TrackingVehicleParameters.model_validate(_VEHICLE.model_dump() | {field: value})


def test_configuration_roundtrip_is_explicit_strict_and_frozen() -> None:
    config = TrajectoryTrackingConfig.model_validate(
        OmegaConf.to_container(OmegaConf.load(_CONFIG_PATH), resolve=True)
    )
    assert config.longitudinal.ki == 0.1
    assert config.longitudinal.kd == 0.0
    assert TrajectoryTrackingConfig.model_validate_json(config.model_dump_json()) == config
    dumped = OmegaConf.create(config.model_dump())
    assert TrajectoryTrackingConfig.model_validate(OmegaConf.to_container(dumped)) == config
    with pytest.raises(ValidationError):
        config.lookahead.base_m = 99.0
    with pytest.raises(FrozenInstanceError):
        _state().heading_rad = 99.0
    for field in TrajectoryTrackingConfig.model_fields:
        mapping = config.model_dump()
        del mapping[field]
        with pytest.raises(ValidationError):
            TrajectoryTrackingConfig.model_validate(mapping)
    with pytest.raises(ValidationError):
        TrajectoryTrackingConfig.model_validate(config.model_dump() | {"legacy": True})
    for section in ("lookahead", "longitudinal"):
        for field in config.model_dump()[section]:
            mapping = config.model_dump()
            del mapping[section][field]
            with pytest.raises(ValidationError):
                TrajectoryTrackingConfig.model_validate(mapping)


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("lookahead", "min_m", 0.0),
        ("lookahead", "min_m", 100.0),
        ("lookahead", "base_m", -1.0),
        ("lookahead", "speed_gain_s", -1.0),
        ("longitudinal", "kp", "0.5"),
        ("longitudinal", "ki", True),
        ("longitudinal", "kd", -1.0),
        ("longitudinal", "kp", np.nan),
        ("longitudinal", "integral_limit", np.inf),
        ("longitudinal", "output_limit", 1.1),
        ("longitudinal", "output_limit", 0.0),
        ("longitudinal", "extra", 0.0),
    ],
)
def test_invalid_configuration_fails(section: str, field: str, value) -> None:
    mapping = _config().model_dump()
    mapping[section][field] = value
    with pytest.raises(ValidationError):
        TrajectoryTrackingConfig.model_validate(mapping)


def test_nonfinite_computation_fails_before_saturation_and_preserves_pid_memory() -> None:
    controller = _controller(_reference(0.5), _config(kp=0.1, ki=1.0))
    first = _compute(controller, _state())
    with pytest.raises((ValueError, FloatingPointError)):
        controller.compute(_state(speed=-1e308), 0.0, 100.0)
    second = _compute(controller, _state())
    assert second.throttle_brake == pytest.approx(first.throttle_brake + 0.01)
