"""Cross-entry execution consistency for the canonical closed-loop cadence.

Training rollout and evaluation run the same real closed-loop system. These tests
pin the shared execution semantics at the environment boundary: the serial slot
used by evaluation and the fixed-slot vector environment used by training must
produce identical prefixes, domain facts and raw terminal facts for the same
scenario, initial state and trajectory.
"""

from __future__ import annotations

import numpy as np
import pytest

from eco_planner.contracts import (
    CLOSED_LOOP_EXECUTION_STEPS,
    DECISION_INTERVAL_S,
    SIMULATOR_STEP_S,
    evaluation_plan_cycles,
)
from eco_planner.envs import MetaDriveEnvSlot
from eco_planner.evaluation.artifacts.trace import allocate_trace_arrays
from eco_planner.runtime.envs import (
    VectorEnvScenario,
    VectorMetaDriveEnv,
    WorkerResetResult,
    WorkerStepResult,
    operation_results,
)
from tests.simulation.test_closed_loop import (
    _environment_config,
    _off_route_trajectory,
    _straight_trajectory,
)

_RECORD_TERMINAL_FIELDS = (
    "arrive_dest",
    "out_of_road",
    "crash_vehicle",
    "crash_object",
    "crash_building",
    "crash_human",
    "crash_sidewalk",
)


def _domain_facts(step) -> dict[str, object]:
    metrics = step.metrics
    return {
        "speeds_mps": np.asarray([item.speed_mps for item in metrics]),
        "stopped": np.asarray([item.stopped for item in metrics]),
        "wrong_direction": np.asarray([item.wrong_direction for item in metrics]),
        "collision": np.asarray([item.collision for item in metrics]),
        "step_distance_m": np.asarray([item.step_distance_m for item in metrics]),
        "route_heading_error_rad": np.asarray([item.route_heading_error_rad for item in metrics]),
        "position_error_m": np.asarray([item.position_error_m for item in metrics]),
        "heading_error_rad": np.asarray([item.heading_error_rad for item in metrics]),
    }


def _assert_execution_parity(serial_step, vector_step) -> None:
    serial = serial_step.execution
    vector = vector_step.execution

    assert serial.substep_states.shape[0] == vector.substep_states.shape[0]
    np.testing.assert_allclose(serial.substep_states, vector.substep_states, rtol=0, atol=1e-6)
    np.testing.assert_allclose(serial.start_center, vector.start_center, rtol=0, atol=1e-6)
    np.testing.assert_allclose(serial.world_centers, vector.world_centers, rtol=0, atol=1e-6)
    np.testing.assert_allclose(serial.world_headings, vector.world_headings, rtol=0, atol=1e-6)
    np.testing.assert_array_equal(serial.substep_terminated, vector.substep_terminated)
    np.testing.assert_array_equal(serial.substep_truncated, vector.substep_truncated)
    assert serial.route_completion == pytest.approx(vector.route_completion, abs=1e-9)

    serial_facts = _domain_facts(serial_step)
    vector_facts = _domain_facts(vector_step)
    for name, serial_values in serial_facts.items():
        vector_values = vector_facts[name]
        assert serial_values.shape == vector_values.shape
        if serial_values.dtype == np.bool_:
            np.testing.assert_array_equal(serial_values, vector_values)
        else:
            np.testing.assert_allclose(serial_values, vector_values, rtol=0, atol=1e-6)

    for name in _RECORD_TERMINAL_FIELDS:
        assert getattr(serial, name) == getattr(vector, name), name
    assert serial_step.terminated is vector_step.terminated
    assert serial_step.truncated is vector_step.truncated


@pytest.mark.simulator
def test_serial_and_vector_execution_share_canonical_prefix_and_facts() -> None:
    config = _environment_config("S")
    scenario = VectorEnvScenario(name="slot-0", map="S", seed=0)
    trajectory = _straight_trajectory()

    with MetaDriveEnvSlot(
        dict(config),
        mode="no_traffic",
        map_query_radius_m=100.0,
        history_warmup_steps=0,
    ) as serial_slot:
        serial_reset = serial_slot.reset(map_name=scenario.map, seed=scenario.seed)
        serial_step = serial_slot.step(trajectory).execution

    with VectorMetaDriveEnv(
        [dict(config)],
        mode="no_traffic",
        map_query_radius_m=100.0,
        history_warmup_steps=0,
        scenarios=(scenario,),
    ) as envs:
        resets = envs.reset((scenario,))
        steps = envs.step((trajectory,))
        reset_result = operation_results(resets, WorkerResetResult)[0]
        vector_step = operation_results(steps, WorkerStepResult)[0].step

    np.testing.assert_allclose(
        serial_reset.state.vehicle_state[:2], reset_result.initial_state[:2], rtol=0, atol=1e-6
    )
    assert serial_step.execution.substep_states.shape[0] == CLOSED_LOOP_EXECUTION_STEPS
    assert vector_step.execution.substep_states.shape[0] == CLOSED_LOOP_EXECUTION_STEPS
    assert vector_step.execution.substep_states.shape[0] * SIMULATOR_STEP_S == pytest.approx(
        DECISION_INTERVAL_S
    )
    _assert_execution_parity(serial_step, vector_step)


@pytest.mark.simulator
def test_early_terminal_truncates_prefix_consistently_across_entries() -> None:
    config = _environment_config("SXS")
    query_radius_m = 5.0
    scenario = VectorEnvScenario(name="off-route", map="SXS", seed=0)

    with MetaDriveEnvSlot(
        dict(config),
        mode="no_traffic",
        map_query_radius_m=query_radius_m,
        history_warmup_steps=0,
    ) as source:
        source.reset(map_name=scenario.map, seed=scenario.seed)
        trajectory = _off_route_trajectory(source.backend, query_radius_m)

    with MetaDriveEnvSlot(
        dict(config),
        mode="no_traffic",
        map_query_radius_m=query_radius_m,
        history_warmup_steps=0,
    ) as serial_slot:
        serial_slot.reset(map_name=scenario.map, seed=scenario.seed)
        serial_step = serial_slot.step(trajectory).execution

    with VectorMetaDriveEnv(
        [dict(config)],
        mode="no_traffic",
        map_query_radius_m=query_radius_m,
        history_warmup_steps=0,
        scenarios=(scenario,),
    ) as envs:
        envs.reset((scenario,))
        steps = envs.step((trajectory,))
        vector_step = operation_results(steps, WorkerStepResult)[0].step

    executed = serial_step.execution.substep_states.shape[0]
    assert 1 <= executed < CLOSED_LOOP_EXECUTION_STEPS
    assert executed == vector_step.execution.substep_states.shape[0]
    assert serial_step.execution.out_of_road is True
    assert serial_step.terminated is True
    assert bool(serial_step.execution.substep_terminated[-1]) is True
    assert not serial_step.execution.substep_terminated[:-1].any()
    _assert_execution_parity(serial_step, vector_step)


def test_horizon_and_cycle_conversion_uses_one_canonical_prefix() -> None:
    for horizon in (1, 5, 6, 20, 300):
        cycles = evaluation_plan_cycles(horizon)
        covered = cycles * CLOSED_LOOP_EXECUTION_STEPS
        assert covered >= horizon
        assert covered - horizon < CLOSED_LOOP_EXECUTION_STEPS

    arrays = allocate_trace_arrays(max_plan_cycles=4, max_warmup_steps=0, guided=False)
    assert arrays["executed_states"].shape == (4 * CLOSED_LOOP_EXECUTION_STEPS, 7)
    assert arrays["executed_plan_indices"].shape == (4 * CLOSED_LOOP_EXECUTION_STEPS,)
    assert arrays["planning_anchors"].shape == (4, 7)
    assert DECISION_INTERVAL_S == pytest.approx(CLOSED_LOOP_EXECUTION_STEPS * SIMULATOR_STEP_S)
