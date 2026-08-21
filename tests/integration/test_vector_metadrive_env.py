from __future__ import annotations

import numpy as np
import pytest
import torch

from eco_planner.envs import (
    MetaDriveObservationAdapter,
    NoTrafficMetaDriveObservationAdapter,
    TrajectoryMetaDriveEnv,
    VectorEnvScenario,
    VectorMetaDriveEnv,
)
from eco_planner.models.config import OfficialDiffusionPlannerConfig


def _environment_config(map_sequence: str) -> dict[str, object]:
    return {
        "use_render": False,
        "map": map_sequence,
        "num_scenarios": 4,
        "traffic_density": 0.0,
        "random_traffic": False,
        "random_spawn_lane_index": False,
        "physics_world_step_size": 0.02,
        "decision_repeat": 5,
        "trajectory_horizon": 80,
        "trajectory_execution_steps": 5,
        "programmatic_lane_speed_limit_kmh": 50.0,
    }


def _straight_trajectory(speed_mps: float = 5.0) -> np.ndarray:
    trajectory = np.zeros((80, 4), dtype=np.float32)
    trajectory[:, 0] = np.arange(1, 81, dtype=np.float32) * speed_mps * 0.1
    trajectory[:, 2] = 1.0
    return trajectory


def _stationary_trajectory() -> np.ndarray:
    trajectory = np.zeros((80, 4), dtype=np.float32)
    trajectory[:, 2] = 1.0
    return trajectory


def _assert_same_observation(
    actual: dict[str, torch.Tensor], expected: dict[str, torch.Tensor]
) -> None:
    assert set(actual) == set(expected)
    for name in actual:
        torch.testing.assert_close(actual[name], expected[name], rtol=0.0, atol=1e-5)


def _warmup_traffic(
    env: TrajectoryMetaDriveEnv,
    adapter: MetaDriveObservationAdapter,
) -> None:
    for _ in range(4):
        _, _, terminated, truncated, info = env.step(_stationary_trajectory())
        assert not terminated
        assert not truncated
        adapter.append_frames(info["trajectory_execution"].traffic_frames)


@pytest.mark.simulator
@pytest.mark.parametrize("num_envs", (1, 2, 4))
def test_vector_environment_fixed_slots_complete_no_traffic_lifecycle(
    official_model_config: OfficialDiffusionPlannerConfig,
    num_envs: int,
) -> None:
    config = _environment_config("S")
    scenarios = tuple(
        VectorEnvScenario(name=f"slot-{slot}", map="S", seed=slot) for slot in range(num_envs)
    )
    with VectorMetaDriveEnv(
        [config.copy() for _ in range(num_envs)],
        mode="no_traffic",
        model_config=official_model_config,
        map_query_radius_m=100.0,
        history_warmup_steps=0,
    ) as envs:
        resets = envs.reset(scenarios)
        steps = envs.step([_straight_trajectory() for _ in range(num_envs)])

    assert [result.slot for result in resets] == list(range(num_envs))
    assert [result.slot for result in steps] == list(range(num_envs))
    for reset, step in zip(resets, steps, strict=True):
        assert reset.scenario == scenarios[reset.slot]
        assert reset.observation["ego_current_state"].shape == (10,)
        assert all(value.device.type == "cpu" for value in reset.observation.values())
        assert reset.timing.environment_s >= 0.0
        assert reset.timing.observation_s >= 0.0
        assert reset.timing.ipc_send_s >= 0.0
        assert reset.timing.ipc_receive_s >= 0.0
        assert step.execution.substep_states.shape == (5, 7)
        assert not step.terminated
        assert not step.truncated


@pytest.mark.simulator
def test_resetting_one_vector_slot_does_not_change_another_slot(
    official_model_config: OfficialDiffusionPlannerConfig,
) -> None:
    config = _environment_config("S")
    first = VectorEnvScenario(name="first", map="S", seed=2)
    second = VectorEnvScenario(name="second", map="S", seed=1)
    replacement = VectorEnvScenario(name="replacement", map="S", seed=3)
    baseline = TrajectoryMetaDriveEnv(config.copy())
    adapter = NoTrafficMetaDriveObservationAdapter(official_model_config, 100.0)
    try:
        baseline.reset(seed=second.seed)
        adapter.reset(baseline)
        expected_reset_observation = adapter.build(baseline)
        _, expected_reward, expected_terminated, expected_truncated, expected_info = baseline.step(
            _straight_trajectory()
        )
        expected_next_observation = adapter.build(baseline)

        with VectorMetaDriveEnv(
            [config.copy(), config.copy()],
            mode="no_traffic",
            model_config=official_model_config,
            map_query_radius_m=100.0,
            history_warmup_steps=0,
        ) as envs:
            resets = envs.reset((first, second))
            envs.reset_at(0, replacement)
            actual = envs.step_at(1, _straight_trajectory())
    finally:
        baseline.close()

    _assert_same_observation(resets[1].observation, expected_reset_observation)
    _assert_same_observation(actual.observation, expected_next_observation)
    assert actual.reward == pytest.approx(expected_reward, abs=1e-12)
    assert actual.terminated is expected_terminated
    assert actual.truncated is expected_truncated
    np.testing.assert_allclose(
        actual.execution.substep_states,
        expected_info["trajectory_execution"].substep_states,
        rtol=0.0,
        atol=1e-12,
    )


@pytest.mark.simulator
def test_vector_environment_builds_traffic_observations_after_worker_warmup(
    official_model_config: OfficialDiffusionPlannerConfig,
) -> None:
    config = _environment_config("SC")
    config.update({"traffic_density": 0.05, "traffic_mode": "trigger", "horizon": 30})
    baseline = TrajectoryMetaDriveEnv(config.copy())
    adapter = MetaDriveObservationAdapter(official_model_config, 100.0)
    try:
        baseline.reset(seed=0)
        adapter.reset(baseline.initial_traffic_frame, env=baseline)
        _warmup_traffic(baseline, adapter)
        expected_reset_observation = adapter.build(baseline)
        _, expected_reward, expected_terminated, expected_truncated, expected_info = baseline.step(
            _stationary_trajectory()
        )
        adapter.append_frames(expected_info["trajectory_execution"].traffic_frames)
        expected_next_observation = adapter.build(baseline)

        with VectorMetaDriveEnv(
            [config],
            mode="traffic",
            model_config=official_model_config,
            map_query_radius_m=100.0,
            history_warmup_steps=20,
        ) as envs:
            reset = envs.reset((VectorEnvScenario(name="traffic", map="SC", seed=0),))[0]
            step = envs.step((_stationary_trajectory(),))[0]
    finally:
        baseline.close()

    assert reset.observation["neighbor_agents_past"].shape == (32, 21, 11)
    assert torch.count_nonzero(reset.observation["neighbor_agents_past"]).item() > 0
    assert step.execution.traffic_frames[-1].simulator_step == 25
    assert step.observation["neighbor_agents_past"].shape == (32, 21, 11)
    _assert_same_observation(reset.observation, expected_reset_observation)
    _assert_same_observation(step.observation, expected_next_observation)
    assert step.reward == pytest.approx(expected_reward, abs=1e-12)
    assert step.terminated is expected_terminated
    assert step.truncated is expected_truncated
    np.testing.assert_allclose(
        step.execution.substep_states,
        expected_info["trajectory_execution"].substep_states,
        rtol=0.0,
        atol=1e-12,
    )
