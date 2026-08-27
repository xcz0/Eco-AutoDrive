"""Small real-MetaDrive boundary checks for evaluation and rollout."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from eco_planner.envs import (
    PlannerObservationSpec,
    TrajectoryMetaDriveEnv,
    VectorEnvScenario,
    VectorMetaDriveEnv,
)
from eco_planner.envs.metadrive.observation import MetaDriveObservationAdapter
from eco_planner.models.config import OfficialDiffusionPlannerConfig


def _environment_config(
    map_sequence: str,
    *,
    traffic_density: float = 0.0,
    trajectory_execution_steps: int = 5,
) -> dict[str, object]:
    return {
        "use_render": False,
        "map": map_sequence,
        "num_scenarios": 2,
        "traffic_density": traffic_density,
        "random_traffic": False,
        "random_spawn_lane_index": False,
        "physics_world_step_size": 0.02,
        "decision_repeat": 5,
        "trajectory_horizon": 80,
        "trajectory_execution_steps": trajectory_execution_steps,
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


@pytest.mark.simulator
def test_trajectory_environment_executes_evaluation_prefix_with_valid_audit() -> None:
    env = TrajectoryMetaDriveEnv(_environment_config("S"))
    try:
        env.reset(seed=0)
        _, reward, terminated, truncated, info = env.step(_straight_trajectory())
        execution = info["trajectory_execution"]

        assert execution.substep_states.shape == (5, 7)
        assert execution.substep_terminated.shape == (5,)
        assert execution.substep_truncated.shape == (5,)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert np.isfinite(reward)
        assert np.isfinite(execution.substep_states).all()
        assert np.isfinite(execution.substep_native_energy_ml).all()
        assert np.isfinite(execution.substep_executed_fuel_proxy_energy_ml).all()
        assert np.all(execution.substep_native_energy_ml >= 0.0)
        assert np.all(execution.substep_executed_fuel_proxy_energy_ml >= 0.0)
        np.testing.assert_allclose(
            execution.substep_states[-1, :2], np.asarray(env.agent.position), atol=1e-3
        )
    finally:
        env.close()


@pytest.mark.simulator
def test_traffic_history_enters_planner_observation(
    official_model_config: OfficialDiffusionPlannerConfig,
) -> None:
    config = _environment_config("SC", traffic_density=0.1)
    config["horizon"] = 30
    env = TrajectoryMetaDriveEnv(config)
    adapter = MetaDriveObservationAdapter(official_model_config, 100.0)
    try:
        env.reset(seed=0)
        adapter.reset(env, env.initial_traffic_frame)
        for _ in range(4):
            _, _, terminated, truncated, info = env.step(_stationary_trajectory())
            assert not terminated and not truncated
            adapter.append_frames(info["trajectory_execution"].traffic_frames)
        observation, audit = adapter.build(env)
    finally:
        env.close()

    assert observation["neighbor_agents_past"].shape == (32, 21, 11)
    assert observation["static_objects"].shape == (5, 10)
    assert observation["lanes"].shape == (70, 20, 12)
    assert observation["route_lanes"].shape == (25, 20, 12)
    assert torch.count_nonzero(observation["neighbor_agents_past"]).item() > 0
    assert audit.participant_count_in_radius > 0
    assert all(
        torch.isfinite(value).all() for value in observation.values() if value.is_floating_point()
    )


@pytest.mark.simulator
def test_two_slot_vector_rollout_executes_current_training_path(
    official_model_config: OfficialDiffusionPlannerConfig,
) -> None:
    scenarios = (
        VectorEnvScenario(name="slot-0", map="S", seed=0),
        VectorEnvScenario(name="slot-1", map="S", seed=1),
    )
    with VectorMetaDriveEnv(
        [_environment_config("S", trajectory_execution_steps=1) for _ in scenarios],
        mode="no_traffic",
        observation_spec=PlannerObservationSpec.from_planner_config(official_model_config),
        map_query_radius_m=100.0,
        history_warmup_steps=0,
        scenarios=scenarios,
    ) as envs:
        resets = envs.reset(scenarios)
        steps = envs.step([_straight_trajectory() for _ in scenarios])

    assert [item.slot for item in resets] == [0, 1]
    assert [item.slot for item in steps] == [0, 1]
    for reset, step in zip(resets, steps, strict=True):
        assert reset.observation["ego_current_state"].shape == (10,)
        assert step.execution.substep_states.shape == (1, 7)
        assert isinstance(step.terminated, bool)
        assert isinstance(step.truncated, bool)
        assert np.isfinite(step.reward)
