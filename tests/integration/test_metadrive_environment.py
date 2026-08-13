from __future__ import annotations

import numpy as np
import pytest
import torch

from eco_planner.envs import (
    MetaDriveMapAdapter,
    MetaDriveObservationAdapter,
    NoTrafficMetaDriveObservationAdapter,
    TrajectoryExecutionRecord,
    TrajectoryMetaDriveEnv,
)
from eco_planner.evaluation.runtime import FabricInferenceRuntime
from eco_planner.models.config import OfficialDiffusionPlannerConfig


def _environment_config(map_sequence: str) -> dict[str, object]:
    return {
        "use_render": False,
        "map": map_sequence,
        "num_scenarios": 1,
        "traffic_density": 0.0,
        "random_traffic": False,
        "random_spawn_lane_index": False,
        "physics_world_step_size": 0.02,
        "decision_repeat": 5,
        "trajectory_horizon": 80,
        "trajectory_execution_steps": 5,
        "programmatic_lane_speed_limit_kmh": 50.0,
    }


def _straight_trajectory(speed_mps: float) -> np.ndarray:
    trajectory = np.zeros((80, 4), dtype=np.float32)
    trajectory[:, 0] = np.arange(1, 81, dtype=np.float32) * speed_mps * 0.1
    trajectory[:, 2] = 1.0
    return trajectory


def _turning_trajectory() -> np.ndarray:
    trajectory = _straight_trajectory(5.0)
    headings = np.linspace(0.02, 1.6, 80, dtype=np.float32)
    trajectory[:, 1] = np.linspace(0.01, 3.0, 80, dtype=np.float32)
    trajectory[:, 2] = np.cos(headings)
    trajectory[:, 3] = np.sin(headings)
    return trajectory


def _stationary_trajectory() -> np.ndarray:
    trajectory = np.zeros((80, 4), dtype=np.float32)
    trajectory[:, 2] = 1.0
    return trajectory


@pytest.mark.simulator
def test_trajectory_environment_executes_five_simulator_steps() -> None:
    env = TrajectoryMetaDriveEnv(_environment_config("S"))
    try:
        env.reset(seed=0)
        start_position = np.asarray(env.agent.position, dtype=np.float64)
        start_heading = float(env.agent.heading_theta)

        _, reward, terminated, truncated, info = env.step(_straight_trajectory(5.0))
        execution = TrajectoryExecutionRecord.from_info(info)

        displacement = np.asarray(env.agent.position, dtype=np.float64) - start_position
        forward_progress = float(
            displacement @ np.array([np.cos(start_heading), np.sin(start_heading)])
        )
        assert env.action_space.shape == (80, 4)
        assert env.engine.episode_step == 5
        assert info["trajectory_execution_steps"] == 5
        assert info["trajectory_reward_sum"] == pytest.approx(reward)
        assert info["trajectory_world_centers"].shape == (80, 2)
        assert info["trajectory_world_headings"].shape == (80,)
        assert info["trajectory_substep_states"].shape == (5, 7)
        assert info["trajectory_substep_rewards"].shape == (5,)
        assert info["trajectory_substep_terminated"].shape == (5,)
        assert info["trajectory_substep_truncated"].shape == (5,)
        assert info["trajectory_target_centers"].shape == (5, 2)
        assert info["trajectory_target_headings"].shape == (5,)
        assert info["trajectory_position_errors_m"].shape == (5,)
        assert info["trajectory_heading_errors_rad"].shape == (5,)
        assert execution.substep_states.shape == (5, 7)
        assert execution.route_completion == pytest.approx(info["route_completion"])
        np.testing.assert_allclose(
            info["trajectory_substep_states"][-1, :2],
            np.asarray(env.agent.position),
        )
        np.testing.assert_allclose(
            info["trajectory_substep_states"][:, :2],
            info["trajectory_target_centers"],
            atol=1e-3,
        )
        np.testing.assert_allclose(
            info["trajectory_substep_states"][:, 2],
            info["trajectory_target_headings"],
            atol=1e-4,
        )
        assert forward_progress == pytest.approx(2.5, abs=1e-3)
        assert float(env.agent.heading_theta) == pytest.approx(start_heading, abs=1e-4)
        assert float(env.agent.speed) == pytest.approx(5.0, abs=0.1)
        assert np.isfinite(reward)
        assert not terminated
        assert not truncated
    finally:
        env.close()


@pytest.mark.simulator
def test_trajectory_environment_matches_variable_heading_waypoints() -> None:
    env = TrajectoryMetaDriveEnv(_environment_config("S"))
    try:
        env.reset(seed=0)
        _, _, _, _, info = env.step(_turning_trajectory())

        np.testing.assert_allclose(
            info["trajectory_substep_states"][:, :2],
            info["trajectory_target_centers"],
            atol=1e-3,
        )
        np.testing.assert_allclose(
            info["trajectory_substep_states"][:, 2],
            info["trajectory_target_headings"],
            atol=1e-4,
        )
    finally:
        env.close()


@pytest.mark.simulator
def test_trajectory_environment_returns_consecutive_traffic_frames() -> None:
    config = _environment_config("SC")
    config["traffic_density"] = 0.1
    env = TrajectoryMetaDriveEnv(config)
    try:
        env.reset(seed=0)
        initial = env.initial_traffic_frame
        start_position = np.asarray(env.agent.position, dtype=np.float64).copy()

        _, _, terminated, truncated, info = env.step(_stationary_trajectory())

        frames = info["traffic_substep_frames"]
        assert initial.simulator_step == 0
        assert tuple(frame.simulator_step for frame in frames) == (1, 2, 3, 4, 5)
        assert any(frame.participants for frame in frames)
        assert not terminated
        assert not truncated
        np.testing.assert_allclose(env.agent.position, start_position, atol=1e-3)
    finally:
        env.close()


@pytest.mark.simulator
def test_traffic_adapter_builds_after_real_two_second_warmup(
    official_model_config: OfficialDiffusionPlannerConfig,
) -> None:
    config = _environment_config("SC")
    config["traffic_density"] = 0.1
    config["horizon"] = 30
    env = TrajectoryMetaDriveEnv(config)
    adapter = MetaDriveObservationAdapter(official_model_config, 100.0)
    try:
        env.reset(seed=0)
        start_position = np.asarray(env.agent.position, dtype=np.float64).copy()
        adapter.reset(env.initial_traffic_frame)
        for _ in range(4):
            _, _, terminated, truncated, info = env.step(_stationary_trajectory())
            assert not terminated
            assert not truncated
            adapter.append_frames(info["traffic_substep_frames"])

        observation = adapter.build(env)

        assert observation["neighbor_agents_past"].shape == (1, 32, 21, 11)
        assert torch.count_nonzero(observation["neighbor_agents_past"]).item() > 0
        assert adapter.last_audit.participant_count_in_radius > 0
        np.testing.assert_allclose(env.agent.position, start_position, atol=1e-3)
    finally:
        env.close()


@pytest.mark.simulator
@pytest.mark.parametrize(
    ("map_sequence", "expected_speed_limit_counts"),
    [("S", {50.0: 18}), ("SC", {20.0: 12, 50.0: 18})],
)
def test_programmatic_lane_speed_limits_replace_only_unset_sentinel(
    map_sequence: str,
    expected_speed_limit_counts: dict[float, int],
) -> None:
    env = TrajectoryMetaDriveEnv(_environment_config(map_sequence))
    try:
        env.reset(seed=0)
        first_audit = env.programmatic_lane_speed_limit_audit
        lanes = env.current_map.road_network.get_all_lanes()
        first_limits = [float(lane.speed_limit) for lane in lanes]
        env.reset(seed=0)
        second_audit = env.programmatic_lane_speed_limit_audit
        lanes = env.current_map.road_network.get_all_lanes()
        second_limits = [float(lane.speed_limit) for lane in lanes]

        assert 1000.0 not in first_limits
        assert first_limits == second_limits
        assert first_audit == second_audit
        assert first_audit["speed_limit_sentinel_replaced_count"] == 18
        assert first_audit["speed_limit_existing_preserved_count"] == sum(
            count
            for speed_limit, count in expected_speed_limit_counts.items()
            if speed_limit != 50.0
        )
        observed_counts = {
            speed_limit: first_limits.count(speed_limit) for speed_limit in set(first_limits)
        }
        assert observed_counts == expected_speed_limit_counts
    finally:
        env.close()


@pytest.mark.simulator
@pytest.mark.parametrize("map_sequence", ["S", "SC"])
def test_map_adapter_is_deterministic_on_programmatic_maps(
    map_sequence: str,
    official_model_config: OfficialDiffusionPlannerConfig,
) -> None:
    env = TrajectoryMetaDriveEnv(_environment_config(map_sequence))
    adapter = MetaDriveMapAdapter(official_model_config, query_radius_m=100.0)
    try:
        env.reset(seed=0)
        first = adapter.build(env)
        env.reset(seed=0)
        second = adapter.build(env)

        assert first["lanes"].shape == (1, 70, 20, 12)
        assert first["route_lanes"].shape == (1, 25, 20, 12)
        assert torch.count_nonzero(first["route_lanes"]).item() > 0
        for name in first:
            if first[name].dtype == torch.float32:
                assert torch.isfinite(first[name]).all()
            torch.testing.assert_close(first[name], second[name], rtol=0.0, atol=0.0)
    finally:
        env.close()


@pytest.mark.simulator
def test_programmatic_map_adapter_encodes_configured_speed_limit_semantics(
    official_model_config: OfficialDiffusionPlannerConfig,
) -> None:
    env = TrajectoryMetaDriveEnv(_environment_config("S"))
    adapter = MetaDriveMapAdapter(official_model_config, query_radius_m=100.0)
    try:
        env.reset(seed=0)
        observation = adapter.build(env)
        valid = observation["lanes_has_speed_limit"]
        observed_speed_limits = observation["lanes_speed_limit"][valid].detach().cpu().numpy()

        assert observed_speed_limits.size > 0
        assert not np.isclose(observed_speed_limits, 1000.0 / 3.6).any()
        np.testing.assert_allclose(observed_speed_limits, 50.0 / 3.6, atol=1e-6)
    finally:
        env.close()


@pytest.mark.simulator
@pytest.mark.slow
@pytest.mark.parametrize("map_sequence", ["S", "SC"])
def test_official_planner_executes_no_traffic_closed_loop_cycle(
    map_sequence: str,
    stage0_runtime: FabricInferenceRuntime,
) -> None:
    env = TrajectoryMetaDriveEnv(_environment_config(map_sequence))
    adapter = NoTrafficMetaDriveObservationAdapter(stage0_runtime.planner_config, 100.0)
    generator = stage0_runtime.new_noise_generator()
    try:
        env.reset(seed=0)
        observation = adapter.build(env)
        _, _, planner_result = stage0_runtime.infer(observation, generator)
        prediction = planner_result.prediction
        ego_trajectory = prediction[0, 0].detach().cpu().numpy().astype(np.float32)
        _, _, terminated, truncated, info = env.step(ego_trajectory)

        assert prediction.shape == (1, 11, 80, 4)
        assert torch.isfinite(prediction).all()
        assert info["trajectory_execution_steps"] >= 1
        assert info["trajectory_substep_states"].shape[1] == 7
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
    finally:
        env.close()
