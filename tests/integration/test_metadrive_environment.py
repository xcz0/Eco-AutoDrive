from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from metadrive.utils import merge_dicts

from eco_planner.envs import (
    TrajectoryMetaDriveEnv,
    collate_observations,
)
from eco_planner.envs.map_adapter import MetaDriveMapAdapter
from eco_planner.envs.observation_adapter import (
    MetaDriveObservationAdapter,
    NoTrafficMetaDriveObservationAdapter,
)
from eco_planner.evaluation.config import RuntimeConfig, parse_evaluation_config
from eco_planner.evaluation.runtime import (
    FabricInferenceRuntime,
    create_fabric_inference_runtime,
)
from eco_planner.models import Dpm10SamplerConfig, NoGuidanceConfig
from eco_planner.models.config import OfficialDiffusionPlannerConfig


class _LegacyPhysicsTrajectoryMetaDriveEnv(TrajectoryMetaDriveEnv):
    def _step_planner_simulator(self, actions: dict[str, object]) -> dict[str, object]:
        before_info = self.engine.before_step(actions)
        for _ in range(self.config["decision_repeat"]):
            for name, manager in self.engine.managers.items():
                if name != "record_manager":
                    manager.step()
            self.engine.step_physics_world()
        after_info = self.engine.after_step()
        return merge_dicts(
            after_info,
            before_info,
            allow_new_keys=True,
            without_copy=True,
        )


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
        reset_observation, _ = env.reset(seed=0)
        start_position = np.asarray(env.agent.position, dtype=np.float64)
        start_heading = float(env.agent.heading_theta)

        _, reward, terminated, truncated, info = env.step(_straight_trajectory(5.0))
        execution = info["trajectory_execution"]

        np.testing.assert_array_equal(reset_observation, np.zeros(1, dtype=np.float32))
        assert env.engine.sensors == {}

        displacement = np.asarray(env.agent.position, dtype=np.float64) - start_position
        forward_progress = float(
            displacement @ np.array([np.cos(start_heading), np.sin(start_heading)])
        )
        assert env.action_space.shape == (80, 4)
        assert env.engine.episode_step == 5
        assert info["trajectory_execution_steps"] == 5
        assert info["trajectory_reward_sum"] == pytest.approx(reward)
        assert info["trajectory_execution"] is execution
        assert "trajectory_substep_states" not in info
        assert execution.substep_states.shape == (5, 7)
        assert execution.substep_energy_ml.shape == (5,)
        assert execution.substep_episode_energy_ml.shape == (5,)
        assert np.isfinite(execution.substep_energy_ml).all()
        assert np.isfinite(execution.substep_episode_energy_ml).all()
        assert np.all(execution.substep_energy_ml >= 0.0)
        assert np.all(np.diff(execution.substep_episode_energy_ml) >= 0.0)
        assert execution.substep_episode_energy_ml[-1] == pytest.approx(info["episode_energy"])
        assert execution.route_completion == pytest.approx(info["route_completion"])
        np.testing.assert_allclose(
            execution.substep_states[-1, :2],
            np.asarray(env.agent.position),
        )
        np.testing.assert_allclose(
            execution.substep_states[:, :2],
            execution.target_centers,
            atol=1e-3,
        )
        np.testing.assert_allclose(
            execution.substep_states[:, 2],
            execution.target_headings,
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
def test_trajectory_environment_executes_one_rollout_substep() -> None:
    config = _environment_config("S")
    config["trajectory_execution_steps"] = 1
    env = TrajectoryMetaDriveEnv(config)
    try:
        env.reset(seed=0)
        _, _, terminated, truncated, info = env.step(_straight_trajectory(5.0))
        execution = info["trajectory_execution"]

        assert env.engine.episode_step == 1
        assert info["trajectory_execution_steps"] == 1
        assert execution.substep_states.shape == (1, 7)
        assert not terminated
        assert not truncated
    finally:
        env.close()


@pytest.mark.simulator
def test_trajectory_environment_rejects_non_planner_observations() -> None:
    image_config = _environment_config("S")
    image_config["image_observation"] = True
    with pytest.raises(ValueError, match="image_observation"):
        TrajectoryMetaDriveEnv(image_config)

    agent_config = _environment_config("S")
    agent_config["agent_observation"] = object
    with pytest.raises(ValueError, match="agent_observation"):
        TrajectoryMetaDriveEnv(agent_config)


@pytest.mark.simulator
def test_trajectory_environment_matches_variable_heading_waypoints() -> None:
    env = TrajectoryMetaDriveEnv(_environment_config("S"))
    try:
        env.reset(seed=0)
        _, _, _, _, info = env.step(_turning_trajectory())

        execution = info["trajectory_execution"]
        np.testing.assert_allclose(
            execution.substep_states[:, :2], execution.target_centers, atol=1e-3
        )
        np.testing.assert_allclose(
            execution.substep_states[:, 2],
            execution.target_headings,
            atol=1e-4,
        )
    finally:
        env.close()


@pytest.mark.simulator
def test_trajectory_environment_stops_prefix_at_episode_horizon() -> None:
    config = _environment_config("S")
    config["horizon"] = 2
    env = TrajectoryMetaDriveEnv(config)
    try:
        env.reset(seed=0)
        _, _, terminated, truncated, info = env.step(_straight_trajectory(5.0))
        execution = info["trajectory_execution"]

        assert not terminated
        assert truncated
        assert execution.substep_states.shape == (2, 7)
        assert tuple(frame.simulator_step for frame in execution.traffic_frames) == (1, 2)
        assert execution.substep_truncated.tolist() == [False, True]
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
        assert set(env.engine.sensors) == {"lidar"}
        start_position = np.asarray(env.agent.position, dtype=np.float64).copy()

        _, _, terminated, truncated, info = env.step(_stationary_trajectory())

        frames = info["trajectory_execution"].traffic_frames
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
        adapter.reset(env, env.initial_traffic_frame)
        for _ in range(4):
            _, _, terminated, truncated, info = env.step(_stationary_trajectory())
            assert not terminated
            assert not truncated
            adapter.append_frames(info["trajectory_execution"].traffic_frames)

        observation, audit = adapter.build(env)

        assert observation["neighbor_agents_past"].shape == (32, 21, 11)
        assert torch.count_nonzero(observation["neighbor_agents_past"]).item() > 0
        assert audit.participant_count_in_radius > 0
        np.testing.assert_allclose(env.agent.position, start_position, atol=1e-3)
    finally:
        env.close()


def _traffic_observation_sequence(
    env_class: type[TrajectoryMetaDriveEnv],
    model_config: OfficialDiffusionPlannerConfig,
) -> list[tuple[dict[str, torch.Tensor], object, np.ndarray, float]]:
    config = _environment_config("SCSCSCSCSC")
    config["traffic_density"] = 0.05
    config["horizon"] = 100
    env = env_class(config)
    adapter = MetaDriveObservationAdapter(model_config, 100.0)
    records: list[tuple[dict[str, torch.Tensor], object, np.ndarray, float]] = []
    try:
        env.reset(seed=0)
        adapter.reset(env, env.initial_traffic_frame)
        for _ in range(4):
            _, _, _, _, info = env.step(_stationary_trajectory())
            adapter.append_frames(info["trajectory_execution"].traffic_frames)
        for _ in range(8):
            observation, audit = adapter.build(env)
            _, _, terminated, truncated, info = env.step(_straight_trajectory(5.0))
            assert not terminated
            assert not truncated
            execution = info["trajectory_execution"]
            adapter.append_frames(execution.traffic_frames)
            records.append(
                (
                    observation,
                    audit,
                    execution.substep_states.copy(),
                    execution.route_completion,
                )
            )
        return records
    finally:
        env.close()


@pytest.mark.simulator
def test_batched_bullet_substeps_match_legacy_planner_boundary(
    official_model_config: OfficialDiffusionPlannerConfig,
) -> None:
    legacy = _traffic_observation_sequence(
        _LegacyPhysicsTrajectoryMetaDriveEnv, official_model_config
    )
    optimized = _traffic_observation_sequence(TrajectoryMetaDriveEnv, official_model_config)

    for legacy_cycle, optimized_cycle in zip(legacy, optimized, strict=True):
        legacy_observation, legacy_audit, legacy_states, legacy_route = legacy_cycle
        observation, audit, states, route = optimized_cycle
        for name in legacy_observation:
            torch.testing.assert_close(
                observation[name], legacy_observation[name], rtol=0.0, atol=1e-5
            )
        assert audit.selected_participant_ids == legacy_audit.selected_participant_ids
        assert audit.participant_count_in_radius == legacy_audit.participant_count_in_radius
        assert audit.static_object_count_in_radius == legacy_audit.static_object_count_in_radius
        assert audit.nearest_participant_distance_m == pytest.approx(
            legacy_audit.nearest_participant_distance_m, abs=1e-3
        )
        np.testing.assert_allclose(states, legacy_states, rtol=0.0, atol=1e-12)
        assert route == pytest.approx(legacy_route, abs=1e-12)


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
def test_programmatic_lane_speed_limit_profile_follows_generated_blocks() -> None:
    config = _environment_config("SSS")
    config["programmatic_lane_speed_limit_profile_kmh"] = [50.0, 30.0, 50.0]
    env = TrajectoryMetaDriveEnv(config)
    try:
        env.reset(seed=0)

        audit = env.programmatic_lane_speed_limit_audit
        lane_limits = [
            float(lane.speed_limit) for lane in env.current_map.road_network.get_all_lanes()
        ]

        assert audit["block_speed_limit_profile_kmh"] == (50.0, 30.0, 50.0)
        assert audit["block_speed_limit_profile_applied_lane_count"] > 0
        assert 30.0 in lane_limits
        assert 50.0 in lane_limits
    finally:
        env.close()


@pytest.mark.simulator
def test_energy_traffic_scenario_satisfies_route_length_contract() -> None:
    config_dir = Path(__file__).resolve().parents[2] / "configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        config = compose(config_name="jobs/evaluation/energy_traffic")
    parsed = parse_evaluation_config(config)
    scenario = parsed.scenarios[0]
    env_config = dict(parsed.env)
    env_config["map"] = scenario.map
    env = TrajectoryMetaDriveEnv(env_config)
    try:
        env.reset(seed=scenario.seed)

        assert 2_000.0 <= env.route_length_m <= 5_000.0
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
        first = adapter.build_arrays(env)
        env.reset(seed=0)
        second = adapter.build_arrays(env)

        assert first["lanes"].shape == (70, 20, 12)
        assert first["route_lanes"].shape == (25, 20, 12)
        assert np.count_nonzero(first["route_lanes"]) > 0
        for name in first:
            if first[name].dtype == np.float32:
                assert np.isfinite(first[name]).all()
            np.testing.assert_array_equal(first[name], second[name])
    finally:
        env.close()


@pytest.mark.simulator
def test_indexed_map_adapter_matches_full_lane_scan(
    official_model_config: OfficialDiffusionPlannerConfig,
) -> None:
    env = TrajectoryMetaDriveEnv(_environment_config("SCSCSCSCSC"))
    indexed = MetaDriveMapAdapter(official_model_config, query_radius_m=100.0)
    reference = MetaDriveMapAdapter(official_model_config, query_radius_m=100.0)
    try:
        env.reset(seed=0)
        indexed.reset(env)
        reference.reset(env)
        reference._candidate_snapshots = lambda _: reference._lane_snapshots  # type: ignore[method-assign]

        for _ in range(3):
            indexed_result = indexed.build_arrays(env)
            reference_result = reference.build_arrays(env)
            for name in indexed_result:
                np.testing.assert_array_equal(indexed_result[name], reference_result[name])
            env.step(_straight_trajectory(5.0))
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
        observation = adapter.build_arrays(env)
        valid = observation["lanes_has_speed_limit"]
        observed_speed_limits = observation["lanes_speed_limit"][valid]

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
    baseline_runtime: FabricInferenceRuntime,
) -> None:
    env = TrajectoryMetaDriveEnv(_environment_config(map_sequence))
    adapter = NoTrafficMetaDriveObservationAdapter(baseline_runtime.planner_config, 100.0)
    generator = baseline_runtime.new_noise_generator()
    try:
        env.reset(seed=0)
        observation = adapter.build(env)
        planner_result = baseline_runtime.infer(collate_observations([observation]), generator)
        ego_trajectory = planner_result.ego_trajectory
        prediction = planner_result.audit_result()["prediction"]
        _, _, terminated, truncated, info = env.step(ego_trajectory)

        assert prediction.shape == (1, 11, 80, 4)
        assert torch.isfinite(prediction).all()
        assert info["trajectory_execution_steps"] >= 1
        assert info["trajectory_execution"].substep_states.shape[1] == 7
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
    finally:
        env.close()


@pytest.mark.gpu
@pytest.mark.simulator
@pytest.mark.slow
def test_cuda_bf16_completes_traffic_warmup_and_first_inference(
    baseline_checkpoint_dir,
) -> None:
    runtime = create_fabric_inference_runtime(
        RuntimeConfig(accelerator="cuda", precision="bf16-mixed", seed=0),
        Dpm10SamplerConfig(),
        NoGuidanceConfig(),
        baseline_checkpoint_dir / "args.json",
        baseline_checkpoint_dir / "model.pth",
    )
    env_config = _environment_config("SSSS")
    env_config.update({"traffic_density": 0.05, "traffic_mode": "trigger"})
    env = TrajectoryMetaDriveEnv(env_config)
    adapter = MetaDriveObservationAdapter(runtime.planner_config, 100.0)
    try:
        env.reset(seed=0)
        adapter.reset(env, env.initial_traffic_frame)
        for _ in range(4):
            _, _, terminated, truncated, info = env.step(_stationary_trajectory())
            assert not terminated and not truncated
            adapter.append_frames(info["trajectory_execution"].traffic_frames)
        observation, _ = adapter.build(env)

        result = runtime.infer(collate_observations([observation]), runtime.new_noise_generator())

        assert all(
            value.dtype in {torch.float32, torch.bool} and value.device.type == "cpu"
            for value in observation.values()
        )
        audit = result.audit_result()
        assert audit["initial_noise"].dtype == torch.float32
        prediction = audit["prediction"]
        assert prediction.dtype == torch.float32
        assert prediction.shape == (1, 11, 80, 4)
    finally:
        env.close()
