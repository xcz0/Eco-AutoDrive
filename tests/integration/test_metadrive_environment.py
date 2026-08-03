from __future__ import annotations

import numpy as np
import pytest
import torch

from eco_planner.envs import (
    MetaDriveMapAdapter,
    NoTrafficMetaDriveObservationAdapter,
    TrajectoryMetaDriveEnv,
)
from eco_planner.models.config import OfficialDiffusionPlannerConfig
from eco_planner.models.pretrained import CheckpointLoadReport, PretrainedDiffusionPlanner


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
    }


def _straight_trajectory(speed_mps: float) -> np.ndarray:
    trajectory = np.zeros((80, 4), dtype=np.float32)
    trajectory[:, 0] = np.arange(1, 81, dtype=np.float32) * speed_mps * 0.1
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
        np.testing.assert_allclose(
            info["trajectory_substep_states"][-1, :2],
            np.asarray(env.agent.position),
        )
        assert forward_progress == pytest.approx(2.5, abs=0.35)
        assert float(env.agent.heading_theta) == pytest.approx(start_heading, abs=1e-3)
        assert float(env.agent.speed) == pytest.approx(5.0, abs=0.1)
        assert np.isfinite(reward)
        assert not terminated
        assert not truncated
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
        first = adapter.build(env, torch.device("cpu"))
        env.reset(seed=0)
        second = adapter.build(env, torch.device("cpu"))

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
@pytest.mark.slow
@pytest.mark.parametrize("map_sequence", ["S", "SC"])
def test_official_planner_executes_no_traffic_closed_loop_cycle(
    map_sequence: str,
    stage0_planner: tuple[PretrainedDiffusionPlanner, CheckpointLoadReport],
) -> None:
    planner, _ = stage0_planner
    env = TrajectoryMetaDriveEnv(_environment_config(map_sequence))
    adapter = NoTrafficMetaDriveObservationAdapter(planner.config, 100.0)
    generator = torch.Generator(device="cpu").manual_seed(0)
    try:
        env.reset(seed=0)
        observation = adapter.build(env, torch.device("cpu"))
        noise = torch.randn(
            (1, 11, 80, 4),
            dtype=torch.float32,
            generator=generator,
        )

        prediction = planner.predict(observation, noise)
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
