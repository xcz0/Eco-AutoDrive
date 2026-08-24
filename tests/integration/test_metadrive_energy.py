from __future__ import annotations

import numpy as np
import pytest

from eco_planner.envs import TrajectoryExecutionRecord, TrajectoryMetaDriveEnv


def _environment_config() -> dict[str, object]:
    return {
        "use_render": False,
        "map": "S",
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


@pytest.mark.simulator
def test_trajectory_environment_records_metadrive_energy() -> None:
    env = TrajectoryMetaDriveEnv(_environment_config())
    try:
        env.reset(seed=0)
        _, _, _, _, info = env.step(_straight_trajectory(5.0))
        execution = info["trajectory_execution"]

        assert isinstance(execution, TrajectoryExecutionRecord)
        assert execution.trajectory_energy_ml > 0.0
        assert execution.episode_energy_ml == pytest.approx(execution.trajectory_energy_ml)
        assert info["trajectory_energy_ml"] == pytest.approx(execution.trajectory_energy_ml)
        assert info["episode_energy_ml"] == pytest.approx(execution.episode_energy_ml)
    finally:
        env.close()
