from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

from eco_planner.envs import MetaDriveEnvSlot, PlannerObservationSpec, TorchRLMetaDriveEnv
from eco_planner.models.config import OfficialDiffusionPlannerConfig


@pytest.mark.simulator
def test_torchrl_adapter_executes_one_no_traffic_transition(
    official_model_config: OfficialDiffusionPlannerConfig,
) -> None:
    slot = MetaDriveEnvSlot(
        _environment_config(),
        mode="no_traffic",
        observation_spec=PlannerObservationSpec.from_planner_config(official_model_config),
        map_query_radius_m=100.0,
        history_warmup_steps=0,
    )
    env = TorchRLMetaDriveEnv(
        slot,
        map_name="S",
        seed=0,
        observation_spec=PlannerObservationSpec.from_planner_config(official_model_config),
    )
    try:
        reset = env.reset()
        transition = env.step(TensorDict({"action": _stationary_trajectory()}, batch_size=[]))
    finally:
        env.close()

    assert reset["ego_current_state"].shape == (10,)
    assert transition["next", "reward"].shape == (1,)
    assert transition["next", "done"].dtype == torch.bool
    assert transition["next", "terminated"].dtype == torch.bool
    assert transition["next", "truncated"].dtype == torch.bool


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
        "trajectory_execution_steps": 1,
        "programmatic_lane_speed_limit_kmh": 50.0,
    }


def _stationary_trajectory() -> torch.Tensor:
    trajectory = torch.zeros((80, 4), dtype=torch.float32)
    trajectory[:, 2] = 1.0
    return trajectory
