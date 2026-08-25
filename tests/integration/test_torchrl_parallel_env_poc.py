from __future__ import annotations

from contextlib import suppress

import pytest
import torch
from tensordict import TensorDict

from eco_planner.envs import (
    PlannerObservationSpec,
    TorchRLParallelScenario,
    create_torchrl_parallel_env_poc,
)
from eco_planner.models.config import OfficialDiffusionPlannerConfig


@pytest.mark.simulator
def test_parallel_env_poc_supports_partial_step_reset_map_replacement_and_audits(
    official_model_config: OfficialDiffusionPlannerConfig,
) -> None:
    env = create_torchrl_parallel_env_poc(
        2,
        _environment_config("S"),
        mode="no_traffic",
        observation_spec=PlannerObservationSpec.from_planner_config(official_model_config),
        map_query_radius_m=100.0,
        history_warmup_steps=0,
        scenarios=(
            TorchRLParallelScenario("S", 0),
            TorchRLParallelScenario("SC", 1),
        ),
    )
    try:
        reset = env.reset(_reset_tensordict((0, 0)))
        stepped = env.step(_step_tensordict(2, (True, True)))
        executions = env.execution_record()
        reset_after_replacement = env.reset(_reset_tensordict((1, 0), (True, False)))
        active_scenarios = env.current_scenario_index()
        partial = env.step(_step_tensordict(2, (True, False)))
    finally:
        env.close()

    assert reset["ego_current_state"].shape == (2, 10)
    assert stepped["next", "reward"].shape == (2, 1)
    assert len(executions) == 2
    assert all(item.substep_states.shape == (1, 7) for item in executions)
    assert active_scenarios == [1, 0]
    assert reset_after_replacement["ego_current_state"].shape == (2, 10)
    assert partial["next", "reward"].shape == (2, 1)


@pytest.mark.simulator
def test_parallel_env_poc_propagates_worker_reset_failures(
    official_model_config: OfficialDiffusionPlannerConfig,
) -> None:
    env = create_torchrl_parallel_env_poc(
        2,
        _environment_config("S"),
        mode="no_traffic",
        observation_spec=PlannerObservationSpec.from_planner_config(official_model_config),
        map_query_radius_m=100.0,
        history_warmup_steps=0,
        scenarios=(TorchRLParallelScenario("S", 0),),
    )
    try:
        env.BATCHED_PIPE_TIMEOUT = 1.0
        env.reset(_reset_tensordict((0, 0)))
        with pytest.raises(RuntimeError, match="worker 0 dead|time limit"):
            env.reset(_reset_tensordict((1, 0), (True, False)))
    finally:
        with suppress(RuntimeError):
            env.close()


@pytest.mark.simulator
def test_parallel_env_poc_keeps_traffic_warmup_history_and_execution_audit(
    official_model_config: OfficialDiffusionPlannerConfig,
) -> None:
    env = create_torchrl_parallel_env_poc(
        1,
        _traffic_environment_config(),
        mode="traffic",
        observation_spec=PlannerObservationSpec.from_planner_config(official_model_config),
        map_query_radius_m=100.0,
        history_warmup_steps=20,
        scenarios=(TorchRLParallelScenario("SC", 0),),
    )
    try:
        reset = env.reset(_reset_tensordict((0,)))
        warmup = env.warmup_execution_records()
        stepped = env.step(_step_tensordict(1, (True,)))
        execution = env.execution_record()
        traffic_audit = env.traffic_observation_audit()
    finally:
        env.close()

    assert reset["ego_current_state"].shape == (1, 10)
    assert len(warmup) == 1
    assert len(warmup[0]) == 20
    assert stepped["next", "reward"].shape == (1, 1)
    assert len(execution) == 1
    assert len(execution[0].traffic_frames) == 1
    assert len(traffic_audit) == 1
    assert traffic_audit[0] is not None


def _environment_config(map_name: str) -> dict[str, object]:
    return {
        "use_render": False,
        "map": map_name,
        "num_scenarios": 4,
        "traffic_density": 0.0,
        "random_traffic": False,
        "random_spawn_lane_index": False,
        "physics_world_step_size": 0.02,
        "decision_repeat": 5,
        "trajectory_horizon": 80,
        "trajectory_execution_steps": 1,
        "programmatic_lane_speed_limit_kmh": 50.0,
    }


def _traffic_environment_config() -> dict[str, object]:
    config = _environment_config("SC")
    config.update({"traffic_density": 0.05, "traffic_mode": "trigger", "horizon": 30})
    return config


def _reset_tensordict(
    scenarios: tuple[int, ...], reset_mask: tuple[bool, ...] | None = None
) -> TensorDict:
    size = len(scenarios)
    values: dict[str, torch.Tensor] = {
        "scenario_index": torch.tensor(scenarios, dtype=torch.int64),
    }
    if reset_mask is not None:
        values["_reset"] = torch.tensor(reset_mask, dtype=torch.bool)
    return TensorDict(values, batch_size=[size])


def _step_tensordict(size: int, step_mask: tuple[bool, ...]) -> TensorDict:
    action = torch.zeros((size, 80, 4), dtype=torch.float32)
    action[..., 2] = 1.0
    return TensorDict(
        {"action": action, "_step": torch.tensor(step_mask, dtype=torch.bool)}, batch_size=[size]
    )
