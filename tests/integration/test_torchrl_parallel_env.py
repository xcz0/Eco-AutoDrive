from __future__ import annotations

from contextlib import suppress

import numpy as np
import pytest

from eco_planner.envs import (
    PlannerObservationSpec,
    VectorEnvScenario,
    VectorMetaDriveEnv,
    VectorMetaDriveWorkerError,
)
from eco_planner.models.config import OfficialDiffusionPlannerConfig


@pytest.mark.simulator
def test_parallel_env_supports_partial_step_reset_map_replacement_and_audits(
    official_model_config: OfficialDiffusionPlannerConfig,
) -> None:
    first = VectorEnvScenario("first", "S", 0)
    replacement = VectorEnvScenario("replacement", "SC", 1)
    env = VectorMetaDriveEnv(
        (_environment_config("S"), _environment_config("S")),
        mode="no_traffic",
        observation_spec=PlannerObservationSpec.from_planner_config(official_model_config),
        map_query_radius_m=100.0,
        history_warmup_steps=0,
        scenarios=(first, replacement),
    )
    try:
        resets = env.reset((first, first))
        steps = env.step((_stationary_trajectory(), _stationary_trajectory()))
        reset_after_replacement = env.reset_at(0, replacement)
        partial = env.step_at(0, _stationary_trajectory())
    finally:
        env.close()
        env.close()

    assert [item.scenario for item in resets] == [first, first]
    assert all(item.route_length_m > 0.0 for item in resets)
    assert all(item.execution.substep_states.shape == (1, 7) for item in steps)
    assert reset_after_replacement.scenario == replacement
    assert partial.slot == 0
    assert partial.execution.substep_states.shape == (1, 7)


@pytest.mark.simulator
def test_parallel_env_propagates_worker_reset_failures(
    official_model_config: OfficialDiffusionPlannerConfig,
) -> None:
    valid = VectorEnvScenario("valid", "S", 0)
    invalid = VectorEnvScenario("invalid", "S", -1)
    env = VectorMetaDriveEnv(
        (_environment_config("S"), _environment_config("S")),
        mode="no_traffic",
        observation_spec=PlannerObservationSpec.from_planner_config(official_model_config),
        map_query_radius_m=100.0,
        history_warmup_steps=0,
        scenarios=(valid, invalid),
    )
    try:
        env.reset((valid, valid))
        with pytest.raises(VectorMetaDriveWorkerError) as exc_info:
            env.reset_at(0, invalid)
        message = str(exc_info.value)
        assert "slot 0" in message
        assert "reset" in message
        assert "Traceback" in message
    finally:
        with suppress(RuntimeError):
            env.close()


@pytest.mark.simulator
def test_parallel_env_keeps_traffic_warmup_history_and_execution_audit(
    official_model_config: OfficialDiffusionPlannerConfig,
) -> None:
    scenario = VectorEnvScenario("traffic", "SC", 0)
    env = VectorMetaDriveEnv(
        (_traffic_environment_config(),),
        mode="traffic",
        observation_spec=PlannerObservationSpec.from_planner_config(official_model_config),
        map_query_radius_m=100.0,
        history_warmup_steps=20,
        scenarios=(scenario,),
    )
    try:
        reset = env.reset((scenario,))[0]
        stepped = env.step((_stationary_trajectory(),))[0]
    finally:
        env.close()

    assert len(reset.warmup_executions) == 20
    assert len(stepped.execution.traffic_frames) == 1
    assert stepped.traffic_audit is not None


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


def _stationary_trajectory() -> np.ndarray:
    trajectory = np.zeros((80, 4), dtype=np.float32)
    trajectory[:, 2] = 1.0
    return trajectory
