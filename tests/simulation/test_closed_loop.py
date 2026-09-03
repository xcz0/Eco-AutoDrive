"""Real MetaDrive boundaries and the checkpoint-backed closed-loop smoke."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from pydantic import TypeAdapter

from eco_planner.envs import (
    MetaDriveEnvSlot,
    PlannerObservationSpec,
    VectorEnvScenario,
    VectorMetaDriveEnv,
    WorkerResetResult,
    WorkerStepResult,
    operation_results,
)
from eco_planner.envs.geometry import rear_axle_position, world_points_to_local
from eco_planner.envs.metadrive.simulator import MetaDriveBackend
from eco_planner.models.config import OfficialDiffusionPlannerConfig
from eco_planner.rl.config import parse_rollout_config
from eco_planner.rl.optimization import PPOConfig, PPOUpdater
from eco_planner.rl.reward import (
    MetaDriveBuiltinRewardConfig,
    PlannerRFTEnergyRewardConfig,
    RewardProfileConfig,
    create_reward_objective,
    score_plannerrft_energy_step,
)
from eco_planner.rl.rollout import collect_rollout_episode, create_fabric_rollout_runtime


@pytest.fixture(scope="module")
def baseline_checkpoint_dir() -> Path:
    checkpoint_dir = Path(__file__).resolve().parents[2] / "checkpoints" / "DP-Origin"
    required_assets = (checkpoint_dir / "args.json", checkpoint_dir / "model.pth")
    missing_assets = [str(path) for path in required_assets if not path.is_file()]
    if missing_assets:
        pytest.skip(f"checkpoint assets are unavailable: {', '.join(missing_assets)}")
    return checkpoint_dir


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


def _off_route_trajectory(env: MetaDriveBackend, query_radius_m: float) -> np.ndarray:
    route_roads = {
        (start, end)
        for start, end in zip(
            env.agent.navigation.checkpoints[:-1], env.agent.navigation.checkpoints[1:], strict=True
        )
    }
    route_lanes = [
        lane
        for lane in env.current_map.road_network.get_all_lanes()
        if lane.index[:2] in route_roads
    ]
    candidates: list[tuple[str, object, float]] = []
    for lane in env.current_map.road_network.get_all_lanes():
        if lane.index[:2] in route_roads:
            continue
        longitudinal = float(lane.length) / 2.0
        point = np.asarray(lane.position(longitudinal, 0.0), dtype=np.float64)
        if all(float(route_lane.distance(point)) > query_radius_m for route_lane in route_lanes):
            candidates.append((repr(lane.index), lane, longitudinal))
    if not candidates:
        raise RuntimeError("test map has no non-route lane outside the local route query")
    _, target_lane, longitudinal = min(candidates, key=lambda candidate: candidate[0])
    target_rear_axle = np.asarray(target_lane.position(longitudinal, 0.0), dtype=np.float64)
    target_heading = float(target_lane.heading_theta_at(longitudinal))
    anchor_rear_axle = rear_axle_position(
        np.asarray(env.agent.position, dtype=np.float64),
        float(env.agent.heading_theta),
        float(env.agent.REAR_WHEELBASE),
    )
    target_local = world_points_to_local(
        target_rear_axle[None], anchor_rear_axle, float(env.agent.heading_theta)
    )[0]
    heading_delta = target_heading - float(env.agent.heading_theta)
    trajectory = np.zeros((80, 4), dtype=np.float32)
    trajectory[:, :2] = target_local
    trajectory[:, 2] = np.cos(heading_delta)
    trajectory[:, 3] = np.sin(heading_delta)
    return trajectory


def _reward_profile(name: str) -> MetaDriveBuiltinRewardConfig | PlannerRFTEnergyRewardConfig:
    config_root = Path(__file__).resolve().parents[2] / "configs" / "components" / "reward"
    raw = OmegaConf.to_container(OmegaConf.load(config_root / f"{name}.yaml"), resolve=True)
    return TypeAdapter(RewardProfileConfig).validate_python(raw)


_BUILTIN_REWARD = _reward_profile("metadrive_builtin_v1")
_ENERGY_REWARD = _reward_profile("plannerrft_energy_v1")


def _ppo_config() -> PPOConfig:
    return PPOConfig(
        name="closed_loop_smoke",
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        target_kl=None,
        value_coefficient=0.5,
        entropy_coefficient=0.01,
        gradient_diagnostics=False,
        learning_rate=0.00025,
        adam_epsilon=1e-5,
        weight_decay=0.0,
        max_gradient_norm=0.5,
        epochs=1,
        batch_size=2,
        minibatch_size=2,
        minibatch_seed=7,
        scheduler_total_optimizer_steps=1,
        scheduler_minimum_learning_rate=0.0,
    )


@pytest.mark.simulator
def test_environment_slot_executes_evaluation_prefix_with_valid_audit(
    official_model_config: OfficialDiffusionPlannerConfig,
) -> None:
    with MetaDriveEnvSlot(
        _environment_config("S"),
        mode="no_traffic",
        observation_spec=PlannerObservationSpec.from_planner_config(official_model_config),
        map_query_radius_m=100.0,
        history_warmup_steps=0,
    ) as slot:
        slot.reset(map_name="S", seed=0)
        result = slot.step(_straight_trajectory())
        execution = result.execution

        assert execution.substep_states.shape == (5, 7)
        assert execution.substep_terminated.shape == (5,)
        assert execution.substep_truncated.shape == (5,)
        assert isinstance(result.terminated, bool)
        assert isinstance(result.truncated, bool)
        assert np.isfinite(result.reward)
        assert np.isfinite(execution.substep_states).all()
        assert np.isfinite(execution.substep_native_energy_ml).all()
        assert np.isfinite(execution.substep_executed_fuel_proxy_energy_ml).all()
        assert np.all(execution.substep_native_energy_ml >= 0.0)
        assert np.all(execution.substep_executed_fuel_proxy_energy_ml >= 0.0)
        np.testing.assert_allclose(
            execution.substep_states[-1, :2],
            np.asarray(slot.backend.agent.position),
            atol=1e-3,
        )


@pytest.mark.simulator
def test_same_scenario_reset_restores_spawn_after_trajectory_step(
    official_model_config: OfficialDiffusionPlannerConfig,
) -> None:
    config = _environment_config("S", trajectory_execution_steps=1)
    with MetaDriveEnvSlot(
        config,
        mode="no_traffic",
        observation_spec=PlannerObservationSpec.from_planner_config(official_model_config),
        map_query_radius_m=100.0,
        history_warmup_steps=0,
    ) as slot:
        initial = slot.reset(map_name="S", seed=0)
        slot.step(_straight_trajectory())

        repeated = slot.reset(map_name="S", seed=0)

    np.testing.assert_allclose(
        repeated.warmup_initial_state,
        initial.warmup_initial_state,
        atol=1e-6,
    )
    assert repeated.route_completion == pytest.approx(initial.route_completion, abs=1e-6)


@pytest.mark.simulator
def test_traffic_history_enters_planner_observation(
    official_model_config: OfficialDiffusionPlannerConfig,
) -> None:
    config = _environment_config("SC", traffic_density=0.1)
    config["horizon"] = 30
    with MetaDriveEnvSlot(
        config,
        mode="traffic",
        observation_spec=PlannerObservationSpec.from_planner_config(official_model_config),
        map_query_radius_m=100.0,
        history_warmup_steps=20,
    ) as slot:
        slot.reset(map_name="SC", seed=0)
        warmup = tuple(slot.warmup())
        result = slot.observe()
    observation = result.observation
    audit = result.traffic_audit

    assert sum(item.substep_states.shape[0] for item in warmup) == 20
    assert observation["neighbor_agents_past"].shape == (32, 21, 11)
    assert observation.batch_size == torch.Size([])
    assert observation["static_objects"].shape == (5, 10)
    assert observation["lanes"].shape == (70, 20, 12)
    assert observation["route_lanes"].shape == (25, 20, 12)
    assert torch.count_nonzero(observation["neighbor_agents_past"]).item() > 0
    assert audit is not None
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
        reset_results = operation_results(resets, WorkerResetResult)
        step_results = operation_results(steps, WorkerStepResult)

    assert resets.batch_size == torch.Size([2])
    assert steps.batch_size == torch.Size([2])
    for index, (_reset, step) in enumerate(zip(reset_results, step_results, strict=True)):
        reset_observation = resets["observation"][index]
        step_observation = steps["observation"][index]
        assert reset_observation.batch_size == torch.Size([])
        assert step_observation.batch_size == torch.Size([])
        assert reset_observation["ego_current_state"].shape == (10,)
        assert step.execution.substep_states.shape == (1, 7)
        assert steps["terminated"][index].dtype is torch.bool
        assert steps["truncated"][index].dtype is torch.bool
        assert torch.isfinite(steps["reward"][index]).all()
        assert torch.count_nonzero(reset_observation["route_lanes"]).item() > 0
        assert torch.count_nonzero(step_observation["route_lanes"]).item() > 0


@pytest.mark.simulator
def test_vector_partial_step_preserves_unselected_slot_and_requested_order(
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
        reset_results = operation_results(resets, WorkerResetResult)
        slot_one = envs.step((_straight_trajectory(),), slots=(1,))
        preserved = slot_one.clone()
        slot_zero = envs.step((_straight_trajectory(),), slots=(0,))

    slot_one_result = operation_results(slot_one, WorkerStepResult)[0]
    slot_zero_result = operation_results(slot_zero, WorkerStepResult)[0]
    np.testing.assert_allclose(
        slot_one_result.execution.start_center,
        reset_results[1].initial_state[:2],
    )
    np.testing.assert_allclose(
        slot_zero_result.execution.start_center,
        reset_results[0].initial_state[:2],
    )
    torch.testing.assert_close(slot_one["observation"], preserved["observation"])
    assert slot_one.batch_size == torch.Size([1])
    assert slot_zero.batch_size == torch.Size([1])


@pytest.mark.simulator
@pytest.mark.parametrize("reward_profile", (_BUILTIN_REWARD, _ENERGY_REWARD))
def test_off_route_lane_is_terminal_with_padded_route_observation(
    official_model_config: OfficialDiffusionPlannerConfig,
    reward_profile: MetaDriveBuiltinRewardConfig | PlannerRFTEnergyRewardConfig,
) -> None:
    query_radius_m = 5.0
    config = _environment_config("SXS", trajectory_execution_steps=1)
    scenario = VectorEnvScenario(name="off-route", map="SXS", seed=0)
    with MetaDriveEnvSlot(
        config,
        mode="no_traffic",
        observation_spec=PlannerObservationSpec.from_planner_config(official_model_config),
        map_query_radius_m=query_radius_m,
        history_warmup_steps=0,
    ) as source:
        source.reset(map_name=scenario.map, seed=scenario.seed)
        trajectory = _off_route_trajectory(source.backend, query_radius_m)

    with VectorMetaDriveEnv(
        [config],
        mode="no_traffic",
        observation_spec=PlannerObservationSpec.from_planner_config(official_model_config),
        map_query_radius_m=query_radius_m,
        history_warmup_steps=0,
        scenarios=(scenario,),
        builtin_reward_config=(
            reward_profile if isinstance(reward_profile, MetaDriveBuiltinRewardConfig) else None
        ),
        reward_objective=create_reward_objective(reward_profile),
    ) as envs:
        envs.reset((scenario,))
        step_batch = envs.step((trajectory,))
        step = operation_results(step_batch, WorkerStepResult)[0]
        observation = step_batch["observation"][0]

    assert bool(step_batch["terminated"][0].item()) is True
    assert bool(step_batch["truncated"][0].item()) is False
    assert step.execution.out_of_road is True
    assert step.execution.substep_terminated.tolist() == [True]
    assert step.execution.substep_truncated.tolist() == [False]
    assert torch.isfinite(step_batch["reward"][0]).all()
    assert all(
        torch.isfinite(value).all() for value in observation.values() if value.is_floating_point()
    )
    assert torch.count_nonzero(observation["route_lanes"]).item() == 0
    assert torch.count_nonzero(observation["route_lanes_speed_limit"]).item() == 0
    assert not torch.any(observation["route_lanes_has_speed_limit"])
    assert len(step.execution.substep_metrics) == 1
    if isinstance(reward_profile, PlannerRFTEnergyRewardConfig):
        assert (
            score_plannerrft_energy_step(
                reward_profile, step.execution.substep_metrics[-1]
            ).reward_gate
            == 0.0
        )


@pytest.mark.simulator
@pytest.mark.slow
def test_real_checkpoint_metadrive_rollout_updates_policy_without_changing_planner(
    baseline_checkpoint_dir: Path,
) -> None:
    config_dir = Path(__file__).resolve().parents[2] / "configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        config = compose(config_name="jobs/training/rollout_smoke")
    parsed = parse_rollout_config(config)
    runtime = create_fabric_rollout_runtime(
        parsed.runtime,
        parsed.sampler,
        parsed.guidance,
        parsed.policy,
        baseline_checkpoint_dir / "args.json",
        baseline_checkpoint_dir / "model.pth",
        parsed.rollout.policy_action_seed,
        planner_compile_mode="eager",
    )

    episode = collect_rollout_episode(
        parsed.scenario,
        runtime,
        parsed.env,
        mode=parsed.rollout.mode,
        map_query_radius_m=parsed.map_query_radius_m,
        history_warmup_steps=parsed.rollout.history_warmup_steps,
        max_transitions=2,
        stopped_speed_threshold_mps=parsed.rollout.stopped_speed_threshold_mps,
    )
    planner_hash = runtime.frozen_planner_hash()
    policy_before = {
        name: parameter.detach().clone() for name, parameter in runtime.policy.state_dict().items()
    }

    update = PPOUpdater(runtime.policy, _ppo_config()).update((episode,))

    assert runtime.checkpoint_report.ema_tensor_count > 0
    assert runtime.checkpoint_report.parameter_count > 0
    assert episode.transition_count == 2
    assert episode.training["guidance_action"].shape == (2, 2)
    assert episode.audit["initial_noise"].shape == (2, 11, 80, 4)
    assert episode.tail_kind == "rollout_limit"
    assert episode.training["next", "reward"].shape == (2, 1)
    assert update.optimizer_step_count == 1
    assert any(
        not torch.equal(parameter, policy_before[name])
        for name, parameter in runtime.policy.state_dict().items()
    )
    assert runtime.frozen_planner_hash() == planner_hash
