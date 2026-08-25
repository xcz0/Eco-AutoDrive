from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from eco_planner.models import OfficialDiffusionPlannerConfig, PlannerInferenceResult
from eco_planner.rl.policy import ExplorationPolicy, ExplorationPolicyContext
from eco_planner.rl.runtime import FabricRolloutRuntime
from eco_planner.runtime.fabric import InferenceRuntimeReport


def _assert_trajectory_contract(trajectory: np.ndarray) -> None:
    assert trajectory.shape[-2:] == (80, 4)
    assert trajectory.dtype == np.float32
    assert np.isfinite(trajectory).all()
    assert np.all(np.linalg.norm(trajectory[..., 2:4], axis=-1) > 0.0)


class _CpuFabric:
    device = torch.device("cpu")

    @staticmethod
    def to_device(value: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return value


class _Planner:
    def __init__(self, config: OfficialDiffusionPlannerConfig, hidden_dim: int) -> None:
        self.config = config
        self.hidden_dim = hidden_dim

    def prepare_policy_guidance(
        self,
        observation: dict[str, torch.Tensor],
        noise: torch.Tensor,
        transition_generators: tuple[torch.Generator, ...],
    ) -> SimpleNamespace:
        batch = observation["ego_current_state"].shape[0]
        context = ExplorationPolicyContext(
            scene_tokens=torch.ones((batch, 2, self.hidden_dim), dtype=torch.float32),
            scene_padding_mask=torch.zeros((batch, 2), dtype=torch.bool),
            navigation_tokens=torch.ones((batch, 1, self.hidden_dim), dtype=torch.float32),
            navigation_padding_mask=torch.zeros((batch, 1), dtype=torch.bool),
            reference_trajectory=torch.ones((batch, 80, 4), dtype=torch.float32),
        )
        return SimpleNamespace(policy_context=context, noise=noise)

    @staticmethod
    def complete_policy_guidance(
        prepared: SimpleNamespace, action: torch.Tensor
    ) -> PlannerInferenceResult:
        return PlannerInferenceResult(prediction=prepared.noise, guidance_action=action)


def _runtime(official_model_config, exploration_policy_config) -> FabricRolloutRuntime:
    return FabricRolloutRuntime(
        _CpuFabric(),
        _Planner(official_model_config, exploration_policy_config.hidden_dim),
        ExplorationPolicy(exploration_policy_config),
        InferenceRuntimeReport("cpu", "cpu", "32-true", "32-true", "cpu", 0, 1),
        noise_seed=0,
        policy_action_seed=0,
        checkpoint_report=SimpleNamespace(),
        sampler=SimpleNamespace(),
        guidance_config=SimpleNamespace(),
    )


@torch.no_grad()
def test_rollout_batch_runtime_matches_independent_serial_slots(
    official_model_config,
    baseline_observation: dict[str, torch.Tensor],
    exploration_policy_config,
) -> None:
    batch = 4
    observation = {
        name: value.repeat((batch,) + (1,) * (value.ndim - 1))
        for name, value in baseline_observation.items()
    }
    batch_runtime = _runtime(official_model_config, exploration_policy_config)
    serial_runtime = _runtime(official_model_config, exploration_policy_config)
    serial_runtime.policy.load_state_dict(batch_runtime.policy.state_dict())
    diffusion_seeds = tuple(100 + index for index in range(batch))
    policy_seeds = tuple(200 + index for index in range(batch))

    batched = batch_runtime.decide_batch(
        observation,
        tuple(torch.Generator().manual_seed(seed) for seed in diffusion_seeds),
        tuple(torch.Generator().manual_seed(seed) for seed in policy_seeds),
    )

    assert batched.ego_trajectories.shape == (batch, 80, 4)
    _assert_trajectory_contract(batched.ego_trajectories)
    for index in range(batch):
        single_observation = {name: value[index : index + 1] for name, value in observation.items()}
        serial = serial_runtime.decide(
            single_observation,
            torch.Generator().manual_seed(diffusion_seeds[index]),
            torch.Generator().manual_seed(policy_seeds[index]),
        )
        np.testing.assert_array_equal(batched.ego_trajectories[index], serial.ego_trajectory)
        torch.testing.assert_close(
            batched.training_decision["guidance_action"][index],
            serial.training_decision["guidance_action"][0],
        )
        torch.testing.assert_close(
            batched.training_decision["state_value"][index],
            serial.training_decision["state_value"][0],
        )
        torch.testing.assert_close(
            batched.training_decision["old_joint_guidance_log_prob"][index],
            serial.training_decision["old_joint_guidance_log_prob"][0],
        )


@torch.no_grad()
def test_rollout_bootstrap_batch_returns_one_value_per_slot(
    official_model_config,
    baseline_observation: dict[str, torch.Tensor],
    exploration_policy_config,
) -> None:
    batch = 2
    observation = {
        name: value.repeat((batch,) + (1,) * (value.ndim - 1))
        for name, value in baseline_observation.items()
    }
    runtime = _runtime(official_model_config, exploration_policy_config)

    value = runtime.bootstrap_value_batch(
        observation,
        tuple(torch.Generator().manual_seed(10 + index) for index in range(batch)),
    )

    assert value.shape == (batch,)
