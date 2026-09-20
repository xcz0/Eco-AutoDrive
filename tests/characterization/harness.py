"""Deterministic synthetic harness for the learned-guidance decision characterization tests.

The harness pins every input that defines one learned-guidance decision: a fixed
observation, a fixed frozen planner, a fixed exploration policy, and explicitly seeded
diffusion/policy generators.  Task A refactors the planning ownership around this path,
so these tests must fail if the decision tensors or RNG consumption move.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch
from lightning.fabric import Fabric
from tensordict import TensorDict

from eco_planner.models import (
    CheckpointLoadReport,
    Ddim5SamplerConfig,
    OrthogonalPolicyGuidanceConfig,
    PretrainedDiffusionPlanner,
    sampler_report,
)
from eco_planner.rl.policy import ExplorationPolicy, ExplorationPolicyConfig
from eco_planner.rl.rollout import FabricRolloutRuntime
from eco_planner.rl.rollout.decision import BatchRolloutDecision
from eco_planner.runtime.fabric import InferenceRuntimeReport

DIFFUSION_SEED = 101
POLICY_SEED = 202
POLICY_HIDDEN_DIM = 2
FIXTURE_PATH = Path(__file__).with_name("fixtures") / "learned_guidance_decision.pt"

AUDIT_FIELDS = (
    "prediction",
    "initial_noise",
    "reference_prediction",
    "lateral_target_offset_m",
    "longitudinal_target_speed_fraction",
    "longitudinal_target_speed_delta_mps",
    "lateral_objective_delta",
    "longitudinal_objective_delta",
    "applied_gradient_l2",
    "applied_gradient_max_abs",
    "raw_neighbor_gradient_l2",
    "zero_speed_count",
    "scene_tokens",
    "scene_padding_mask",
    "navigation_tokens",
    "navigation_padding_mask",
    "reference_trajectory",
    "base_action",
    "guidance_action",
    "old_joint_guidance_log_prob",
    "state_value",
    "beta_alpha",
    "beta_beta",
)
RNG_FIELDS = (
    "diffusion_rng_state_before",
    "policy_rng_state_before",
    "diffusion_rng_state_after",
    "policy_rng_state_after",
)


class _IdentityStateNormalizer:
    def inverse(self, value: torch.Tensor) -> torch.Tensor:
        return value


class _PolicyFeatureDenoiser(torch.nn.Module):
    """Deterministic stand-in for the diffusion backbone that exposes neutral features."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.anchor = torch.nn.Parameter(torch.tensor(1.25))

    def encode(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.zeros((inputs["ego_current_state"].shape[0], 1, 1))

    def encode_route(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.zeros((inputs["ego_current_state"].shape[0], 1))

    def encode_policy_features(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        batch = inputs["ego_current_state"].shape[0]
        ego = inputs["ego_current_state"]
        hidden = self.hidden_dim
        return {
            "scene_tokens": ego[:, : 2 * hidden].reshape(batch, 2, hidden),
            "scene_padding_mask": torch.zeros((batch, 2), dtype=torch.bool),
            "navigation_tokens": ego[:, 2 * hidden : 4 * hidden].reshape(batch, 2, hidden),
            "navigation_padding_mask": torch.zeros((batch, 2), dtype=torch.bool),
            "route_encoding": torch.zeros((batch, 1)),
        }

    def denoise(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        encoding: torch.Tensor,
        route_encoding: torch.Tensor,
        current_mask: torch.Tensor,
    ) -> torch.Tensor:
        scale = self.anchor + 0.05 * encoding.reshape(encoding.shape[0], -1).mean(dim=1)
        scaled = sample * scale.reshape(scale.shape[0], 1, 1)
        view = scaled.reshape(scaled.shape[0], scaled.shape[1], -1, 4).clone()
        view[..., 2] = view[..., 2] + 1.0
        return view.reshape(scaled.shape)


def planner_config() -> SimpleNamespace:
    return SimpleNamespace(
        predicted_neighbor_num=2,
        future_len=80,
        observation_normalizer=lambda observation: dict(observation),
        state_normalizer=_IdentityStateNormalizer(),
    )


def sampler_config() -> Ddim5SamplerConfig:
    return Ddim5SamplerConfig(
        name="ddim5",
        num_steps=5,
        timesteps=(1.0, 0.8, 0.6, 0.4, 0.2, 0.0),
        initial_noise_scale=0.5,
        ddim_stochasticity=0.5,
        parity_label="project_noise_scale_0_5",
    )


def guidance_config() -> OrthogonalPolicyGuidanceConfig:
    return OrthogonalPolicyGuidanceConfig(
        name="orthogonal_policy",
        formula_label="centered_energy_gradient_delta_v1",
        lateral_max_offset_m=2.5,
        longitudinal_max_speed_fraction=0.25,
        trajectory_dt_s=0.1,
        gradient_step_coefficient=1.0,
        reference_refresh_cycles=1,
        share_scene_encoding=True,
        share_initial_noise=True,
        share_transition_noise=True,
        heading_norm_epsilon=1e-6,
        zero_speed_tolerance_mps=1e-6,
    )


def policy_config() -> ExplorationPolicyConfig:
    return ExplorationPolicyConfig(
        hidden_dim=POLICY_HIDDEN_DIM,
        reference_mixer_depth=1,
        reference_token_mlp_hidden_dim=8,
        reference_channel_mlp_hidden_dim=8,
        cross_attention_heads=1,
        cross_attention_dropout=0.0,
        fusion_mlp_depth=1,
        fusion_hidden_dim=8,
        initial_concentration=2.0,
        minimum_concentration=1e-4,
    )


def build_policy() -> ExplorationPolicy:
    """Build one deterministic, non-degenerate exploration policy without global RNG."""

    policy = ExplorationPolicy(policy_config())
    with torch.no_grad():
        for index, parameter in enumerate(policy.parameters()):
            values = torch.arange(parameter.numel(), dtype=parameter.dtype).reshape(parameter.shape)
            parameter.copy_(0.02 * torch.sin(values + float(index + 1)))
    policy.eval()
    return policy


def build_observation() -> TensorDict:
    ego = torch.zeros((1, 10), dtype=torch.float32)
    ego[0] = torch.tensor([0.0, 0.0, 1.0, 0.0, 10.0, 2.0, 0.0, -1.0, 0.0, 5.0])
    neighbors = torch.zeros((1, 4, 21, 11), dtype=torch.float32)
    timesteps = torch.arange(21, dtype=torch.float32)
    for index in range(2):
        neighbor = neighbors[0, index]
        neighbor[:, 0] = 12.0 + index * 4.0 + timesteps
        neighbor[:, 2] = 1.0
        neighbor[:, 4] = 10.0
        neighbor[:, 6] = 1.8
        neighbor[:, 7] = 4.8
        neighbor[:, 8] = 1.0
    return TensorDict(
        {"ego_current_state": ego, "neighbor_agents_past": neighbors},
        batch_size=[1],
    )


def build_runtime() -> FabricRolloutRuntime:
    fabric = Fabric(accelerator="cpu", devices=1, precision="32-true")
    planner = PretrainedDiffusionPlanner(  # type: ignore[arg-type]
        planner_config(),
        _PolicyFeatureDenoiser(POLICY_HIDDEN_DIM),
        sampler_config(),
        guidance_config(),
    )
    report = InferenceRuntimeReport(
        requested_accelerator="cpu",
        resolved_accelerator="cpu",
        requested_precision="32-true",
        resolved_precision="32-true",
        device="cpu",
        seed=0,
        world_size=1,
    )
    return FabricRolloutRuntime(
        fabric,
        planner,
        build_policy(),
        report,
        noise_seed=DIFFUSION_SEED,
        policy_action_seed=POLICY_SEED,
        checkpoint_report=CheckpointLoadReport(ema_tensor_count=1, parameter_count=1),
        sampler=sampler_report(sampler_config()),
        guidance_config=guidance_config(),
        planner_compile_mode="eager",
    )


class RngSnapshot:
    """Generator states captured immediately before and after one decision."""

    def __init__(
        self,
        diffusion_before: torch.Tensor,
        policy_before: torch.Tensor,
        diffusion_after: torch.Tensor,
        policy_after: torch.Tensor,
    ) -> None:
        self.diffusion_before = diffusion_before
        self.policy_before = policy_before
        self.diffusion_after = diffusion_after
        self.policy_after = policy_after


def run_decision(*, sample: bool) -> tuple[BatchRolloutDecision, RngSnapshot]:
    """Run one learned-guidance decision with explicitly seeded generators."""

    runtime = build_runtime()
    observation = build_observation()
    diffusion = torch.Generator(device="cpu").manual_seed(DIFFUSION_SEED)
    policy = torch.Generator(device="cpu").manual_seed(POLICY_SEED)
    diffusion_before = diffusion.get_state().clone()
    policy_before = policy.get_state().clone()
    if sample:
        decision = runtime.decide_batch(observation, (diffusion,), (policy,))
    else:
        decision = runtime.decide_batch_mean(observation, (diffusion,))
    return decision, RngSnapshot(
        diffusion_before,
        policy_before,
        diffusion.get_state().clone(),
        policy.get_state().clone(),
    )


def decision_snapshot(decision: BatchRolloutDecision, rng: RngSnapshot) -> dict[str, torch.Tensor]:
    """Capture every decision tensor and RNG boundary that Task A must preserve."""

    audit = decision.audit_result()
    snapshot = {name: audit[name].detach().cpu().clone() for name in AUDIT_FIELDS}
    snapshot["ego_trajectory"] = torch.from_numpy(decision.ego_trajectories.copy())
    snapshot["diffusion_rng_state_before"] = rng.diffusion_before.clone()
    snapshot["policy_rng_state_before"] = rng.policy_before.clone()
    snapshot["diffusion_rng_state_after"] = rng.diffusion_after.clone()
    snapshot["policy_rng_state_after"] = rng.policy_after.clone()
    return snapshot


def write_reference_fixture() -> Path:
    """Regenerate the sampled-decision golden snapshot from the current implementation."""

    decision, rng = run_decision(sample=True)
    snapshot = decision_snapshot(decision, rng)
    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(snapshot, FIXTURE_PATH)
    return FIXTURE_PATH


def load_reference_fixture() -> dict[str, torch.Tensor]:
    return torch.load(FIXTURE_PATH, map_location="cpu", weights_only=True)
