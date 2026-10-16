"""GAE and PPO inputs derived from immutable training trajectories."""

from __future__ import annotations

import torch
from tensordict import TensorDictBase
from torchrl.objectives.value import GAE

from eco_planner.planning.policy import POLICY_CONTEXT_KEYS
from eco_planner.rl.optimization.config import PPOConfig
from eco_planner.rl.rollout.contracts import TRAINING_KEYS

PPO_BATCH_KEYS = (
    *POLICY_CONTEXT_KEYS,
    "guidance_action",
    "old_joint_guidance_log_prob",
    "advantage",
    "value_target",
)


def compute_gae(
    training: TensorDictBase,
    config: PPOConfig,
    *,
    device: torch.device | None = None,
) -> TensorDictBase:
    """Add detached GAE outputs without changing the source trajectory."""

    if len(training.batch_size) != 1:
        raise TypeError("PPO training batch must be a one-dimensional TensorDict")
    if training.batch_size[0] == 0:
        raise ValueError("PPO update requires at least one rollout transition")
    trajectory = training.select(*TRAINING_KEYS)
    if device is not None:
        trajectory = trajectory.to(device)
    estimator = GAE(
        gamma=config.gamma,
        lmbda=config.gae_lambda,
        value_network=None,
        average_gae=False,
        differentiable=False,
        vectorized=True,
        skip_existing=False,
        time_dim=0,
        auto_reset_env=False,
        device=trajectory["state_value"].device,
    )
    estimator(trajectory)
    return trajectory


def build_ppo_batch(
    training: TensorDictBase,
    config: PPOConfig,
    *,
    device: torch.device | None = None,
) -> TensorDictBase:
    return compute_gae(training, config, device=device).select(*PPO_BATCH_KEYS)


def normalize_full_batch_advantage(batch: TensorDictBase) -> None:
    advantage = batch["advantage"]
    if advantage.numel() < 2:
        raise ValueError("advantage normalization requires at least two samples")
    mean = advantage.mean()
    standard_deviation = advantage.std(correction=1)
    if not torch.isfinite(mean) or not torch.isfinite(standard_deviation):
        raise ValueError("advantage normalization statistics must be finite")
    if standard_deviation.item() == 0.0:
        raise ValueError("advantage normalization rejects zero variance")
    normalized = (advantage - mean) / standard_deviation
    if not torch.isfinite(normalized).all():
        raise ValueError("normalized advantage must be finite")
    batch["advantage"] = normalized
