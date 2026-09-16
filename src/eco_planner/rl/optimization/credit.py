from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import torch
from tensordict import TensorDictBase, cat
from torchrl.objectives.value.functional import td_lambda_return_estimate

from eco_planner.rl import (
    PPO_BATCH_KEYS,
    PPOConfig,
    RolloutEpisode,
    build_ppo_batch,
)


def zero_critic_values(episode: RolloutEpisode) -> RolloutEpisode:
    """Reward-only GAE input: V(s)=V(s')=0, which also removes the tail bootstrap."""
    training = episode.training.clone()
    training["state_value"] = torch.zeros_like(training["state_value"])
    training["next", "state_value"] = torch.zeros_like(training["next", "state_value"])
    return replace(episode, training=training)


def discounted_return_batch(episodes: Sequence[RolloutEpisode], gamma: float) -> TensorDictBase:
    """Critic-free diagnostic: per-episode R_t = r_t + gamma * R_{t+1}, no bootstrap."""
    trajectory = cat([episode.training for episode in episodes])
    reward = trajectory["next", "reward"]
    returns = td_lambda_return_estimate(
        gamma=gamma,
        lmbda=1.0,
        next_state_value=torch.zeros_like(reward),
        reward=reward,
        done=trajectory["next", "done"],
        terminated=trajectory["next", "terminated"],
        time_dim=0,
    )
    trajectory["advantage"] = returns
    trajectory["value_target"] = returns.clone()
    return trajectory.select(*PPO_BATCH_KEYS)


def credit_batch(
    episodes: Sequence[RolloutEpisode], config: PPOConfig, form: str
) -> TensorDictBase:
    """Build the full PPO batch for one temporal-credit form; episode boundaries shared."""
    if form == "standard_gae":
        return build_ppo_batch(episodes, config)
    if form == "reward_only_gae":
        return build_ppo_batch([zero_critic_values(episode) for episode in episodes], config)
    if form == "discounted_return":
        return discounted_return_batch(episodes, config.gamma)
    raise ValueError(f"unknown temporal-credit form: {form}")
