from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import torch
from tensordict import TensorDictBase

from eco_planner.rl import (
    PPO_BATCH_KEYS,
    PPOConfig,
    RolloutEpisode,
    build_ppo_batch,
    concatenate_tensordicts,
)


def zero_critic_values(episode: RolloutEpisode) -> RolloutEpisode:
    """Reward-only GAE input: V(s)=V(s')=0, which also removes the tail bootstrap."""
    training = episode.training.clone()
    training["state_value"] = torch.zeros_like(training["state_value"])
    training["next", "state_value"] = torch.zeros_like(training["next", "state_value"])
    return replace(episode, training=training)


def discounted_return_batch(episodes: Sequence[RolloutEpisode], gamma: float) -> TensorDictBase:
    """Critic-free diagnostic: per-episode R_t = r_t + gamma * R_{t+1}, no bootstrap."""
    trajectories = []
    for episode in episodes:
        reward = episode.training["next", "reward"]
        returns = torch.zeros_like(reward)
        running = torch.zeros_like(reward[0])
        for step in reversed(range(reward.shape[0])):
            running = reward[step] + gamma * running
            returns[step] = running
        trajectory = episode.training.clone()
        trajectory["advantage"] = returns.clone()
        trajectory["value_target"] = returns.clone()
        trajectories.append(trajectory)
    return concatenate_tensordicts(trajectories).select(*PPO_BATCH_KEYS)


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
