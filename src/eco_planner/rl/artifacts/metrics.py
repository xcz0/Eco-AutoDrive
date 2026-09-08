"""Transition-weighted rollout aggregations, independent of PPO mathematics."""

from dataclasses import asdict
from typing import cast

import torch
from tensordict import TensorDictBase
from torchmetrics import MaxMetric, MeanMetric, SumMetric

from eco_planner.rl.artifacts.schema import TrainingUpdateSummary
from eco_planner.rl.optimization.ppo import PPOUpdateReport
from eco_planner.rl.rollout.contracts import RolloutEpisode, concatenate_tensordicts


class RolloutMetrics:
    def __init__(self, trajectory: TensorDictBase) -> None:
        self.trajectory = trajectory

    def _compute(self, name: str, metric: MeanMetric | SumMetric | MaxMetric) -> float:
        value = self.trajectory[name].detach()
        metric.set_dtype(value.dtype)
        metric.to(value.device)
        metric.update(value)
        return float(metric.compute())

    def mean(self, name: str) -> float:
        return self._compute(name, MeanMetric(nan_strategy="error"))

    def sum(self, name: str) -> float:
        return self._compute(name, SumMetric(nan_strategy="error"))

    def maximum(self, name: str) -> float:
        return self._compute(name, MaxMetric(nan_strategy="error"))

    def stopped_fraction(self) -> float:
        metric = MeanMetric(nan_strategy="error").to(self.trajectory.device)
        metric.update(self.trajectory["stopped"].detach().float())
        return float(metric.compute())


def build_update_summary(
    update_index: int, episodes: tuple[RolloutEpisode, ...], report: PPOUpdateReport
) -> TrainingUpdateSummary:
    if not episodes:
        raise ValueError("training update summary requires at least one episode")
    reward_profiles = {episode.reward_profile for episode in episodes}
    if len(reward_profiles) != 1:
        raise ValueError("training update cannot mix rollout reward profiles")
    reward_profile = reward_profiles.pop()
    trajectory = concatenate_tensordicts([episode.audit for episode in episodes])
    metrics = RolloutMetrics(trajectory)
    sample_count = trajectory.batch_size[0]
    episode_count = len(episodes)
    mean_episode_length = sample_count / episode_count
    collision = (
        _tensor(trajectory, "crash_vehicle")
        | _tensor(trajectory, "crash_object")
        | _tensor(trajectory, "crash_building")
        | _tensor(trajectory, "crash_human")
    )
    collision |= _tensor(trajectory, "crash_sidewalk")
    state_value = _tensor(trajectory, "state_value")
    beta_alpha = _tensor(trajectory, "beta_alpha")
    beta_beta = _tensor(trajectory, "beta_beta")
    guidance_action = _tensor(trajectory, "guidance_action")
    payload = {
        "update_index": update_index,
        "sample_count": sample_count,
        "episode_count": episode_count,
        "mean_episode_length": float(mean_episode_length),
        "total_reward": metrics.sum("reward_total"),
        "base_reward": metrics.sum("reward_base_total"),
        "mean_safety_gate": metrics.mean("reward_safety_gate"),
        "route_completion_delta": metrics.sum("route_completion_delta"),
        "distance_m": metrics.sum("distance_m"),
        "mean_speed_mps": metrics.mean("speed_mps"),
        "stopped_fraction": metrics.stopped_fraction(),
        "collision_count": int(collision.sum()),
        "out_of_road_count": int(_tensor(trajectory, "out_of_road").sum()),
        "maximum_position_error_m": metrics.maximum("position_error_m"),
        "maximum_heading_error_rad": metrics.maximum("heading_error_rad"),
        "beta_alpha_mean": tuple(float(value) for value in beta_alpha.mean(dim=0)),
        "beta_alpha_min": tuple(float(value) for value in beta_alpha.min(dim=0).values),
        "beta_alpha_max": tuple(float(value) for value in beta_alpha.max(dim=0).values),
        "beta_beta_mean": tuple(float(value) for value in beta_beta.mean(dim=0)),
        "beta_beta_min": tuple(float(value) for value in beta_beta.min(dim=0).values),
        "beta_beta_max": tuple(float(value) for value in beta_beta.max(dim=0).values),
        "action_mean": tuple(float(value) for value in guidance_action.mean(dim=0)),
        "action_std": tuple(float(value) for value in guidance_action.std(dim=0, correction=0)),
        "action_min": tuple(float(value) for value in guidance_action.min(dim=0).values),
        "action_max": tuple(float(value) for value in guidance_action.max(dim=0).values),
        "mean_state_value": float(state_value.mean()),
        "std_state_value": float(state_value.std(correction=0)),
    }
    proxy_total = metrics.sum("executed_fuel_proxy_step_energy_ml")
    distance_total = metrics.sum("step_distance_m")
    payload.update(
        {
            "reward_profile": reward_profile,
            "native_step_energy_total_ml": metrics.sum("native_step_energy_ml"),
            "executed_fuel_proxy_total_ml": proxy_total,
            "executed_fuel_proxy_distance_m": distance_total,
            "executed_fuel_proxy_ml_per_km": (
                proxy_total * 1000.0 / distance_total if distance_total > 0.0 else None
            ),
            "reward_component_means": {
                name: metrics.mean(f"reward_component_{name}")
                for name in ("ttc", "progress", "comfort", "speed", "energy")
            },
            "reward_diagnostic_means": {
                name: metrics.mean(f"reward_diagnostic_{name}")
                for name in ("collision_score", "drivable_score", "wrong_direction_score")
            },
        }
    )
    payload.update(asdict(report))
    return TrainingUpdateSummary.model_validate(payload)


def _tensor(trajectory: TensorDictBase, key: str) -> torch.Tensor:
    return cast(torch.Tensor, trajectory[key])
