"""Online PPO training stability observation and prune conditions."""

from __future__ import annotations

import math

from eco_planner.experiments.ppo_stability.config import PruningConfig
from eco_planner.rl.artifacts import TrainingUpdateSummary


class StabilityViolation(RuntimeError):
    """Expected domain-specific instability detected during a validation run."""


class StabilityMonitor:
    """Accumulate the registered stability objective and hard-prune conditions."""

    def __init__(self, config: PruningConfig) -> None:
        self.config = config
        self.updates: list[TrainingUpdateSummary] = []
        self.baseline_episode_length: float | None = None
        self.minimum_episode_retention = math.inf

    def add(self, update: TrainingUpdateSummary) -> str | None:
        self.updates.append(update)
        if self.baseline_episode_length is None:
            self.baseline_episode_length = update.mean_episode_length
        retention = update.mean_episode_length / self.baseline_episode_length
        self.minimum_episode_retention = min(self.minimum_episode_retention, retention)
        if retention < self.config.minimum_episode_length_retention:
            return "episode_length_below_minimum_retention"
        window = self.updates[-self.config.consecutive_update_count :]
        if len(window) < self.config.consecutive_update_count:
            return None
        if all(
            item.out_of_road_count / item.sample_count >= self.config.out_of_road_fraction
            for item in window
        ):
            return "sustained_out_of_road"
        if all(item.kl_early_stopped for item in window):
            return "sustained_target_kl_early_stop"
        if self.config.clip_fraction is not None and all(
            item.mean_clip_fraction >= self.config.clip_fraction for item in window
        ):
            return "sustained_clip_fraction"
        return None

    def intermediate_payload(self) -> dict[str, object]:
        latest = self.updates[-1]
        return {
            "update": latest.update_index,
            "minimum_episode_length_retention": self.minimum_episode_retention,
            "out_of_road_fraction": latest.out_of_road_count / latest.sample_count,
            "mean_approximate_kl": latest.mean_approximate_kl,
            "mean_clip_fraction": latest.mean_clip_fraction,
            "kl_early_stopped": latest.kl_early_stopped,
        }
