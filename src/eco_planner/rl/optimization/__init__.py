"""PPO optimization and checkpoint public API."""

from eco_planner.rl.optimization.checkpoint import (
    PolicyCheckpointReport,
    TrainingCheckpointReport,
    load_exploration_policy_checkpoint,
    load_training_checkpoint,
    save_exploration_policy_checkpoint,
    save_training_checkpoint,
)
from eco_planner.rl.optimization.config import PPOConfig
from eco_planner.rl.optimization.ppo import (
    PPO_BATCH_KEYS,
    PPOGradientDiagnostics,
    PPOUpdater,
    PPOUpdateReport,
    build_ppo_batch,
    compute_episode_gae,
    normalize_full_batch_advantage,
)

__all__ = [
    "PPO_BATCH_KEYS",
    "build_ppo_batch",
    "normalize_full_batch_advantage",
    "PPOConfig",
    "PPOGradientDiagnostics",
    "PPOUpdater",
    "PPOUpdateReport",
    "PolicyCheckpointReport",
    "TrainingCheckpointReport",
    "compute_episode_gae",
    "load_exploration_policy_checkpoint",
    "load_training_checkpoint",
    "save_exploration_policy_checkpoint",
    "save_training_checkpoint",
]
