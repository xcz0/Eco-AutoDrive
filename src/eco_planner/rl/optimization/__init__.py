"""PPO optimization and checkpoint public API."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from eco_planner.rl.optimization.checkpoint import (
        TrainingCheckpointReport,
        load_training_checkpoint,
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

_EXPORTS = {
    "TrainingCheckpointReport": "eco_planner.rl.optimization.checkpoint",
    "load_training_checkpoint": "eco_planner.rl.optimization.checkpoint",
    "save_training_checkpoint": "eco_planner.rl.optimization.checkpoint",
    "PPOConfig": "eco_planner.rl.optimization.config",
    "PPO_BATCH_KEYS": "eco_planner.rl.optimization.ppo",
    "PPOGradientDiagnostics": "eco_planner.rl.optimization.ppo",
    "PPOUpdater": "eco_planner.rl.optimization.ppo",
    "PPOUpdateReport": "eco_planner.rl.optimization.ppo",
    "build_ppo_batch": "eco_planner.rl.optimization.ppo",
    "compute_episode_gae": "eco_planner.rl.optimization.ppo",
    "normalize_full_batch_advantage": "eco_planner.rl.optimization.ppo",
}


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_EXPORTS[name]), name)
    globals()[name] = value
    return value


__all__ = [
    "PPO_BATCH_KEYS",
    "build_ppo_batch",
    "normalize_full_batch_advantage",
    "PPOConfig",
    "PPOGradientDiagnostics",
    "PPOUpdater",
    "PPOUpdateReport",
    "TrainingCheckpointReport",
    "compute_episode_gae",
    "load_training_checkpoint",
    "save_training_checkpoint",
]
