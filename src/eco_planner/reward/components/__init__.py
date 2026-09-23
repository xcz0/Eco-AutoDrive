"""Reusable reward-component mappings over objective-neutral metrics."""

from .comfort import ComfortRewardConfig, comfort_score
from .energy import EnergyRewardConfig, calibrated_band_score, energy_score
from .progress import ProgressRewardConfig, progress_score
from .safety import RewardGatesConfig, safety_gate
from .speed import SpeedRewardConfig, speed_score
from .strict import StrictRewardModel
from .ttc import TTCRewardConfig, ttc_score

__all__ = [
    "ComfortRewardConfig",
    "EnergyRewardConfig",
    "ProgressRewardConfig",
    "RewardGatesConfig",
    "SpeedRewardConfig",
    "StrictRewardModel",
    "TTCRewardConfig",
    "calibrated_band_score",
    "comfort_score",
    "energy_score",
    "progress_score",
    "safety_gate",
    "speed_score",
    "ttc_score",
]
