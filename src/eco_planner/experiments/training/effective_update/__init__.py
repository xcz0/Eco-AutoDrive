"""Issue #94 Task F: standard PPO effective-update region search."""

from .config import EffectiveUpdateStudyConfig, GateConfig, GridConfig, load_effective_update_study
from .diagnostics import (
    beta_statistics,
    evaluate_gate_f,
    evaluate_heldout_change,
    extract_arm_metrics,
    heldout_metric_values,
)
from .runner import run

__all__ = [
    "EffectiveUpdateStudyConfig",
    "GateConfig",
    "GridConfig",
    "beta_statistics",
    "evaluate_gate_f",
    "evaluate_heldout_change",
    "extract_arm_metrics",
    "heldout_metric_values",
    "load_effective_update_study",
    "run",
]
