"""Issue #94 Task G: objective positive control (R0 vs Rstress)."""

from .config import (
    GateGConfig,
    ObjectivePositiveControlStudyConfig,
    PpoFreezeConfig,
    R0ArmConfig,
    RstressArmConfig,
    load_objective_positive_control_study,
)
from .diagnostics import (
    evaluate_gate_g,
    flatten_probe_delta,
    heldout_values,
    paired_heldout_deltas,
    probe_cosine,
    probe_rms,
    probe_sign_agreement,
)
from .runner import run

__all__ = [
    "GateGConfig",
    "ObjectivePositiveControlStudyConfig",
    "PpoFreezeConfig",
    "R0ArmConfig",
    "RstressArmConfig",
    "evaluate_gate_g",
    "flatten_probe_delta",
    "heldout_values",
    "load_objective_positive_control_study",
    "paired_heldout_deltas",
    "probe_cosine",
    "probe_rms",
    "probe_sign_agreement",
    "run",
]
