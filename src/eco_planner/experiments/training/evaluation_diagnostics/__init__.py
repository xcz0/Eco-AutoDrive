"""Issue #94 Task H: deterministic + stochastic evaluation diagnostics."""

from .config import (
    EvaluationDiagnosticsStudyConfig,
    load_evaluation_diagnostics_study,
)
from .diagnostics import (
    beta_probe_statistics,
    build_diagnostics_summary,
    paired_beta_deltas,
    stochastic_metric_summary,
)
from .runner import run

__all__ = [
    "EvaluationDiagnosticsStudyConfig",
    "beta_probe_statistics",
    "build_diagnostics_summary",
    "load_evaluation_diagnostics_study",
    "paired_beta_deltas",
    "run",
    "stochastic_metric_summary",
]
