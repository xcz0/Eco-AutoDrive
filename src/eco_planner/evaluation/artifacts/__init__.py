"""Public typed artifact and offline reporting API without eager online imports."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .io import (
        load_episode_summary,
        load_job_summary,
        load_runtime_metadata,
        load_trace_artifact,
        validate_episode_artifact,
        validate_matrix_episode,
        write_episode_artifacts,
    )
    from .models import (
        CompletedEpisodeSummary,
        EnergySummary,
        EpisodeMetrics,
        EpisodeSummary,
        ErrorValues,
        ExecutionErrorSummary,
        FailedEpisodeSummary,
        FailurePhase,
        JobSummary,
        MapInputAudit,
        NoGuidanceSummary,
        RuntimeMetadata,
        SamplerSummary,
        ScenarioSummary,
        SpeedSummary,
        TerminationSummary,
        TrafficObservationSummary,
        WarmupSummary,
    )
    from .report import build_matrix_report, summarize_matrix
    from .summary import compute_episode_metrics, compute_trace_energy

_EXPORTS = {
    "CompletedEpisodeSummary": (".models", "CompletedEpisodeSummary"),
    "EnergySummary": (".models", "EnergySummary"),
    "EpisodeMetrics": (".models", "EpisodeMetrics"),
    "EpisodeSummary": (".models", "EpisodeSummary"),
    "ErrorValues": (".models", "ErrorValues"),
    "ExecutionErrorSummary": (".models", "ExecutionErrorSummary"),
    "FailurePhase": (".models", "FailurePhase"),
    "FailedEpisodeSummary": (".models", "FailedEpisodeSummary"),
    "JobSummary": (".models", "JobSummary"),
    "MapInputAudit": (".models", "MapInputAudit"),
    "NoGuidanceSummary": (".models", "NoGuidanceSummary"),
    "RuntimeMetadata": (".models", "RuntimeMetadata"),
    "SamplerSummary": (".models", "SamplerSummary"),
    "ScenarioSummary": (".models", "ScenarioSummary"),
    "SpeedSummary": (".models", "SpeedSummary"),
    "TerminationSummary": (".models", "TerminationSummary"),
    "TrafficObservationSummary": (".models", "TrafficObservationSummary"),
    "WarmupSummary": (".models", "WarmupSummary"),
    "load_episode_summary": (".io", "load_episode_summary"),
    "load_job_summary": (".io", "load_job_summary"),
    "load_runtime_metadata": (".io", "load_runtime_metadata"),
    "load_trace_artifact": (".io", "load_trace_artifact"),
    "validate_episode_artifact": (".io", "validate_episode_artifact"),
    "validate_matrix_episode": (".io", "validate_matrix_episode"),
    "write_episode_artifacts": (".io", "write_episode_artifacts"),
    "compute_episode_metrics": (".summary", "compute_episode_metrics"),
    "compute_trace_energy": (".summary", "compute_trace_energy"),
    "build_matrix_report": (".report", "build_matrix_report"),
    "summarize_matrix": (".report", "summarize_matrix"),
}

__all__ = [
    "CompletedEpisodeSummary",
    "EnergySummary",
    "EpisodeMetrics",
    "EpisodeSummary",
    "ErrorValues",
    "ExecutionErrorSummary",
    "FailedEpisodeSummary",
    "FailurePhase",
    "JobSummary",
    "MapInputAudit",
    "NoGuidanceSummary",
    "RuntimeMetadata",
    "SamplerSummary",
    "ScenarioSummary",
    "SpeedSummary",
    "TerminationSummary",
    "TrafficObservationSummary",
    "WarmupSummary",
    "build_matrix_report",
    "compute_episode_metrics",
    "compute_trace_energy",
    "load_episode_summary",
    "load_job_summary",
    "load_runtime_metadata",
    "load_trace_artifact",
    "summarize_matrix",
    "validate_episode_artifact",
    "validate_matrix_episode",
    "write_episode_artifacts",
]


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
