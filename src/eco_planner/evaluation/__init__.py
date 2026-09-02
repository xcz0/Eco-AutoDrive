"""Public closed-loop evaluation API with lazily loaded online dependencies."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .agent import DiffusionEvaluationAgent, EvaluationAgent, EvaluationDecision
    from .config import (
        EvaluationJobConfig,
        ModelPathsConfig,
        ScenarioConfig,
        parse_evaluation_config,
    )
    from .engine import run_evaluation, run_evaluation_agent
    from .io import (
        load_episode_summary,
        load_job_summary,
        load_runtime_metadata,
        load_trace_artifact,
    )
    from .models import (
        CompletedEpisodeSummary,
        EnergySummary,
        EpisodeMetrics,
        EpisodeSummary,
        ErrorValues,
        ExecutionErrorSummary,
        FailedEpisodeSummary,
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
    from .runtime import (
        FabricInferenceRuntime,
        InferenceDecision,
        create_fabric_inference_runtime,
    )

_EXPORTS = {
    "DiffusionEvaluationAgent": (".agent", "DiffusionEvaluationAgent"),
    "EvaluationAgent": (".agent", "EvaluationAgent"),
    "EvaluationDecision": (".agent", "EvaluationDecision"),
    "EvaluationJobConfig": (".config", "EvaluationJobConfig"),
    "ModelPathsConfig": (".config", "ModelPathsConfig"),
    "ScenarioConfig": (".config", "ScenarioConfig"),
    "parse_evaluation_config": (".config", "parse_evaluation_config"),
    "run_evaluation": (".engine", "run_evaluation"),
    "run_evaluation_agent": (".engine", "run_evaluation_agent"),
    "load_episode_summary": (".io", "load_episode_summary"),
    "load_job_summary": (".io", "load_job_summary"),
    "load_runtime_metadata": (".io", "load_runtime_metadata"),
    "load_trace_artifact": (".io", "load_trace_artifact"),
    "CompletedEpisodeSummary": (".models", "CompletedEpisodeSummary"),
    "EnergySummary": (".models", "EnergySummary"),
    "EpisodeMetrics": (".models", "EpisodeMetrics"),
    "EpisodeSummary": (".models", "EpisodeSummary"),
    "ErrorValues": (".models", "ErrorValues"),
    "ExecutionErrorSummary": (".models", "ExecutionErrorSummary"),
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
    "build_matrix_report": (".report", "build_matrix_report"),
    "summarize_matrix": (".report", "summarize_matrix"),
    "FabricInferenceRuntime": (".runtime", "FabricInferenceRuntime"),
    "InferenceDecision": (".runtime", "InferenceDecision"),
    "create_fabric_inference_runtime": (".runtime", "create_fabric_inference_runtime"),
}

__all__ = [
    "CompletedEpisodeSummary",
    "DiffusionEvaluationAgent",
    "EnergySummary",
    "EpisodeMetrics",
    "EpisodeSummary",
    "ErrorValues",
    "EvaluationAgent",
    "EvaluationDecision",
    "EvaluationJobConfig",
    "ExecutionErrorSummary",
    "FabricInferenceRuntime",
    "FailedEpisodeSummary",
    "InferenceDecision",
    "JobSummary",
    "MapInputAudit",
    "ModelPathsConfig",
    "NoGuidanceSummary",
    "RuntimeMetadata",
    "SamplerSummary",
    "ScenarioConfig",
    "ScenarioSummary",
    "SpeedSummary",
    "TerminationSummary",
    "TrafficObservationSummary",
    "WarmupSummary",
    "build_matrix_report",
    "create_fabric_inference_runtime",
    "load_episode_summary",
    "load_job_summary",
    "load_runtime_metadata",
    "load_trace_artifact",
    "parse_evaluation_config",
    "run_evaluation",
    "run_evaluation_agent",
    "summarize_matrix",
]


def __getattr__(name: str) -> Any:
    """Resolve public API members without loading online dependencies for offline readers."""

    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
