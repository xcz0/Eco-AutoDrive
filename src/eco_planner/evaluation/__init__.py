"""Public closed-loop evaluation API with lazily loaded online dependencies."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .artifacts import (
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
        build_matrix_report,
        load_episode_summary,
        load_job_summary,
        load_runtime_metadata,
        load_trace_artifact,
        summarize_matrix,
    )
    from .config import (
        EvaluationJobConfig,
        parse_evaluation_config,
    )
    from .engine import run_evaluation, run_evaluation_agent
    from .inference import (
        DiffusionEvaluationAgent,
        EvaluationAgent,
        EvaluationDecision,
        FabricInferenceRuntime,
        InferenceDecision,
        create_fabric_inference_runtime,
    )

_EXPORTS = {
    "DiffusionEvaluationAgent": (".inference", "DiffusionEvaluationAgent"),
    "EvaluationAgent": (".inference", "EvaluationAgent"),
    "EvaluationDecision": (".inference", "EvaluationDecision"),
    "EvaluationJobConfig": (".config", "EvaluationJobConfig"),
    "parse_evaluation_config": (".config", "parse_evaluation_config"),
    "run_evaluation": (".engine", "run_evaluation"),
    "run_evaluation_agent": (".engine", "run_evaluation_agent"),
    "load_episode_summary": (".artifacts", "load_episode_summary"),
    "load_job_summary": (".artifacts", "load_job_summary"),
    "load_runtime_metadata": (".artifacts", "load_runtime_metadata"),
    "load_trace_artifact": (".artifacts", "load_trace_artifact"),
    "CompletedEpisodeSummary": (".artifacts", "CompletedEpisodeSummary"),
    "EnergySummary": (".artifacts", "EnergySummary"),
    "EpisodeMetrics": (".artifacts", "EpisodeMetrics"),
    "EpisodeSummary": (".artifacts", "EpisodeSummary"),
    "ErrorValues": (".artifacts", "ErrorValues"),
    "ExecutionErrorSummary": (".artifacts", "ExecutionErrorSummary"),
    "FailedEpisodeSummary": (".artifacts", "FailedEpisodeSummary"),
    "JobSummary": (".artifacts", "JobSummary"),
    "MapInputAudit": (".artifacts", "MapInputAudit"),
    "NoGuidanceSummary": (".artifacts", "NoGuidanceSummary"),
    "RuntimeMetadata": (".artifacts", "RuntimeMetadata"),
    "SamplerSummary": (".artifacts", "SamplerSummary"),
    "ScenarioSummary": (".artifacts", "ScenarioSummary"),
    "SpeedSummary": (".artifacts", "SpeedSummary"),
    "TerminationSummary": (".artifacts", "TerminationSummary"),
    "TrafficObservationSummary": (".artifacts", "TrafficObservationSummary"),
    "WarmupSummary": (".artifacts", "WarmupSummary"),
    "build_matrix_report": (".artifacts", "build_matrix_report"),
    "summarize_matrix": (".artifacts", "summarize_matrix"),
    "FabricInferenceRuntime": (".inference", "FabricInferenceRuntime"),
    "InferenceDecision": (".inference", "InferenceDecision"),
    "create_fabric_inference_runtime": (".inference", "create_fabric_inference_runtime"),
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
    "NoGuidanceSummary",
    "RuntimeMetadata",
    "SamplerSummary",
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
