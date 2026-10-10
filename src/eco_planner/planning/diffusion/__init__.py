"""Application-facing diffusion planner API."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .checkpoint import CheckpointLoadReport
    from .config import (
        Ddim5SamplerConfig,
        Dpm10SamplerConfig,
        GuidanceConfig,
        NoGuidanceConfig,
        OfficialDiffusionPlannerConfig,
        OrthogonalPolicyGuidanceConfig,
        OrthogonalReferenceGuidanceConfig,
        SamplerConfig,
        SamplerReport,
        parse_guidance_config,
        parse_sampler_config,
        sampler_report,
    )
    from .guidance import GuidanceDiagnostics
    from .network import DiffusionRepresentations
    from .planner import (
        PlannerInferenceResult,
        PretrainedDiffusionPlanner,
        load_official_diffusion_planner,
    )

_EXPORTS = {
    "CheckpointLoadReport": (".checkpoint", "CheckpointLoadReport"),
    "Ddim5SamplerConfig": (".config", "Ddim5SamplerConfig"),
    "Dpm10SamplerConfig": (".config", "Dpm10SamplerConfig"),
    "GuidanceConfig": (".config", "GuidanceConfig"),
    "NoGuidanceConfig": (".config", "NoGuidanceConfig"),
    "OfficialDiffusionPlannerConfig": (".config", "OfficialDiffusionPlannerConfig"),
    "OrthogonalPolicyGuidanceConfig": (".config", "OrthogonalPolicyGuidanceConfig"),
    "OrthogonalReferenceGuidanceConfig": (".config", "OrthogonalReferenceGuidanceConfig"),
    "SamplerConfig": (".config", "SamplerConfig"),
    "SamplerReport": (".config", "SamplerReport"),
    "parse_guidance_config": (".config", "parse_guidance_config"),
    "parse_sampler_config": (".config", "parse_sampler_config"),
    "sampler_report": (".config", "sampler_report"),
    "GuidanceDiagnostics": (".guidance", "GuidanceDiagnostics"),
    "DiffusionRepresentations": (".network", "DiffusionRepresentations"),
    "PlannerInferenceResult": (".planner", "PlannerInferenceResult"),
    "PretrainedDiffusionPlanner": (".planner", "PretrainedDiffusionPlanner"),
    "load_official_diffusion_planner": (".planner", "load_official_diffusion_planner"),
}


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, attribute = _EXPORTS[name]
    value = getattr(import_module(module, __name__), attribute)
    globals()[name] = value
    return value


__all__ = [
    "CheckpointLoadReport",
    "Ddim5SamplerConfig",
    "DiffusionRepresentations",
    "Dpm10SamplerConfig",
    "GuidanceConfig",
    "GuidanceDiagnostics",
    "NoGuidanceConfig",
    "OfficialDiffusionPlannerConfig",
    "OrthogonalPolicyGuidanceConfig",
    "OrthogonalReferenceGuidanceConfig",
    "PlannerInferenceResult",
    "PretrainedDiffusionPlanner",
    "SamplerConfig",
    "SamplerReport",
    "load_official_diffusion_planner",
    "parse_guidance_config",
    "parse_sampler_config",
    "sampler_report",
]
