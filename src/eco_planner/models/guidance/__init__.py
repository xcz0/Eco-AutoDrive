"""Reference-guidance configuration, computation, and diagnostics."""

from eco_planner.models.guidance.config import (
    GuidanceConfig,
    NoGuidanceConfig,
    OrthogonalReferenceGuidanceConfig,
    parse_guidance_config,
    validate_guidance_sampler,
)
from eco_planner.models.guidance.contracts import GuidanceGradientResult, validate_guidance_action
from eco_planner.models.guidance.diagnostics import GuidanceDiagnostics
from eco_planner.models.guidance.orthogonal import OrthogonalGuidance

__all__ = [
    "GuidanceConfig",
    "GuidanceDiagnostics",
    "GuidanceGradientResult",
    "NoGuidanceConfig",
    "OrthogonalGuidance",
    "OrthogonalReferenceGuidanceConfig",
    "parse_guidance_config",
    "validate_guidance_action",
    "validate_guidance_sampler",
]
