"""Planning-owned diffusion planner and guidance policy decision semantics."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .diffusion_inference import DiffusionRuntime, create_diffusion_runtime
    from .inference import (
        PlanningInference,
        PolicyGuidanceRuntime,
        create_policy_guidance_runtime,
    )
    from .result import (
        DiffusionDecisionResult,
        GuidanceAction,
        PolicyDecision,
        PolicyGuidanceDecisionResult,
    )

_EXPORTS = {
    "DiffusionRuntime": (".diffusion_inference", "DiffusionRuntime"),
    "create_diffusion_runtime": (".diffusion_inference", "create_diffusion_runtime"),
    "PlanningInference": (".inference", "PlanningInference"),
    "PolicyGuidanceRuntime": (".inference", "PolicyGuidanceRuntime"),
    "create_policy_guidance_runtime": (".inference", "create_policy_guidance_runtime"),
    "DiffusionDecisionResult": (".result", "DiffusionDecisionResult"),
    "GuidanceAction": (".result", "GuidanceAction"),
    "PolicyDecision": (".result", "PolicyDecision"),
    "PolicyGuidanceDecisionResult": (".result", "PolicyGuidanceDecisionResult"),
}


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, attribute = _EXPORTS[name]
    value = getattr(import_module(module, __name__), attribute)
    globals()[name] = value
    return value


__all__ = [
    "DiffusionDecisionResult",
    "DiffusionRuntime",
    "GuidanceAction",
    "PlanningInference",
    "PolicyDecision",
    "PolicyGuidanceDecisionResult",
    "PolicyGuidanceRuntime",
    "create_policy_guidance_runtime",
    "create_diffusion_runtime",
]
