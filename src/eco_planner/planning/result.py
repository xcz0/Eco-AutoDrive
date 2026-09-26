"""Device results of planning-owned diffusion and learned-guidance decisions."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .diffusion import GuidanceDiagnostics, PlannerInferenceResult
from .policy import ExplorationPolicyContext, ExplorationPolicyOutput


@dataclass(frozen=True)
class DiffusionInferenceTiming:
    """Opt-in synchronized model timings, excluding host result adaptation."""

    host_to_device_s: float
    execution_s: float


@dataclass(frozen=True)
class DiffusionDecisionResult:
    """One base/fixed/manual decision before workflow host or artifact mapping."""

    initial_noise: torch.Tensor
    planner: PlannerInferenceResult
    timing: DiffusionInferenceTiming | None = None


@dataclass(frozen=True)
class GuidanceAction:
    """Sampled guidance action in both unit-Beta and guidance coordinates."""

    base_action: torch.Tensor
    guidance_action: torch.Tensor
    joint_log_prob: torch.Tensor


@dataclass(frozen=True)
class PolicyDecision:
    """Policy inputs, actor/critic output, and the resulting guidance action."""

    inputs: ExplorationPolicyContext
    output: ExplorationPolicyOutput
    action: GuidanceAction


@dataclass(frozen=True)
class PolicyGuidanceDecisionResult:
    """One learned-guidance decision before any RL or evaluation adaptation."""

    prediction: torch.Tensor
    initial_noise: torch.Tensor
    reference_prediction: torch.Tensor
    policy: PolicyDecision
    guidance_diagnostics: GuidanceDiagnostics
