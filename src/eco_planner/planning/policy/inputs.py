"""Exploration policy input contract and diffusion-representation adapter."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from tensordict import TensorDict, TensorDictBase

from eco_planner.contracts import PLANNER_HORIZON

from ..diffusion import DiffusionRepresentations
from .config import ExplorationPolicyConfig

_REFERENCE_STATE_DIM = 4
POLICY_CONTEXT_KEYS = (
    "scene_tokens",
    "scene_padding_mask",
    "navigation_tokens",
    "navigation_padding_mask",
    "reference_trajectory",
)


@dataclass(frozen=True)
class ExplorationPolicyContext:
    """Frozen scene/navigation features and ego-local physical reference trajectory."""

    scene_tokens: torch.Tensor
    scene_padding_mask: torch.Tensor
    navigation_tokens: torch.Tensor
    navigation_padding_mask: torch.Tensor
    reference_trajectory: torch.Tensor


def build_policy_inputs(
    representations: DiffusionRepresentations,
    reference_prediction: torch.Tensor,
) -> ExplorationPolicyContext:
    """Map one diffusion encoding and ego reference to the policy input contract."""

    return ExplorationPolicyContext(
        scene_tokens=representations.scene_tokens,
        scene_padding_mask=representations.scene_padding_mask,
        navigation_tokens=representations.navigation_tokens,
        navigation_padding_mask=representations.navigation_padding_mask,
        reference_trajectory=reference_prediction[:, 0],
    )


def policy_context_tensordict(context: ExplorationPolicyContext) -> TensorDictBase:
    """Expose a typed policy context through the common TensorDict key contract."""

    return TensorDict(
        {key: getattr(context, key) for key in POLICY_CONTEXT_KEYS},
        batch_size=[context.reference_trajectory.shape[0]],
    )


def validate_exploration_policy_context(
    context: ExplorationPolicyContext, config: ExplorationPolicyConfig
) -> None:
    """Strictly validate a policy context at a rollout or explicit debug boundary."""

    _validate_context_structure(context, config)
    tensors = {
        "scene_tokens": context.scene_tokens,
        "navigation_tokens": context.navigation_tokens,
        "reference_trajectory": context.reference_trajectory,
    }
    if any(not torch.isfinite(value).all() for value in tensors.values()):
        raise ValueError("policy context features must be finite")
    all_padding = torch.cat(
        [context.scene_padding_mask, context.navigation_padding_mask], dim=1
    ).all(dim=1)
    if torch.any(all_padding):
        raise ValueError("every policy batch item requires at least one valid context token")


def _validate_context_structure(
    context: ExplorationPolicyContext, config: ExplorationPolicyConfig
) -> None:
    scene = context.scene_tokens
    navigation = context.navigation_tokens
    reference = context.reference_trajectory
    if scene.ndim != 3 or scene.shape[2] != config.hidden_dim:
        raise ValueError("scene tokens must have shape [B, N, hidden_dim]")
    if navigation.ndim != 3 or navigation.shape[2] != config.hidden_dim:
        raise ValueError("navigation tokens must have shape [B, M, hidden_dim]")
    batch = scene.shape[0]
    if tuple(reference.shape) != (
        batch,
        PLANNER_HORIZON,
        _REFERENCE_STATE_DIM,
    ):
        raise ValueError(f"reference trajectory must have shape [B, {PLANNER_HORIZON}, 4]")
    if navigation.shape[0] != batch:
        raise ValueError("policy context tensors must share the batch dimension")
    if scene.dtype != navigation.dtype or scene.dtype != reference.dtype:
        raise TypeError("policy context features must share dtype")
    if scene.device != navigation.device or scene.device != reference.device:
        raise ValueError("policy context features must share device")
    if not scene.dtype.is_floating_point:
        raise TypeError("policy context features must use a floating dtype")
    masks = {
        "scene padding mask": (context.scene_padding_mask, (batch, scene.shape[1])),
        "navigation padding mask": (
            context.navigation_padding_mask,
            (batch, navigation.shape[1]),
        ),
    }
    for name, (mask, shape) in masks.items():
        if mask.dtype != torch.bool:
            raise TypeError(f"{name} must be a bool tensor")
        if tuple(mask.shape) != shape:
            raise ValueError(f"{name} has an invalid shape")
        if mask.device != scene.device:
            raise ValueError(f"{name} must share the feature device")
