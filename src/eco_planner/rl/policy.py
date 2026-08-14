"""Exploration Policy, affine Beta actions, and typed tensor contracts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
from timm.layers import Mlp
from torch import nn
from torch.distributions import Beta
from torch.nn import functional as F

from eco_planner.rl.config import ExplorationPolicyConfig


@dataclass(frozen=True)
class ExplorationPolicyContext:
    """Frozen scene/navigation features and ego-local physical reference trajectory."""

    scene_tokens: torch.Tensor
    scene_padding_mask: torch.Tensor
    navigation_tokens: torch.Tensor
    navigation_padding_mask: torch.Tensor
    reference_trajectory: torch.Tensor


@dataclass(frozen=True)
class BetaGuidanceParameters:
    """Independent lateral/longitudinal Beta parameters with shape ``[B, 2]``."""

    alpha: torch.Tensor
    beta: torch.Tensor


@dataclass(frozen=True)
class ExplorationPolicyAction:
    """One auditable action and its base/transformed joint probability quantities."""

    base_action: torch.Tensor
    guidance_action: torch.Tensor
    joint_base_log_prob: torch.Tensor
    joint_guidance_log_prob: torch.Tensor
    joint_guidance_entropy: torch.Tensor


@dataclass(frozen=True)
class ExplorationPolicyOutput:
    """Policy distribution and scalar value for one batch."""

    distribution: BetaGuidanceDistribution
    value: torch.Tensor

    @property
    def parameters(self) -> BetaGuidanceParameters:
        return self.distribution.parameters


class BetaGuidanceDistribution:
    """Two independent Betas transformed from ``u`` to guidance ``2u - 1``."""

    def __init__(self, parameters: BetaGuidanceParameters) -> None:
        _validate_parameters(parameters)
        self.parameters = parameters
        self._base_distribution = Beta(parameters.alpha, parameters.beta)

    def sample(self, generator: torch.Generator) -> ExplorationPolicyAction:
        """Draw a non-reparameterized action from only the supplied RNG stream."""

        with torch.no_grad():
            base_action = self._draw_base(generator)
        return self.evaluate_base_action(base_action)

    def rsample(self, generator: torch.Generator) -> ExplorationPolicyAction:
        """Draw a differentiable action from only the supplied RNG stream."""

        return self.evaluate_base_action(self._draw_base(generator))

    def mean(self) -> ExplorationPolicyAction:
        """Return the deterministic evaluation action; Beta mode is intentionally absent."""

        parameters = self.parameters
        return self.evaluate_base_action(parameters.alpha / (parameters.alpha + parameters.beta))

    def evaluate_base_action(self, base_action: torch.Tensor) -> ExplorationPolicyAction:
        """Recompute old/new policy quantities for a stored base action ``u``."""

        _validate_base_action(base_action, self.parameters.alpha)
        guidance_action = 2.0 * base_action - 1.0
        joint_base_log_prob = self._base_distribution.log_prob(base_action).sum(dim=-1)
        log_jacobian = 2.0 * math.log(2.0)
        joint_guidance_log_prob = joint_base_log_prob - log_jacobian
        joint_guidance_entropy = self._base_distribution.entropy().sum(dim=-1) + log_jacobian
        _validate_probability_result(joint_base_log_prob, "joint base log-prob")
        _validate_probability_result(joint_guidance_log_prob, "joint guidance log-prob")
        _validate_probability_result(joint_guidance_entropy, "joint guidance entropy")
        return ExplorationPolicyAction(
            base_action=base_action,
            guidance_action=guidance_action,
            joint_base_log_prob=joint_base_log_prob,
            joint_guidance_log_prob=joint_guidance_log_prob,
            joint_guidance_entropy=joint_guidance_entropy,
        )

    def evaluate_guidance_action(self, guidance_action: torch.Tensor) -> ExplorationPolicyAction:
        """Evaluate a strictly interior transformed action without clipping."""

        _validate_guidance_action(guidance_action, self.parameters.alpha)
        return self.evaluate_base_action((guidance_action + 1.0) / 2.0)

    def _draw_base(self, generator: torch.Generator) -> torch.Tensor:
        expected_device = self.parameters.alpha.device
        generator_device = torch.device(generator.device)
        if generator_device.type != expected_device.type or (
            generator_device.index is not None and generator_device.index != expected_device.index
        ):
            raise ValueError("policy generator must use the policy distribution device")
        alpha_draw = torch._standard_gamma(self.parameters.alpha, generator=generator)
        beta_draw = torch._standard_gamma(self.parameters.beta, generator=generator)
        base_action = alpha_draw / (alpha_draw + beta_draw)
        _validate_base_action(base_action, self.parameters.alpha)
        return base_action


class _ReferenceMixerBlock(nn.Module):
    def __init__(
        self,
        horizon: int,
        hidden_dim: int,
        token_hidden_dim: int,
        channel_hidden_dim: int,
    ) -> None:
        super().__init__()
        self.token_norm = nn.LayerNorm(hidden_dim)
        self.token_mlp = Mlp(horizon, token_hidden_dim, horizon, act_layer=nn.GELU, drop=0.0)
        self.channel_norm = nn.LayerNorm(hidden_dim)
        self.channel_mlp = Mlp(
            hidden_dim, channel_hidden_dim, hidden_dim, act_layer=nn.GELU, drop=0.0
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        token_update = self.token_norm(value).transpose(1, 2)
        value = value + self.token_mlp(token_update).transpose(1, 2)
        return value + self.channel_mlp(self.channel_norm(value))


class ExplorationPolicy(nn.Module):
    """Shared-trunk actor/value policy over lateral and longitudinal guidance."""

    def __init__(self, config: ExplorationPolicyConfig) -> None:
        super().__init__()
        self.config = config
        self.reference_projection = nn.Linear(config.reference_state_dim, config.hidden_dim)
        self.reference_mixers = nn.ModuleList(
            [
                _ReferenceMixerBlock(
                    config.reference_horizon,
                    config.hidden_dim,
                    config.reference_token_mlp_hidden_dim,
                    config.reference_channel_mlp_hidden_dim,
                )
                for _ in range(config.reference_mixer_depth)
            ]
        )
        self.reference_norm = nn.LayerNorm(config.hidden_dim)
        self.query_norm = nn.LayerNorm(config.hidden_dim)
        self.context_norm = nn.LayerNorm(config.hidden_dim)
        self.cross_attention = nn.MultiheadAttention(
            config.hidden_dim,
            config.cross_attention_heads,
            dropout=config.cross_attention_dropout,
            batch_first=True,
        )
        fusion_layers: list[nn.Module] = [nn.Linear(config.hidden_dim, config.fusion_hidden_dim)]
        for _ in range(config.fusion_mlp_depth):
            fusion_layers.extend(
                [nn.GELU(), nn.Linear(config.fusion_hidden_dim, config.fusion_hidden_dim)]
            )
        fusion_layers.append(nn.GELU())
        self.fusion_trunk = nn.Sequential(*fusion_layers)
        self.actor_head = nn.Linear(config.fusion_hidden_dim, 4)
        self.value_head = nn.Linear(config.fusion_hidden_dim, 1)
        self._initialize_symmetric_actor()

    def forward(self, context: ExplorationPolicyContext) -> ExplorationPolicyOutput:
        _validate_context(context, self.config)
        reference = self.reference_projection(context.reference_trajectory)
        for mixer in self.reference_mixers:
            reference = mixer(reference)
        reference = self.reference_norm(reference)
        conditioning = torch.cat([context.scene_tokens, context.navigation_tokens], dim=1)
        padding_mask = torch.cat(
            [context.scene_padding_mask, context.navigation_padding_mask], dim=1
        )
        normalized_conditioning = self.context_norm(conditioning)
        attended = self.cross_attention(
            self.query_norm(reference),
            normalized_conditioning,
            normalized_conditioning,
            key_padding_mask=padding_mask,
            need_weights=False,
        )[0]
        fused = self.fusion_trunk((reference + attended).mean(dim=1))
        raw_parameters = self.actor_head(fused).view(-1, 2, 2)
        concentrations = F.softplus(raw_parameters) + self.config.minimum_concentration
        parameters = BetaGuidanceParameters(
            alpha=concentrations[..., 0], beta=concentrations[..., 1]
        )
        value = self.value_head(fused).squeeze(-1)
        if tuple(value.shape) != (context.reference_trajectory.shape[0],):
            raise RuntimeError("exploration policy value must have shape [B]")
        if not torch.isfinite(value).all():
            raise ValueError("exploration policy value must be finite")
        return ExplorationPolicyOutput(BetaGuidanceDistribution(parameters), value)

    def act(
        self,
        context: ExplorationPolicyContext,
        sampling: Literal["sample", "rsample", "mean"],
        generator: torch.Generator | None = None,
    ) -> tuple[ExplorationPolicyOutput, ExplorationPolicyAction]:
        """Run forward and apply the explicitly selected sampling semantics."""

        if sampling not in {"sample", "rsample", "mean"}:
            raise ValueError("sampling must be 'sample', 'rsample', or 'mean'")
        output = self(context)
        if sampling == "mean":
            if generator is not None:
                raise ValueError("mean evaluation must not receive a generator")
            return output, output.distribution.mean()
        if generator is None:
            raise ValueError(f"{sampling} requires an explicit policy generator")
        if sampling == "sample":
            return output, output.distribution.sample(generator)
        if sampling == "rsample":
            return output, output.distribution.rsample(generator)
        raise AssertionError("validated sampling mode was not handled")

    def _initialize_symmetric_actor(self) -> None:
        nn.init.zeros_(self.actor_head.weight)
        raw_initial = _inverse_softplus(
            self.config.initial_concentration - self.config.minimum_concentration
        )
        nn.init.constant_(self.actor_head.bias, raw_initial)


def _inverse_softplus(value: float) -> float:
    return value + math.log(-math.expm1(-value))


def _validate_parameters(parameters: BetaGuidanceParameters) -> None:
    alpha = parameters.alpha
    beta = parameters.beta
    if not isinstance(alpha, torch.Tensor) or not isinstance(beta, torch.Tensor):
        raise TypeError("Beta parameters must be torch.Tensor values")
    if alpha.ndim != 2 or tuple(alpha.shape[1:]) != (2,) or beta.shape != alpha.shape:
        raise ValueError("Beta alpha and beta must both have shape [B, 2]")
    if alpha.dtype != beta.dtype or alpha.device != beta.device:
        raise ValueError("Beta alpha and beta must share dtype and device")
    if not alpha.dtype.is_floating_point:
        raise TypeError("Beta parameters must use a floating dtype")
    if not torch.isfinite(alpha).all() or not torch.isfinite(beta).all():
        raise ValueError("Beta parameters must be finite")
    if torch.any(alpha <= 0.0) or torch.any(beta <= 0.0):
        raise ValueError("Beta parameters must be strictly positive")


def _validate_base_action(action: torch.Tensor, expected: torch.Tensor) -> None:
    _validate_action_tensor(action, expected, "base action")
    if torch.any((action <= 0.0) | (action >= 1.0)):
        raise ValueError("base action must be strictly inside (0, 1)")


def _validate_guidance_action(action: torch.Tensor, expected: torch.Tensor) -> None:
    _validate_action_tensor(action, expected, "guidance action")
    if torch.any((action <= -1.0) | (action >= 1.0)):
        raise ValueError("guidance action must be strictly inside (-1, 1)")


def _validate_action_tensor(action: torch.Tensor, expected: torch.Tensor, name: str) -> None:
    if not isinstance(action, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if action.shape != expected.shape:
        raise ValueError(f"{name} must have shape [B, 2]")
    if action.dtype != expected.dtype or action.device != expected.device:
        raise ValueError(f"{name} must share dtype and device with the distribution")
    if not torch.isfinite(action).all():
        raise ValueError(f"{name} must be finite")


def _validate_probability_result(value: torch.Tensor, name: str) -> None:
    if value.ndim != 1 or not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite with shape [B]")


def _validate_context(context: ExplorationPolicyContext, config: ExplorationPolicyConfig) -> None:
    tensors = {
        "scene_tokens": context.scene_tokens,
        "navigation_tokens": context.navigation_tokens,
        "reference_trajectory": context.reference_trajectory,
    }
    if any(not isinstance(value, torch.Tensor) for value in tensors.values()):
        raise TypeError("policy context features must be torch.Tensor values")
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
        config.reference_horizon,
        config.reference_state_dim,
    ):
        raise ValueError("reference trajectory must have shape [B, 80, 4]")
    if navigation.shape[0] != batch:
        raise ValueError("policy context tensors must share the batch dimension")
    if scene.dtype != navigation.dtype or scene.dtype != reference.dtype:
        raise TypeError("policy context features must share dtype")
    if scene.device != navigation.device or scene.device != reference.device:
        raise ValueError("policy context features must share device")
    if not scene.dtype.is_floating_point:
        raise TypeError("policy context features must use a floating dtype")
    if any(not torch.isfinite(value).all() for value in tensors.values()):
        raise ValueError("policy context features must be finite")
    masks = {
        "scene padding mask": (context.scene_padding_mask, (batch, scene.shape[1])),
        "navigation padding mask": (
            context.navigation_padding_mask,
            (batch, navigation.shape[1]),
        ),
    }
    for name, (mask, shape) in masks.items():
        if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool:
            raise TypeError(f"{name} must be a bool torch.Tensor")
        if tuple(mask.shape) != shape:
            raise ValueError(f"{name} has an invalid shape")
        if mask.device != scene.device:
            raise ValueError(f"{name} must share the feature device")
    all_padding = torch.cat(
        [context.scene_padding_mask, context.navigation_padding_mask], dim=1
    ).all(dim=1)
    if torch.any(all_padding):
        raise ValueError("every policy batch item requires at least one valid context token")
