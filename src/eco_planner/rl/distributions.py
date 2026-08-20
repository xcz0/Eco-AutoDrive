"""Canonical affine-Beta probability math and explicit-RNG sampling."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch.distributions import AffineTransform, Beta, Independent, TransformedDistribution


@dataclass(frozen=True)
class AffineBetaParameters:
    """Independent lateral/longitudinal Beta parameters with shape ``[B, 2]``."""

    alpha: torch.Tensor
    beta: torch.Tensor


@dataclass(frozen=True)
class AffineBetaAction:
    """An action in both unit-Beta and guidance coordinates."""

    base_action: torch.Tensor
    guidance_action: torch.Tensor
    joint_base_log_prob: torch.Tensor
    joint_guidance_log_prob: torch.Tensor
    joint_guidance_entropy: torch.Tensor


class ExplicitGeneratorBetaSampler:
    """Draw Beta samples from an explicit replayable generator.

    PyTorch distributions do not accept a generator argument.  This isolates the
    required generator-aware gamma draw until a public API supplies one.
    """

    @staticmethod
    def draw(
        parameters: AffineBetaParameters,
        generator: torch.Generator,
        *,
        validate_args: bool,
    ) -> torch.Tensor:
        if validate_args:
            _validate_parameters(parameters)
        expected_device = parameters.alpha.device
        generator_device = torch.device(generator.device)
        if generator_device.type != expected_device.type or (
            generator_device.index is not None and generator_device.index != expected_device.index
        ):
            raise ValueError("policy generator must use the policy distribution device")
        alpha_draw = torch._standard_gamma(parameters.alpha, generator=generator)
        beta_draw = torch._standard_gamma(parameters.beta, generator=generator)
        base_action = alpha_draw / (alpha_draw + beta_draw)
        if validate_args:
            _validate_base_action(base_action, parameters.alpha)
        return base_action


class AffineBeta(TransformedDistribution):
    """Two independent Betas transformed from ``u`` to guidance ``2u - 1``."""

    arg_constraints: dict[str, object] = {}
    has_rsample = True

    def __init__(
        self, alpha: torch.Tensor, beta: torch.Tensor, validate_args: bool | None = None
    ) -> None:
        if validate_args is None:
            validate_args = True
        parameters = AffineBetaParameters(alpha, beta)
        if validate_args:
            _validate_parameters(parameters)
        self.parameters = parameters
        base = Independent(Beta(parameters.alpha, parameters.beta, validate_args=validate_args), 1)
        super().__init__(base, [AffineTransform(loc=-1.0, scale=2.0)], validate_args=validate_args)

    @property
    def mean(self) -> torch.Tensor:
        parameters = self.parameters
        return 2.0 * parameters.alpha / (parameters.alpha + parameters.beta) - 1.0

    def entropy(self) -> torch.Tensor:
        result = self.base_dist.entropy() + 2.0 * math.log(2.0)
        if self._validate_args:
            _validate_probability_result(result, "joint guidance entropy")
        return result

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        if self._validate_args:
            _validate_guidance_action(value, self.parameters.alpha)
        result = super().log_prob(value)
        if self._validate_args:
            _validate_probability_result(result, "joint guidance log-prob")
        return result

    def sample(self, generator: torch.Generator) -> AffineBetaAction:  # type: ignore[override]
        """Draw a non-differentiable action from only the supplied RNG stream."""

        with torch.no_grad():
            base_action = ExplicitGeneratorBetaSampler.draw(
                self.parameters, generator, validate_args=self._validate_args
            )
        return self.evaluate_base_action(base_action)

    def rsample(self, generator: torch.Generator) -> AffineBetaAction:  # type: ignore[override]
        """Draw a differentiable action from only the supplied RNG stream."""

        return self.evaluate_base_action(
            ExplicitGeneratorBetaSampler.draw(
                self.parameters, generator, validate_args=self._validate_args
            )
        )

    def action_mean(self) -> AffineBetaAction:
        parameters = self.parameters
        return self.evaluate_base_action(parameters.alpha / (parameters.alpha + parameters.beta))

    def evaluate_base_action(self, base_action: torch.Tensor) -> AffineBetaAction:
        if self._validate_args:
            _validate_base_action(base_action, self.parameters.alpha)
        guidance_action = 2.0 * base_action - 1.0
        joint_base_log_prob = self.base_dist.log_prob(base_action)
        joint_guidance_log_prob = self.log_prob(guidance_action)
        if self._validate_args:
            _validate_probability_result(joint_base_log_prob, "joint base log-prob")
        return AffineBetaAction(
            base_action=base_action,
            guidance_action=guidance_action,
            joint_base_log_prob=joint_base_log_prob,
            joint_guidance_log_prob=joint_guidance_log_prob,
            joint_guidance_entropy=self.entropy(),
        )

    def evaluate_guidance_action(self, guidance_action: torch.Tensor) -> AffineBetaAction:
        if self._validate_args:
            _validate_guidance_action(guidance_action, self.parameters.alpha)
        return self.evaluate_base_action((guidance_action + 1.0) / 2.0)


def _validate_parameters(parameters: AffineBetaParameters) -> None:
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
