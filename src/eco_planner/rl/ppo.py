"""TorchRL-backed GAE and clipped PPO updates for Stage 5."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import (
    ProbabilisticTensorDictModule,
    ProbabilisticTensorDictSequential,
    TensorDictModule,
)
from torch import nn
from torch.distributions import AffineTransform, Beta, Independent, TransformedDistribution
from torch.nn.utils import clip_grad_norm_
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE

from eco_planner.rl.policy import ExplorationPolicy, ExplorationPolicyContext
from eco_planner.rl.ppo_config import PPOOptimizationConfig
from eco_planner.rl.rollout import RolloutEpisode

_CONTEXT_KEYS = (
    "scene_tokens",
    "scene_padding_mask",
    "navigation_tokens",
    "navigation_padding_mask",
    "reference_trajectory",
)


@dataclass(frozen=True)
class GAEEstimate:
    """One episode's non-differentiable advantage and value target."""

    advantage: torch.Tensor
    value_target: torch.Tensor

    def __post_init__(self) -> None:
        if self.advantage.ndim != 2 or self.advantage.shape[-1] != 1:
            raise ValueError("GAE advantage must have shape [T, 1]")
        if self.value_target.shape != self.advantage.shape:
            raise ValueError("GAE value target must match advantage shape")
        for name, value in (("advantage", self.advantage), ("value target", self.value_target)):
            if value.device.type != "cpu" or value.dtype != torch.float32:
                raise TypeError(f"GAE {name} must be a CPU float32 tensor")
            if value.requires_grad or not torch.isfinite(value).all():
                raise ValueError(f"GAE {name} must be finite and detached")


@dataclass(frozen=True)
class PPOUpdateReport:
    """Scalar diagnostics from one complete PPO update."""

    sample_count: int
    optimizer_step_count: int
    mean_policy_loss: float
    mean_value_loss: float
    mean_entropy_loss: float
    mean_total_loss: float
    mean_approximate_kl: float
    mean_clip_fraction: float
    mean_entropy: float
    mean_explained_variance: float
    maximum_pre_clip_gradient_norm: float
    final_learning_rate: float


class _AffineBetaDistribution(TransformedDistribution):
    """Two independent Betas transformed to the strict guidance interval (-1, 1)."""

    arg_constraints: dict[str, object] = {}
    has_rsample = True

    def __init__(self, alpha: torch.Tensor, beta: torch.Tensor) -> None:
        if alpha.ndim != 2 or alpha.shape[-1] != 2 or beta.shape != alpha.shape:
            raise ValueError("affine Beta parameters must both have shape [B, 2]")
        if alpha.dtype != beta.dtype or alpha.device != beta.device:
            raise ValueError("affine Beta parameters must share dtype and device")
        if not torch.isfinite(alpha).all() or not torch.isfinite(beta).all():
            raise ValueError("affine Beta parameters must be finite")
        self.alpha = alpha
        self.beta = beta
        base = Independent(Beta(alpha, beta, validate_args=True), 1)
        super().__init__(
            base,
            [AffineTransform(loc=-1.0, scale=2.0)],
            validate_args=True,
        )

    @property
    def mean(self) -> torch.Tensor:
        return 2.0 * self.alpha / (self.alpha + self.beta) - 1.0

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        self._validate_guidance_action(value)
        result = super().log_prob(value)
        if result.ndim != 1 or not torch.isfinite(result).all():
            raise ValueError("affine Beta joint log-probability must be finite with shape [B]")
        return result

    def entropy(self) -> torch.Tensor:
        result = self.base_dist.entropy() + 2.0 * math.log(2.0)
        if result.ndim != 1 or not torch.isfinite(result).all():
            raise ValueError("affine Beta joint entropy must be finite with shape [B]")
        return result

    def _validate_guidance_action(self, value: torch.Tensor) -> None:
        if value.shape != self.alpha.shape or value.dtype != self.alpha.dtype:
            raise ValueError("guidance action must match affine Beta parameter shape and dtype")
        if value.device != self.alpha.device:
            raise ValueError("guidance action must use the affine Beta parameter device")
        if not torch.isfinite(value).all() or torch.any((value <= -1.0) | (value >= 1.0)):
            raise ValueError("guidance action must be finite and strictly inside (-1, 1)")


class _PolicyParameterAdapter(nn.Module):
    def __init__(self, policy: ExplorationPolicy) -> None:
        super().__init__()
        self.policy = policy

    def forward(
        self,
        scene_tokens: torch.Tensor,
        scene_padding_mask: torch.Tensor,
        navigation_tokens: torch.Tensor,
        navigation_padding_mask: torch.Tensor,
        reference_trajectory: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.policy(
            ExplorationPolicyContext(
                scene_tokens=scene_tokens,
                scene_padding_mask=scene_padding_mask,
                navigation_tokens=navigation_tokens,
                navigation_padding_mask=navigation_padding_mask,
                reference_trajectory=reference_trajectory,
            )
        )
        return output.parameters.alpha, output.parameters.beta


class _PolicyValueAdapter(nn.Module):
    def __init__(self, policy: ExplorationPolicy) -> None:
        super().__init__()
        self.policy = policy

    def forward(
        self,
        scene_tokens: torch.Tensor,
        scene_padding_mask: torch.Tensor,
        navigation_tokens: torch.Tensor,
        navigation_padding_mask: torch.Tensor,
        reference_trajectory: torch.Tensor,
    ) -> torch.Tensor:
        value = self.policy(
            ExplorationPolicyContext(
                scene_tokens=scene_tokens,
                scene_padding_mask=scene_padding_mask,
                navigation_tokens=navigation_tokens,
                navigation_padding_mask=navigation_padding_mask,
                reference_trajectory=reference_trajectory,
            )
        ).value
        return value.unsqueeze(-1)


def estimate_episode_gae(
    episode: RolloutEpisode,
    config: PPOOptimizationConfig,
) -> GAEEstimate:
    """Use TorchRL GAE on one explicit episode boundary without normalization."""

    if not isinstance(episode, RolloutEpisode):
        raise TypeError("GAE input must be a RolloutEpisode")
    if not isinstance(config, PPOOptimizationConfig):
        raise TypeError("GAE config must be PPOOptimizationConfig")
    transition_count = len(episode.transitions)
    values = torch.stack([step.old_value for step in episode.transitions], dim=0)
    rewards = torch.stack([step.reward.total_score for step in episode.transitions], dim=0)
    next_values = torch.cat([values[1:], episode.tail_bootstrap_value.reshape(1, 1)], dim=0)
    done = torch.zeros((transition_count, 1), dtype=torch.bool)
    done[-1] = True
    terminated = torch.tensor([[step.terminated] for step in episode.transitions], dtype=torch.bool)
    tensordict = TensorDict(
        {
            "state_value": values,
            "next": TensorDict(
                {
                    "state_value": next_values,
                    "reward": rewards,
                    "done": done,
                    "terminated": terminated,
                },
                batch_size=[transition_count],
            ),
        },
        batch_size=[transition_count],
    )
    estimator = GAE(
        gamma=config.gamma,
        lmbda=config.gae_lambda,
        value_network=None,
        average_gae=False,
        differentiable=False,
        vectorized=False,
        skip_existing=False,
        time_dim=0,
        auto_reset_env=False,
    )
    estimator(tensordict)
    return GAEEstimate(
        advantage=tensordict["advantage"].detach().clone(),
        value_target=tensordict["value_target"].detach().clone(),
    )


class PPOUpdater:
    """Own TorchRL ClipPPOLoss plus the Stage-5 Adam/cosine update state."""

    def __init__(self, policy: ExplorationPolicy, config: PPOOptimizationConfig) -> None:
        if not isinstance(policy, ExplorationPolicy):
            raise TypeError("PPO updater policy must be ExplorationPolicy")
        if not isinstance(config, PPOOptimizationConfig):
            raise TypeError("PPO updater config must be PPOOptimizationConfig")
        parameters = tuple(policy.parameters())
        if not parameters or any(not parameter.requires_grad for parameter in parameters):
            raise ValueError("all Exploration Policy parameters must be trainable")
        devices = {parameter.device for parameter in parameters}
        if len(devices) != 1:
            raise ValueError("Exploration Policy parameters must use one device")
        self.policy = policy
        self.policy.eval()
        self.config = config
        self.device = devices.pop()
        self._parameters = parameters
        actor, critic = _build_torchrl_policy_adapters(policy)
        self.loss_module = ClipPPOLoss(
            actor_network=actor,
            critic_network=critic,
            clip_epsilon=config.clip_epsilon,
            entropy_bonus=True,
            entropy_coeff=config.entropy_coefficient,
            critic_coeff=config.value_coefficient,
            loss_critic_type=config.value_loss,
            normalize_advantage=False,
            separate_losses=False,
            functional=False,
            reduction="mean",
            clip_value=None,
            device=self.device,
        )
        self.loss_module.set_keys(
            action="guidance_action",
            sample_log_prob="old_joint_guidance_log_prob",
            value="state_value",
            advantage="advantage",
            value_target="value_target",
            reward="reward",
            done="done",
            terminated="terminated",
        )
        self.loss_module.eval()
        self.optimizer = torch.optim.Adam(
            self._parameters,
            lr=config.learning_rate,
            eps=config.adam_epsilon,
            weight_decay=config.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.scheduler_total_optimizer_steps,
            eta_min=config.scheduler_minimum_learning_rate,
        )
        self._minibatch_generator = torch.Generator(device="cpu").manual_seed(config.minibatch_seed)
        self._completed_optimizer_steps = 0

    def update(self, episodes: Sequence[RolloutEpisode]) -> PPOUpdateReport:
        """Perform all configured PPO epochs over one immutable rollout batch."""

        self.policy.eval()
        batch = _prepare_ppo_batch(episodes, self.config)
        sample_count = batch.batch_size[0]
        if sample_count != self.config.batch_size:
            raise ValueError(
                f"PPO batch contains {sample_count} samples, expected {self.config.batch_size}"
            )
        required_steps = self.config.optimizer_steps_per_update
        if self._completed_optimizer_steps + required_steps > (
            self.config.scheduler_total_optimizer_steps
        ):
            raise RuntimeError("PPO update would exceed the configured scheduler horizon")
        _normalize_full_batch_advantage(batch)
        frozen_inputs = {
            key: batch[key].clone()
            for key in ("old_joint_guidance_log_prob", "state_value", "advantage", "value_target")
        }
        batch = batch.to(self.device)
        metric_values: dict[str, list[float]] = {
            "loss_objective": [],
            "loss_critic": [],
            "loss_entropy": [],
            "total_loss": [],
            "kl_approx": [],
            "clip_fraction": [],
            "entropy": [],
            "explained_variance": [],
            "gradient_norm": [],
        }
        for _epoch in range(self.config.epochs):
            for host_indices in self._epoch_minibatch_indices(sample_count):
                indices = host_indices.to(self.device)
                minibatch = batch[indices]
                losses = self.loss_module(minibatch)
                total_loss = (
                    losses["loss_objective"] + losses["loss_critic"] + losses["loss_entropy"]
                )
                _require_finite_scalar(total_loss, "total PPO loss")
                self.optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                gradient_norm = clip_grad_norm_(
                    self._parameters,
                    self.config.max_gradient_norm,
                    error_if_nonfinite=True,
                )
                _require_finite_scalar(gradient_norm, "pre-clip gradient norm")
                self.optimizer.step()
                self.scheduler.step()
                self._completed_optimizer_steps += 1
                for name in (
                    "loss_objective",
                    "loss_critic",
                    "loss_entropy",
                    "kl_approx",
                    "clip_fraction",
                    "entropy",
                    "explained_variance",
                ):
                    _require_finite_scalar(losses[name], name)
                    metric_values[name].append(float(losses[name].detach().mean().cpu()))
                metric_values["total_loss"].append(float(total_loss.detach().cpu()))
                metric_values["gradient_norm"].append(float(gradient_norm.detach().cpu()))
        host_batch = batch.to("cpu")
        for key, expected in frozen_inputs.items():
            if not torch.equal(host_batch[key], expected):
                raise RuntimeError(f"PPO update mutated frozen batch field {key!r}")
        return PPOUpdateReport(
            sample_count=sample_count,
            optimizer_step_count=required_steps,
            mean_policy_loss=_mean(metric_values["loss_objective"]),
            mean_value_loss=_mean(metric_values["loss_critic"]),
            mean_entropy_loss=_mean(metric_values["loss_entropy"]),
            mean_total_loss=_mean(metric_values["total_loss"]),
            mean_approximate_kl=_mean(metric_values["kl_approx"]),
            mean_clip_fraction=_mean(metric_values["clip_fraction"]),
            mean_entropy=_mean(metric_values["entropy"]),
            mean_explained_variance=_mean(metric_values["explained_variance"]),
            maximum_pre_clip_gradient_norm=max(metric_values["gradient_norm"]),
            final_learning_rate=float(self.optimizer.param_groups[0]["lr"]),
        )

    def _epoch_minibatch_indices(self, sample_count: int) -> tuple[torch.Tensor, ...]:
        if sample_count != self.config.batch_size:
            raise ValueError("minibatch indexing requires the configured full batch size")
        permutation = torch.randperm(sample_count, generator=self._minibatch_generator)
        return tuple(
            permutation[start : start + self.config.minibatch_size]
            for start in range(0, sample_count, self.config.minibatch_size)
        )


def _build_torchrl_policy_adapters(
    policy: ExplorationPolicy,
) -> tuple[ProbabilisticTensorDictSequential, TensorDictModule]:
    parameter_module = TensorDictModule(
        _PolicyParameterAdapter(policy),
        in_keys=list(_CONTEXT_KEYS),
        out_keys=["alpha", "beta"],
    )
    distribution_module = ProbabilisticTensorDictModule(
        in_keys={"alpha": "alpha", "beta": "beta"},
        out_keys=["guidance_action"],
        distribution_class=_AffineBetaDistribution,
        return_log_prob=True,
        log_prob_key="joint_guidance_log_prob",
    )
    actor = ProbabilisticTensorDictSequential(parameter_module, distribution_module)
    critic = TensorDictModule(
        _PolicyValueAdapter(policy),
        in_keys=list(_CONTEXT_KEYS),
        out_keys=["state_value"],
    )
    return actor, critic


def _prepare_ppo_batch(
    episodes: Sequence[RolloutEpisode],
    config: PPOOptimizationConfig,
) -> TensorDictBase:
    if isinstance(episodes, (str, bytes)) or not isinstance(episodes, Sequence):
        raise TypeError("PPO update episodes must be a sequence")
    episode_tuple = tuple(episodes)
    if not episode_tuple:
        raise ValueError("PPO update requires at least one rollout episode")
    estimates = [estimate_episode_gae(episode, config) for episode in episode_tuple]
    transitions = [step for episode in episode_tuple for step in episode.transitions]
    sample_count = len(transitions)
    contexts = [step.policy_context for step in transitions]
    next_payload = TensorDict(
        {
            "reward": torch.cat([step.reward.total_score for step in transitions], dim=0).reshape(
                sample_count, 1
            ),
            "done": torch.cat(
                [
                    torch.tensor(
                        [
                            [index == len(episode.transitions) - 1]
                            for index in range(len(episode.transitions))
                        ],
                        dtype=torch.bool,
                    )
                    for episode in episode_tuple
                ],
                dim=0,
            ),
            "terminated": torch.tensor(
                [[step.terminated] for step in transitions], dtype=torch.bool
            ),
        },
        batch_size=[sample_count],
    )
    return TensorDict(
        {
            "scene_tokens": torch.cat([context.scene_tokens for context in contexts], dim=0),
            "scene_padding_mask": torch.cat(
                [context.scene_padding_mask for context in contexts], dim=0
            ),
            "navigation_tokens": torch.cat(
                [context.navigation_tokens for context in contexts], dim=0
            ),
            "navigation_padding_mask": torch.cat(
                [context.navigation_padding_mask for context in contexts], dim=0
            ),
            "reference_trajectory": torch.cat(
                [context.reference_trajectory for context in contexts], dim=0
            ),
            "guidance_action": torch.cat([step.guidance_action for step in transitions], dim=0),
            "old_joint_guidance_log_prob": torch.cat(
                [step.old_joint_guidance_log_prob for step in transitions], dim=0
            ),
            "state_value": torch.cat([step.old_value for step in transitions], dim=0).reshape(
                sample_count, 1
            ),
            "advantage": torch.cat([estimate.advantage for estimate in estimates], dim=0),
            "value_target": torch.cat([estimate.value_target for estimate in estimates], dim=0),
            "next": next_payload,
        },
        batch_size=[sample_count],
    )


def _normalize_full_batch_advantage(batch: TensorDictBase) -> None:
    advantage = batch["advantage"]
    if advantage.numel() < 2:
        raise ValueError("advantage normalization requires at least two samples")
    mean = advantage.mean()
    standard_deviation = advantage.std(correction=1)
    if not torch.isfinite(mean) or not torch.isfinite(standard_deviation):
        raise ValueError("advantage normalization statistics must be finite")
    if standard_deviation.item() == 0.0:
        raise ValueError("advantage normalization rejects zero variance")
    normalized = (advantage - mean) / standard_deviation
    if not torch.isfinite(normalized).all():
        raise ValueError("normalized advantage must be finite")
    batch["advantage"] = normalized


def _require_finite_scalar(value: torch.Tensor, name: str) -> None:
    if not isinstance(value, torch.Tensor) or value.numel() != 1:
        raise ValueError(f"{name} must be a scalar tensor")
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"{name} must be finite")


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("PPO metric aggregation requires at least one value")
    result = sum(values) / len(values)
    if not math.isfinite(result):
        raise FloatingPointError("PPO metric mean must be finite")
    return result
