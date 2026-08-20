"""TorchRL-backed GAE and clipped PPO updates over compact training trajectories."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from tensordict import TensorDictBase
from tensordict.nn import (
    ProbabilisticTensorDictModule,
    ProbabilisticTensorDictSequential,
    TensorDictModule,
)
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE

from eco_planner.rl.config import PPOConfig
from eco_planner.rl.distributions import AffineBeta
from eco_planner.rl.policy import ExplorationPolicy, ExplorationPolicyContext
from eco_planner.rl.rollout import RolloutEpisode

_CONTEXT_KEYS = (
    "scene_tokens",
    "scene_padding_mask",
    "navigation_tokens",
    "navigation_padding_mask",
    "reference_trajectory",
)
_PPO_BATCH_KEYS = (
    *_CONTEXT_KEYS,
    "guidance_action",
    "old_joint_guidance_log_prob",
    "advantage",
    "value_target",
)
_PPO_IMMUTABLE_KEYS = (
    "guidance_action",
    "old_joint_guidance_log_prob",
    "advantage",
    "value_target",
)

_PPO_UPDATE_METRIC_NAMES = (
    "loss_objective",
    "loss_critic",
    "loss_entropy",
    "total_loss",
    "kl_approx",
    "clip_fraction",
    "entropy",
    "explained_variance",
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
            if value.dtype != torch.float32:
                raise TypeError(f"GAE {name} must be a float32 tensor")
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


def estimate_episode_gae(episode: RolloutEpisode, config: PPOConfig) -> GAEEstimate:
    """Use TorchRL GAE directly on one trajectory TensorDict."""

    if not isinstance(episode, RolloutEpisode):
        raise TypeError("GAE input must be a RolloutEpisode")
    if not isinstance(config, PPOConfig):
        raise TypeError("GAE config must be PPOConfig")
    tensordict = episode.training_trajectory.select("state_value", "next").clone()
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
    """Own TorchRL ClipPPOLoss plus Adam/cosine update and resume state."""

    def __init__(self, policy: ExplorationPolicy, config: PPOConfig) -> None:
        if not isinstance(policy, ExplorationPolicy):
            raise TypeError("PPO updater policy must be ExplorationPolicy")
        if not isinstance(config, PPOConfig):
            raise TypeError("PPO updater config must be PPOConfig")
        parameters = tuple(policy.parameters())
        if not parameters or any(not parameter.requires_grad for parameter in parameters):
            raise ValueError("all ExplorationPolicy parameters must be trainable")
        devices = {parameter.device for parameter in parameters}
        if len(devices) != 1:
            raise ValueError("ExplorationPolicy parameters must use one device")
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

    @property
    def completed_optimizer_steps(self) -> int:
        return self._completed_optimizer_steps

    def checkpoint_state(self) -> dict[str, object]:
        """Return non-module state required to reproduce future minibatches."""

        return {
            "completed_optimizer_steps": self._completed_optimizer_steps,
            "minibatch_generator_state": self._minibatch_generator.get_state(),
        }

    def restore_checkpoint_state(self, state: Mapping[str, object]) -> None:
        expected = {"completed_optimizer_steps", "minibatch_generator_state"}
        if set(state) != expected:
            raise ValueError("PPO checkpoint state has unexpected fields")
        completed = state["completed_optimizer_steps"]
        generator_state = state["minibatch_generator_state"]
        if (
            type(completed) is not int
            or not 0 <= completed <= self.config.scheduler_total_optimizer_steps
        ):
            raise ValueError("PPO checkpoint optimizer step count is invalid")
        if not isinstance(generator_state, torch.Tensor) or generator_state.dtype != torch.uint8:
            raise TypeError("PPO checkpoint minibatch generator state must be uint8")
        self._completed_optimizer_steps = completed
        self._minibatch_generator.set_state(generator_state)

    def update(self, episodes: Sequence[RolloutEpisode]) -> PPOUpdateReport:
        """Perform all configured PPO epochs over one immutable rollout batch."""

        self.policy.eval()
        batch = _batch_trajectories(episodes, self.config)
        sample_count = batch.batch_size[0]
        if sample_count != self.config.batch_size:
            raise ValueError(
                f"PPO batch contains {sample_count} samples, expected {self.config.batch_size}"
            )
        required_steps = self.config.optimizer_steps_per_update
        if (
            self._completed_optimizer_steps + required_steps
            > self.config.scheduler_total_optimizer_steps
        ):
            raise RuntimeError("PPO update would exceed the configured scheduler horizon")
        _normalize_full_batch_advantage(batch)
        batch = batch.to(self.device)
        frozen_inputs = {key: batch[key].clone() for key in _PPO_IMMUTABLE_KEYS}
        metric_totals = torch.zeros(
            len(_PPO_UPDATE_METRIC_NAMES), device=self.device, dtype=torch.float64
        )
        maximum_gradient_norm = torch.zeros((), device=self.device, dtype=torch.float64)
        for _epoch in range(self.config.epochs):
            for host_indices in self._epoch_minibatch_indices(sample_count):
                minibatch = batch[host_indices.to(self.device)]
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
                metric_tensors = (
                    losses["loss_objective"],
                    losses["loss_critic"],
                    losses["loss_entropy"],
                    total_loss,
                    losses["kl_approx"],
                    losses["clip_fraction"],
                    losses["entropy"],
                    losses["explained_variance"],
                )
                for name, value in zip(_PPO_UPDATE_METRIC_NAMES, metric_tensors, strict=True):
                    _require_finite_scalar(value, name)
                scalar_metrics = tuple(value.detach().mean() for value in metric_tensors)
                metric_totals.add_(torch.stack(scalar_metrics, dim=0).to(dtype=torch.float64))
                maximum_gradient_norm = torch.maximum(
                    maximum_gradient_norm, gradient_norm.detach().to(dtype=torch.float64)
                )
        for key, expected in frozen_inputs.items():
            if not torch.equal(batch[key], expected):
                raise RuntimeError(f"PPO update mutated frozen batch field {key!r}")
        host_metrics = torch.cat(
            (
                metric_totals.div(required_steps),
                maximum_gradient_norm.unsqueeze(0),
            )
        ).cpu()
        if not torch.isfinite(host_metrics).all():
            raise FloatingPointError("PPO update diagnostics must be finite")
        metric_values = tuple(float(value) for value in host_metrics)
        return PPOUpdateReport(
            sample_count=sample_count,
            optimizer_step_count=required_steps,
            mean_policy_loss=metric_values[0],
            mean_value_loss=metric_values[1],
            mean_entropy_loss=metric_values[2],
            mean_total_loss=metric_values[3],
            mean_approximate_kl=metric_values[4],
            mean_clip_fraction=metric_values[5],
            mean_entropy=metric_values[6],
            mean_explained_variance=metric_values[7],
            maximum_pre_clip_gradient_norm=metric_values[8],
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
        _PolicyParameterAdapter(policy), in_keys=list(_CONTEXT_KEYS), out_keys=["alpha", "beta"]
    )
    distribution_module = ProbabilisticTensorDictModule(
        in_keys={"alpha": "alpha", "beta": "beta"},
        out_keys=["guidance_action"],
        distribution_class=AffineBeta,
        distribution_kwargs={"validate_args": False},
        return_log_prob=True,
        log_prob_key="joint_guidance_log_prob",
    )
    actor = ProbabilisticTensorDictSequential(parameter_module, distribution_module)
    critic = TensorDictModule(
        _PolicyValueAdapter(policy), in_keys=list(_CONTEXT_KEYS), out_keys=["state_value"]
    )
    return actor, critic


def _batch_trajectories(episodes: Sequence[RolloutEpisode], config: PPOConfig) -> TensorDictBase:
    if isinstance(episodes, (str, bytes)) or not isinstance(episodes, Sequence):
        raise TypeError("PPO update episodes must be a sequence")
    episode_tuple = tuple(episodes)
    if not episode_tuple:
        raise ValueError("PPO update requires at least one rollout episode")
    trajectories = []
    for episode in episode_tuple:
        estimate = estimate_episode_gae(episode, config)
        trajectories.append(episode.with_gae(estimate.advantage, estimate.value_target))
    return torch.cat(trajectories, dim=0).select(*_PPO_BATCH_KEYS)


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
    finite = torch.isfinite(value).all()
    if value.device.type == "cpu":
        if not finite:
            raise FloatingPointError(f"{name} must be finite")
    else:
        torch._assert_async(finite, f"{name} must be finite")
