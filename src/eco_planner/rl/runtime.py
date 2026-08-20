"""Fabric-owned learned-guidance rollout inference with one host transfer per decision."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from lightning.fabric import Fabric

from eco_planner.evaluation.artifacts.models import FailurePhase
from eco_planner.evaluation.config import RuntimeConfig
from eco_planner.evaluation.failures import EpisodeFailure
from eco_planner.evaluation.runtime.engine import InferenceRuntimeReport, resolve_runtime_settings
from eco_planner.models import (
    CheckpointLoadReport,
    OfficialDiffusionPlannerConfig,
    OrthogonalPolicyGuidanceConfig,
    SamplerConfig,
    SamplerReport,
    load_official_diffusion_planner,
    sampler_report,
)
from eco_planner.rl.config import ExplorationPolicyConfig
from eco_planner.rl.policy import (
    ExplorationPolicy,
    ExplorationPolicyContext,
    validate_exploration_policy_context,
)


@dataclass(frozen=True)
class HostRolloutDecision:
    """CPU data produced for one policy-guided planner decision."""

    prediction: np.ndarray
    initial_noise: torch.Tensor
    policy_context: ExplorationPolicyContext
    base_action: torch.Tensor
    guidance_action: torch.Tensor
    old_joint_guidance_log_prob: torch.Tensor
    old_value: torch.Tensor
    beta_alpha: torch.Tensor
    beta_beta: torch.Tensor
    diffusion_rng_state: torch.Tensor
    policy_rng_state: torch.Tensor

    @property
    def ego_trajectory(self) -> np.ndarray:
        return self.prediction[0, 0]


class FabricRolloutRuntime:
    """Own the frozen planner, trainable policy module, and separated rollout RNG streams."""

    def __init__(
        self,
        fabric: Fabric,
        planner: torch.nn.Module,
        policy: ExplorationPolicy,
        report: InferenceRuntimeReport,
        noise_seed: int,
        policy_action_seed: int,
        checkpoint_report: CheckpointLoadReport,
        sampler: SamplerReport,
        guidance_config: OrthogonalPolicyGuidanceConfig,
    ) -> None:
        self._fabric = fabric
        self._planner = planner
        self._policy = policy
        self._policy.eval()
        self.report = report
        self.noise_seed = noise_seed
        self.policy_action_seed = policy_action_seed
        self.checkpoint_report = checkpoint_report
        self.sampler_report = sampler
        self.guidance_config = guidance_config

    @property
    def device(self) -> torch.device:
        return self._fabric.device

    @property
    def fabric(self) -> Fabric:
        """Expose the single-device Fabric owner for training checkpoint persistence."""

        return self._fabric

    @property
    def planner_config(self) -> OfficialDiffusionPlannerConfig:
        """Expose the immutable planner architecture for observation adapters."""

        return self._planner.config

    @property
    def policy(self) -> ExplorationPolicy:
        """Expose the single trainable parameter owner to the PPO updater."""

        return self._policy

    def frozen_planner_hash(self) -> str:
        """Hash the frozen planner parameters in stable name order."""

        digest = hashlib.sha256()
        for name, parameter in sorted(self._planner.named_parameters()):
            if parameter.requires_grad:
                raise RuntimeError(f"planner parameter {name!r} is unexpectedly trainable")
            value = parameter.detach().to(device="cpu").contiguous()
            digest.update(name.encode("utf-8"))
            digest.update(value.numpy().tobytes())
        return digest.hexdigest()

    def new_noise_generator(self, seed: int | None = None) -> torch.Generator:
        selected = self.noise_seed if seed is None else _seed(seed, "noise seed")
        return torch.Generator(device=self.device).manual_seed(selected)

    def new_policy_generator(self, seed: int | None = None) -> torch.Generator:
        selected = self.policy_action_seed if seed is None else _seed(seed, "policy action seed")
        return torch.Generator(device=self.device).manual_seed(selected)

    def decide(
        self,
        observation: Mapping[str, torch.Tensor],
        diffusion_generator: torch.Generator,
        policy_generator: torch.Generator,
    ) -> HostRolloutDecision:
        """Sample one learned action and planner prediction while retaining exact replay state."""

        raw_observation = dict(observation)
        planner_config = self._planner.config
        moved = self._fabric.to_device(raw_observation)
        if not isinstance(moved, dict):
            raise TypeError("Fabric must move rollout observations as a dictionary")
        diffusion_rng_state = diffusion_generator.get_state().detach().cpu().clone()
        policy_rng_state = policy_generator.get_state().detach().cpu().clone()
        noise = torch.randn(
            (1, 1 + planner_config.predicted_neighbor_num, planner_config.future_len, 4),
            dtype=torch.float32,
            device=self.device,
            generator=diffusion_generator,
        )
        with torch.no_grad():
            prepared = self._planner.prepare_policy_guidance(moved, noise, diffusion_generator)
            policy_context = ExplorationPolicyContext(
                scene_tokens=prepared.policy_context.scene_tokens,
                scene_padding_mask=prepared.policy_context.scene_padding_mask,
                navigation_tokens=prepared.policy_context.navigation_tokens,
                navigation_padding_mask=prepared.policy_context.navigation_padding_mask,
                reference_trajectory=prepared.policy_context.reference_trajectory,
            )
            output, action = self._policy.act(policy_context, "rsample", policy_generator)
        with torch.enable_grad():
            result = self._planner.complete_policy_guidance(prepared, action.guidance_action)
        if result.guidance_action is None:
            raise RuntimeError("policy-guided planner did not return its guidance action")
        torch.testing.assert_close(result.guidance_action, action.guidance_action)
        host = _copy_to_host(
            {
                "prediction": (result.prediction, torch.float32),
                "initial_noise": (noise, torch.float32),
                "scene_tokens": (policy_context.scene_tokens, torch.float32),
                "scene_padding_mask": (policy_context.scene_padding_mask, torch.bool),
                "navigation_tokens": (policy_context.navigation_tokens, torch.float32),
                "navigation_padding_mask": (policy_context.navigation_padding_mask, torch.bool),
                "reference_trajectory": (policy_context.reference_trajectory, torch.float32),
                "base_action": (action.base_action, torch.float32),
                "guidance_action": (action.guidance_action, torch.float32),
                "old_joint_guidance_log_prob": (action.joint_guidance_log_prob, torch.float32),
                "old_value": (output.value, torch.float32),
                "beta_alpha": (output.parameters.alpha, torch.float32),
                "beta_beta": (output.parameters.beta, torch.float32),
            },
            self.device,
        )
        _validate_finite(host)
        host_context = ExplorationPolicyContext(
            scene_tokens=host["scene_tokens"],
            scene_padding_mask=host["scene_padding_mask"],
            navigation_tokens=host["navigation_tokens"],
            navigation_padding_mask=host["navigation_padding_mask"],
            reference_trajectory=host["reference_trajectory"],
        )
        _validate_rollout_context(host_context, self._policy.config)
        return HostRolloutDecision(
            prediction=host["prediction"].numpy(),
            initial_noise=host["initial_noise"],
            policy_context=host_context,
            base_action=host["base_action"],
            guidance_action=host["guidance_action"],
            old_joint_guidance_log_prob=host["old_joint_guidance_log_prob"],
            old_value=host["old_value"],
            beta_alpha=host["beta_alpha"],
            beta_beta=host["beta_beta"],
            diffusion_rng_state=diffusion_rng_state,
            policy_rng_state=policy_rng_state,
        )

    def bootstrap_value(
        self,
        observation: Mapping[str, torch.Tensor],
        diffusion_generator: torch.Generator,
    ) -> torch.Tensor:
        """Evaluate the old critic on a tail state without sampling or executing an action."""

        raw_observation = dict(observation)
        planner_config = self._planner.config
        moved = self._fabric.to_device(raw_observation)
        if not isinstance(moved, dict):
            raise TypeError("Fabric must move rollout observations as a dictionary")
        noise = torch.randn(
            (1, 1 + planner_config.predicted_neighbor_num, planner_config.future_len, 4),
            dtype=torch.float32,
            device=self.device,
            generator=diffusion_generator,
        )
        with torch.no_grad():
            prepared = self._planner.prepare_policy_guidance(moved, noise, diffusion_generator)
            context = ExplorationPolicyContext(
                scene_tokens=prepared.policy_context.scene_tokens,
                scene_padding_mask=prepared.policy_context.scene_padding_mask,
                navigation_tokens=prepared.policy_context.navigation_tokens,
                navigation_padding_mask=prepared.policy_context.navigation_padding_mask,
                reference_trajectory=prepared.policy_context.reference_trajectory,
            )
            value = self._policy(context).value
        host = _copy_to_host({"bootstrap_value": (value, torch.float32)}, self.device)[
            "bootstrap_value"
        ]
        _validate_finite({"bootstrap_value": host})
        return host


def create_fabric_rollout_runtime(
    runtime_config: RuntimeConfig,
    sampler_config: SamplerConfig,
    guidance_config: OrthogonalPolicyGuidanceConfig,
    policy_config: ExplorationPolicyConfig,
    args_path: Path,
    checkpoint_path: Path,
    policy_action_seed: int,
) -> FabricRolloutRuntime:
    """Load the frozen planner and an exploration policy through one single-device Fabric."""

    if type(policy_action_seed) is not int or policy_action_seed < 0:
        raise ValueError("policy_action_seed must be a non-negative integer")
    settings = resolve_runtime_settings(runtime_config)
    fabric = Fabric(
        accelerator=settings.resolved_accelerator,
        devices=settings.devices,
        precision=settings.resolved_precision,
    )
    fabric.seed_everything(settings.seed, workers=True, verbose=False)
    planner, checkpoint_report = load_official_diffusion_planner(
        args_path, checkpoint_path, sampler_config, guidance_config
    )
    policy = ExplorationPolicy(policy_config)
    wrapped_planner = fabric.setup_module(planner)
    wrapped_policy = fabric.setup_module(policy)
    policy = _unwrap_exploration_policy(wrapped_policy)
    report = InferenceRuntimeReport(
        requested_accelerator=settings.requested_accelerator,
        resolved_accelerator=settings.resolved_accelerator,
        requested_precision=settings.requested_precision,
        resolved_precision=settings.resolved_precision,
        device=str(fabric.device),
        seed=settings.seed,
        world_size=int(fabric.world_size),
    )
    if report.world_size != 1:
        raise RuntimeError("rollout runtime requires Fabric world_size=1")
    return FabricRolloutRuntime(
        fabric,
        wrapped_planner,
        policy,
        report,
        noise_seed=settings.seed,
        policy_action_seed=policy_action_seed,
        checkpoint_report=checkpoint_report,
        sampler=sampler_report(sampler_config),
        guidance_config=guidance_config,
    )


def _copy_to_host(
    tensors: Mapping[str, tuple[torch.Tensor, torch.dtype]], device: torch.device
) -> dict[str, torch.Tensor]:
    if device.type != "cuda":
        return {
            name: value.detach().to(device="cpu", dtype=dtype)
            for name, (value, dtype) in tensors.items()
        }
    copied: dict[str, torch.Tensor] = {}
    for name, (value, dtype) in tensors.items():
        destination = torch.empty(value.shape, dtype=dtype, device="cpu", pin_memory=True)
        destination.copy_(value.detach().to(dtype=dtype), non_blocking=True)
        copied[name] = destination
    torch.cuda.current_stream(device).synchronize()
    return copied


def _unwrap_exploration_policy(module: torch.nn.Module) -> ExplorationPolicy:
    if isinstance(module, ExplorationPolicy):
        return module
    unwrapped = getattr(module, "module", None)
    if not isinstance(unwrapped, ExplorationPolicy):
        raise TypeError("Fabric did not preserve the ExplorationPolicy module")
    return unwrapped


def _validate_finite(tensors: Mapping[str, torch.Tensor]) -> None:
    for name, value in tensors.items():
        if value.dtype.is_floating_point and not torch.isfinite(value).all():
            raise EpisodeFailure(
                FailurePhase.INFERENCE,
                RuntimeError(f"rollout host tensor {name!r} contains non-finite values"),
            )


def _validate_rollout_context(
    context: ExplorationPolicyContext, config: ExplorationPolicyConfig
) -> None:
    try:
        validate_exploration_policy_context(context, config)
    except (TypeError, ValueError) as error:
        raise EpisodeFailure(FailurePhase.INFERENCE, error) from error


def _seed(value: int, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value
