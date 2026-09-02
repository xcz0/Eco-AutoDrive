"""Fabric-owned evaluation inference runtime."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from time import perf_counter
from typing import cast

import torch
from lightning.fabric import Fabric
from torch import nn

from eco_planner.envs.array_types import BatchObservation
from eco_planner.models import (
    CheckpointLoadReport,
    GuidanceConfig,
    NoGuidanceConfig,
    OfficialDiffusionPlannerConfig,
    SamplerConfig,
    SamplerReport,
    load_official_diffusion_planner,
    sampler_report,
)
from eco_planner.runtime.config import RuntimeConfig
from eco_planner.runtime.fabric import (
    InferenceRuntimeReport,
    create_single_device_fabric,
)
from eco_planner.runtime.host_transfer import HostTransfer
from eco_planner.runtime.random import sample_batched_standard_normal

from .decision import (
    BatchInferenceTiming,
    InferenceDecision,
    prepare_batch_inference_decision,
    synchronize_if_cuda,
    validate_artifact_observation_fields,
    validate_optional_guidance_result,
)


class FabricInferenceRuntime:
    """Own Fabric, the wrapped planner, and inference-time tensor placement."""

    def __init__(
        self,
        fabric: Fabric,
        planner: nn.Module,
        planner_config: OfficialDiffusionPlannerConfig,
        checkpoint_report: CheckpointLoadReport,
        report: InferenceRuntimeReport,
        sampler_report: SamplerReport,
        guidance_config: GuidanceConfig,
    ) -> None:
        self._fabric = fabric
        self._planner = planner
        self.planner_config = planner_config
        self.checkpoint_report = checkpoint_report
        self.report = report
        self.sampler_report = sampler_report
        self.guidance_config = guidance_config
        self._host_transfer = HostTransfer(fabric.device)

    @property
    def device(self) -> torch.device:
        return self._fabric.device

    def new_noise_generator(self) -> torch.Generator:
        """Create one persistent per-episode generator from the configured runtime seed."""

        return torch.Generator(device=self.device).manual_seed(self.report.seed)

    def sample_noise(self, generators: Sequence[torch.Generator]) -> torch.Tensor:
        """Draw one standard-normal planner input from each slot-owned RNG stream."""

        config = self.planner_config
        return sample_batched_standard_normal(
            generators,
            (1 + config.predicted_neighbor_num, config.future_len, 4),
            device=self.device,
        )

    def infer(
        self,
        observation: BatchObservation | Mapping[str, torch.Tensor],
        generator: torch.Generator,
    ) -> InferenceDecision:
        """Run one planner pass through the shared batched inference path."""

        return self.infer_batch(observation, self.sample_noise((generator,)), (generator,))

    def infer_batch(
        self,
        observation: BatchObservation | Mapping[str, torch.Tensor],
        standard_normal_noise: torch.Tensor,
        transition_generators: Sequence[torch.Generator | None],
        *,
        profile: bool = False,
    ) -> InferenceDecision:
        """Run a batch with independently owned per-slot diffusion RNG streams."""

        raw_observation = cast(dict[str, torch.Tensor], dict(observation))
        batch = validate_artifact_observation_fields(raw_observation, self.planner_config)
        config = self.planner_config
        expected_shape = (batch, 1 + config.predicted_neighbor_num, config.future_len, 4)
        if tuple(standard_normal_noise.shape) != expected_shape:
            raise ValueError(
                f"standard_normal_noise has shape {tuple(standard_normal_noise.shape)}, "
                f"expected {expected_shape}"
            )
        if (
            standard_normal_noise.dtype != torch.float32
            or standard_normal_noise.device != self.device
        ):
            raise TypeError("standard_normal_noise must be float32 on the runtime device")
        if len(transition_generators) != batch:
            raise ValueError("transition_generators must contain one generator per batch item")

        h2d_started = perf_counter() if profile else 0.0
        moved = self._fabric.to_device(raw_observation)
        synchronize_if_cuda(self.device, profile)
        host_to_device_s = perf_counter() - h2d_started if profile else 0.0
        if not isinstance(moved, dict) or not all(
            isinstance(name, str) and isinstance(value, torch.Tensor)
            for name, value in moved.items()
        ):
            raise TypeError("Fabric must return a string-to-tensor observation mapping")
        device_observation = cast(dict[str, torch.Tensor], moved)
        execution_started = perf_counter() if profile else 0.0
        if isinstance(self.guidance_config, NoGuidanceConfig):
            with torch.inference_mode():
                result = self._planner(
                    device_observation, standard_normal_noise, transition_generators
                )
        else:
            with torch.enable_grad():
                result = self._planner(
                    device_observation, standard_normal_noise, transition_generators
                )
        synchronize_if_cuda(self.device, profile)
        execution_s = perf_counter() - execution_started if profile else 0.0
        prediction = result.prediction.detach()
        if tuple(prediction.shape) != expected_shape:
            raise RuntimeError(
                f"Diffusion Planner prediction has shape {tuple(prediction.shape)}, "
                f"expected {expected_shape}"
            )
        if prediction.device != self.device:
            raise RuntimeError("Diffusion Planner prediction must remain on the runtime device")
        validate_optional_guidance_result(
            result,
            self.guidance_config,
            expected_shape,
            self.sampler_report.num_steps,
            self.device,
        )
        execution, resolve_audit, execution_to_host_s = prepare_batch_inference_decision(
            standard_normal_noise,
            result,
            self._host_transfer,
            profile=profile,
        )
        timing = (
            BatchInferenceTiming(
                host_to_device_s=host_to_device_s,
                execution_s=execution_s,
                execution_to_host_s=execution_to_host_s,
            )
            if profile
            else None
        )
        return InferenceDecision(execution, resolve_audit, timing)


def create_fabric_inference_runtime(
    runtime_config: RuntimeConfig,
    sampler_config: SamplerConfig,
    guidance_config: GuidanceConfig,
    args_path: Path,
    checkpoint_path: Path,
) -> FabricInferenceRuntime:
    """Resolve settings, seed all RNGs, and assemble the frozen planner with Fabric."""

    fabric, report = create_single_device_fabric(
        runtime_config, configure_cuda_matmul_precision=True
    )
    planner, checkpoint_report = load_official_diffusion_planner(
        args_path,
        checkpoint_path,
        sampler_config,
        guidance_config,
    )
    planner_config = planner.config
    wrapped_planner = fabric.setup_module(planner)
    if report.world_size != 1:
        raise RuntimeError("closed-loop inference requires Fabric world_size=1")
    return FabricInferenceRuntime(
        fabric,
        wrapped_planner,
        planner_config,
        checkpoint_report,
        report,
        sampler_report(sampler_config),
        guidance_config,
    )
