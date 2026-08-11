from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from eco_planner.models.config import OfficialDiffusionPlannerConfig
from eco_planner.models.diffusion_planner import DiffusionPlanner
from eco_planner.models.pretrained import PretrainedDiffusionPlanner
from eco_planner.models.sampling_config import Ddim5SamplerConfig, Dpm10SamplerConfig


class _IdentityStateNormalizer:
    def inverse(self, value: torch.Tensor) -> torch.Tensor:
        return value


class _FakeDenoiser(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.ones(()))
        self.samples: list[torch.Tensor] = []

    def encode(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.zeros((inputs["ego_current_state"].shape[0], 1, 1))

    def denoise(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        encoding: torch.Tensor,
        route_lanes: torch.Tensor,
        current_mask: torch.Tensor,
    ) -> torch.Tensor:
        self.samples.append(sample.detach().clone())
        return torch.zeros_like(sample, dtype=torch.float64)


def test_pretrained_planner_remains_frozen(
    official_model_config: OfficialDiffusionPlannerConfig,
) -> None:
    planner = PretrainedDiffusionPlanner(
        official_model_config,
        DiffusionPlanner(official_model_config),
        Dpm10SamplerConfig(),
    )

    assert not planner.training
    assert not planner.model.training
    assert all(not parameter.requires_grad for parameter in planner.parameters())
    assert planner.eval() is planner
    with pytest.raises(RuntimeError, match="cannot enter train mode"):
        planner.train()


def test_pretrained_planner_detects_child_train_mode(
    official_model_config: OfficialDiffusionPlannerConfig,
) -> None:
    planner = PretrainedDiffusionPlanner(
        official_model_config,
        DiffusionPlanner(official_model_config),
        Dpm10SamplerConfig(),
    )
    planner.model.train()

    with pytest.raises(RuntimeError, match="remain in eval mode"):
        planner({}, torch.empty(0))


def test_pretrained_planner_tracks_actual_runtime_device(
    official_model_config: OfficialDiffusionPlannerConfig,
) -> None:
    planner = PretrainedDiffusionPlanner(
        official_model_config,
        DiffusionPlanner(official_model_config),
        Dpm10SamplerConfig(),
    )
    planner.to(torch.device("cpu"))
    assert planner._runtime_device == torch.device("cpu")


@pytest.mark.gpu
def test_pretrained_planner_tracks_cuda_device(
    official_model_config: OfficialDiffusionPlannerConfig,
) -> None:
    planner = PretrainedDiffusionPlanner(
        official_model_config,
        DiffusionPlanner(official_model_config),
        Dpm10SamplerConfig(),
    )
    planner.to(torch.device("cuda"))
    assert planner._runtime_device.type == "cuda"


@pytest.mark.parametrize(
    ("scale", "label"),
    [(1.0, "plannerrft_paper_text"), (0.5, "project_noise_scale_0_5")],
)
def test_pretrained_ddim_applies_explicit_noise_scale_and_sampler_dtype_boundary(
    stage0_observation: dict[str, torch.Tensor],
    scale: float,
    label: str,
) -> None:
    config = SimpleNamespace(
        predicted_neighbor_num=10,
        future_len=80,
        observation_normalizer=lambda observation: dict(observation),
        state_normalizer=_IdentityStateNormalizer(),
    )
    model = _FakeDenoiser()
    sampler_config = Ddim5SamplerConfig(
        name="ddim5",
        num_steps=5,
        timesteps=(1.0, 0.8, 0.6, 0.4, 0.2, 0.0),
        initial_noise_scale=scale,
        ddim_stochasticity=0.0,
        parity_label=label,  # type: ignore[arg-type]
    )
    planner = PretrainedDiffusionPlanner(  # type: ignore[arg-type]
        config,
        model,
        sampler_config,
    )

    prediction = planner(stage0_observation, torch.ones((1, 11, 80, 4)))

    first_sample = model.samples[0].reshape(1, 11, 81, 4)
    assert first_sample.dtype == torch.float32
    assert torch.equal(first_sample[:, :, 1:], torch.full((1, 11, 80, 4), scale))
    assert prediction.dtype == torch.float32
