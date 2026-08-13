from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from eco_planner.models.config import OfficialDiffusionPlannerConfig
from eco_planner.models.diffusion_planner import DiffusionPlanner
from eco_planner.models.guidance import OrthogonalReferenceGuidanceConfig
from eco_planner.models.planning_sampler import PlanningSampler
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


class _IdentityDenoiser(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.ones(()))
        self.encode_calls = 0

    def encode(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        self.encode_calls += 1
        return torch.zeros((inputs["ego_current_state"].shape[0], 1, 1))

    def denoise(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        encoding: torch.Tensor,
        route_lanes: torch.Tensor,
        current_mask: torch.Tensor,
    ) -> torch.Tensor:
        return sample * self.anchor


def _guidance_config() -> OrthogonalReferenceGuidanceConfig:
    return OrthogonalReferenceGuidanceConfig(
        name="orthogonal_reference",
        formula_label="centered_energy_gradient_delta_v1",
        lateral_scale=0.0,
        longitudinal_scale=0.0,
        lateral_max_offset_m=2.5,
        longitudinal_max_speed_fraction=0.25,
        trajectory_dt_s=0.1,
        gradient_step_coefficient=1.0,
        reference_refresh_cycles=1,
        share_scene_encoding=True,
        share_initial_noise=True,
        share_transition_noise=True,
        heading_norm_epsilon=1e-6,
        zero_speed_tolerance_mps=1e-6,
    )


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


def test_pretrained_planner_selects_diffusers_dpm_backend(
    official_model_config: OfficialDiffusionPlannerConfig,
) -> None:
    planner = PretrainedDiffusionPlanner(
        official_model_config,
        DiffusionPlanner(official_model_config),
        Dpm10SamplerConfig(implementation="diffusers"),
    )

    assert isinstance(planner._sampler, PlanningSampler)
    assert planner._sampler.config.implementation == "diffusers"


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
    ("scale", "label", "implementation"),
    [
        (1.0, "plannerrft_paper_text", "legacy"),
        (1.0, "plannerrft_paper_text", "diffusers"),
        (0.5, "project_noise_scale_0_5", "legacy"),
    ],
)
def test_pretrained_ddim_applies_explicit_noise_scale_and_sampler_dtype_boundary(
    stage0_observation: dict[str, torch.Tensor],
    scale: float,
    label: str,
    implementation: str,
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
        implementation=implementation,  # type: ignore[arg-type]
    )
    planner = PretrainedDiffusionPlanner(  # type: ignore[arg-type]
        config,
        model,
        sampler_config,
    )

    result = planner(stage0_observation, torch.ones((1, 11, 80, 4)))

    first_sample = model.samples[0].reshape(1, 11, 81, 4)
    assert first_sample.dtype == torch.float32
    assert torch.equal(first_sample[:, :, 1:], torch.full((1, 11, 80, 4), scale))
    assert result.prediction.dtype == torch.float32
    assert result.reference_prediction is None
    assert result.guidance_diagnostics is None


def test_neutral_reference_guidance_reuses_one_encoding_and_returns_reference_exactly(
    stage0_observation: dict[str, torch.Tensor],
) -> None:
    config = SimpleNamespace(
        predicted_neighbor_num=10,
        future_len=80,
        observation_normalizer=lambda observation: dict(observation),
        state_normalizer=_IdentityStateNormalizer(),
    )
    model = _IdentityDenoiser()
    sampler_config = Ddim5SamplerConfig(
        name="ddim5",
        num_steps=5,
        timesteps=(1.0, 0.8, 0.6, 0.4, 0.2, 0.0),
        initial_noise_scale=1.0,
        ddim_stochasticity=0.0,
        parity_label="plannerrft_paper_text",
    )
    planner = PretrainedDiffusionPlanner(  # type: ignore[arg-type]
        config,
        model,
        sampler_config,
        _guidance_config(),
    )
    noise = torch.zeros((1, 11, 80, 4), dtype=torch.float32)
    noise[..., 0] = torch.arange(1, 81, dtype=torch.float32)
    noise[..., 2] = 1.0

    result = planner(stage0_observation, noise)

    assert model.encode_calls == 1
    assert result.reference_prediction is not None
    assert torch.equal(result.prediction, result.reference_prediction)
    assert result.guidance_action is not None
    assert torch.equal(result.guidance_action, torch.zeros((1, 2)))
    assert result.guidance_diagnostics is not None
    assert torch.count_nonzero(result.guidance_diagnostics.applied_gradient_l2) == 0


@pytest.mark.parametrize("implementation", ("legacy", "diffusers"))
def test_signed_reference_guidance_moves_ego_left_and_right_without_parameter_gradients(
    stage0_observation: dict[str, torch.Tensor],
    implementation: str,
) -> None:
    config = SimpleNamespace(
        predicted_neighbor_num=10,
        future_len=80,
        observation_normalizer=lambda observation: dict(observation),
        state_normalizer=_IdentityStateNormalizer(),
    )
    sampler_config = Ddim5SamplerConfig(
        name="ddim5",
        num_steps=5,
        timesteps=(1.0, 0.8, 0.6, 0.4, 0.2, 0.0),
        initial_noise_scale=1.0,
        ddim_stochasticity=0.0,
        parity_label="plannerrft_paper_text",
        implementation=implementation,  # type: ignore[arg-type]
    )
    noise = torch.zeros((1, 11, 80, 4), dtype=torch.float32)
    noise[..., 0] = torch.arange(1, 81, dtype=torch.float32)
    noise[..., 2] = 1.0

    left_model = _IdentityDenoiser()
    left_planner = PretrainedDiffusionPlanner(  # type: ignore[arg-type]
        config, left_model, sampler_config, _guidance_config()
    )
    left = left_planner(
        stage0_observation,
        noise,
        torch.Generator().manual_seed(11),
        guidance_action=torch.tensor([[1.0, 0.0]]),
    )
    right_model = _IdentityDenoiser()
    right_planner = PretrainedDiffusionPlanner(  # type: ignore[arg-type]
        config, right_model, sampler_config, _guidance_config()
    )
    right = right_planner(
        stage0_observation,
        noise,
        torch.Generator().manual_seed(11),
        guidance_action=torch.tensor([[-1.0, 0.0]]),
    )

    assert left.reference_prediction is not None
    assert right.reference_prediction is not None
    torch.testing.assert_close(
        left.prediction[..., 1] - left.reference_prediction[..., 1],
        -(right.prediction[..., 1] - right.reference_prediction[..., 1]),
        rtol=0.0,
        atol=0.0,
    )
    assert torch.all(left.prediction[:, 0, :, 1] > left.reference_prediction[:, 0, :, 1])
    assert left.guidance_diagnostics is not None
    assert left.guidance_diagnostics.applied_gradient_l2.shape == (1, 5)
    assert left_model.encode_calls == 1
    assert right_model.encode_calls == 1
    assert all(parameter.grad is None for parameter in left_planner.parameters())
    assert all(parameter.grad is None for parameter in right_planner.parameters())


@pytest.mark.parametrize("implementation", ("legacy", "diffusers"))
def test_guided_reference_consumes_one_shared_stochastic_ddim_stream(
    stage0_observation: dict[str, torch.Tensor],
    implementation: str,
) -> None:
    config = SimpleNamespace(
        predicted_neighbor_num=10,
        future_len=80,
        observation_normalizer=lambda observation: dict(observation),
        state_normalizer=_IdentityStateNormalizer(),
    )
    sampler_config = Ddim5SamplerConfig(
        name="ddim5",
        num_steps=5,
        timesteps=(1.0, 0.8, 0.6, 0.4, 0.2, 0.0),
        initial_noise_scale=1.0,
        ddim_stochasticity=1.0,
        parity_label="plannerrft_paper_text",
        implementation=implementation,  # type: ignore[arg-type]
    )
    noise = torch.zeros((1, 11, 80, 4), dtype=torch.float32)
    noise[..., 0] = torch.arange(1, 81, dtype=torch.float32)
    noise[..., 2] = 1.0
    unguided = PretrainedDiffusionPlanner(  # type: ignore[arg-type]
        config, _IdentityDenoiser(), sampler_config
    )
    guided = PretrainedDiffusionPlanner(  # type: ignore[arg-type]
        config, _IdentityDenoiser(), sampler_config, _guidance_config()
    )
    unguided_generator = torch.Generator().manual_seed(37)
    guided_generator = torch.Generator().manual_seed(37)

    baseline = unguided(stage0_observation, noise, unguided_generator)
    result = guided(
        stage0_observation,
        noise,
        guided_generator,
        torch.tensor([[1.0, 0.0]]),
    )

    assert result.reference_prediction is not None
    assert torch.equal(result.reference_prediction, baseline.prediction)
    assert torch.equal(guided_generator.get_state(), unguided_generator.get_state())
