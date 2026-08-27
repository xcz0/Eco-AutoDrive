"""Guidance invariants over a frozen planner and fixed diffusion noise."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from eco_planner.models import (
    Ddim5SamplerConfig,
    NoGuidanceConfig,
    OrthogonalReferenceGuidanceConfig,
    PretrainedDiffusionPlanner,
)


class _IdentityStateNormalizer:
    def inverse(self, value: torch.Tensor) -> torch.Tensor:
        return value


class _IdentityDenoiser(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.ones(()))

    def encode(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.zeros((inputs["ego_current_state"].shape[0], 1, 1))

    def encode_route(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.zeros((inputs["ego_current_state"].shape[0], 1))

    def denoise(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        encoding: torch.Tensor,
        route_encoding: torch.Tensor,
        current_mask: torch.Tensor,
    ) -> torch.Tensor:
        return sample * self.anchor


def _planner_config() -> SimpleNamespace:
    return SimpleNamespace(
        predicted_neighbor_num=10,
        future_len=80,
        observation_normalizer=lambda observation: dict(observation),
        state_normalizer=_IdentityStateNormalizer(),
    )


def _sampler_config() -> Ddim5SamplerConfig:
    return Ddim5SamplerConfig(
        name="ddim5",
        num_steps=5,
        timesteps=(1.0, 0.8, 0.6, 0.4, 0.2, 0.0),
        initial_noise_scale=1.0,
        ddim_stochasticity=0.0,
        parity_label="plannerrft_paper_text",
    )


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


def _noise() -> torch.Tensor:
    noise = torch.zeros((1, 11, 80, 4), dtype=torch.float32)
    noise[..., 0] = torch.arange(1, 81, dtype=torch.float32)
    noise[..., 2] = 1.0
    return noise


def _planner(
    guidance: NoGuidanceConfig | OrthogonalReferenceGuidanceConfig,
) -> PretrainedDiffusionPlanner:
    return PretrainedDiffusionPlanner(  # type: ignore[arg-type]
        _planner_config(), _IdentityDenoiser(), _sampler_config(), guidance
    )


def test_zero_guidance_is_identical_to_the_unguided_reference(
    baseline_observation: dict[str, torch.Tensor],
) -> None:
    unguided = _planner(NoGuidanceConfig())(baseline_observation, _noise())
    guided = _planner(_guidance_config())(baseline_observation, _noise())

    assert guided.reference_prediction is not None
    assert guided.guidance_action is not None
    assert torch.equal(guided.guidance_action, torch.zeros((1, 2)))
    assert torch.equal(guided.prediction, guided.reference_prediction)
    assert torch.equal(guided.prediction, unguided.prediction)


def test_opposite_lateral_guidance_moves_trajectory_in_opposite_directions(
    baseline_observation: dict[str, torch.Tensor],
) -> None:
    left_planner = _planner(_guidance_config())
    left = left_planner(baseline_observation, _noise(), guidance_action=torch.tensor([[1.0, 0.0]]))
    right = _planner(_guidance_config())(
        baseline_observation, _noise(), guidance_action=torch.tensor([[-1.0, 0.0]])
    )

    assert left.reference_prediction is not None
    assert right.reference_prediction is not None
    assert torch.all(left.prediction[:, 0, :, 1] > left.reference_prediction[:, 0, :, 1])
    torch.testing.assert_close(
        left.prediction[..., 1] - left.reference_prediction[..., 1],
        -(right.prediction[..., 1] - right.reference_prediction[..., 1]),
        rtol=0.0,
        atol=0.0,
    )
    assert all(
        not parameter.requires_grad and parameter.grad is None
        for parameter in left_planner.parameters()
    )
