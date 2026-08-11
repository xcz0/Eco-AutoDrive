from __future__ import annotations

import math

import pytest
import torch

from eco_planner.models.guidance import (
    OrthogonalGuidance,
    OrthogonalReferenceGuidanceConfig,
)
from eco_planner.models.normalization import StateNormalizer


def _config() -> OrthogonalReferenceGuidanceConfig:
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


def _identity_normalizer() -> StateNormalizer:
    return StateNormalizer(
        torch.zeros((11, 1, 4)),
        torch.ones((11, 1, 4)),
    )


def _straight_reference() -> tuple[torch.Tensor, torch.Tensor]:
    current = torch.zeros((1, 11, 4), dtype=torch.float32)
    current[..., 2] = 1.0
    reference = torch.zeros((1, 11, 80, 4), dtype=torch.float32)
    reference[..., 0] = torch.arange(1, 81, dtype=torch.float32)
    reference[..., 2] = 1.0
    return current, reference


def test_centered_lateral_guidance_has_exact_neutral_and_opposite_signed_gradients() -> None:
    current, reference = _straight_reference()
    state = torch.cat([current[:, :, None], reference], dim=2).reshape(1, 11, -1)
    sample = state.clone().requires_grad_(True)
    guidance = OrthogonalGuidance(_config(), _identity_normalizer())

    neutral = guidance.gradient(sample, sample, reference, current, torch.zeros((1, 2)))
    positive = guidance.gradient(sample, sample, reference, current, torch.tensor([[1.0, 0.0]]))
    negative = guidance.gradient(sample, sample, reference, current, torch.tensor([[-1.0, 0.0]]))

    assert torch.count_nonzero(neutral.applied_gradient) == 0
    torch.testing.assert_close(
        positive.applied_gradient,
        -negative.applied_gradient,
        rtol=0.0,
        atol=0.0,
    )
    expected_y_gradient = torch.full((80,), -0.0625)
    torch.testing.assert_close(
        positive.applied_gradient.reshape(1, 11, 81, 4)[0, 0, 1:, 1],
        expected_y_gradient,
        rtol=0.0,
        atol=0.0,
    )
    assert torch.count_nonzero(positive.applied_gradient.reshape(1, 11, 81, 4)[:, 1:]) == 0
    assert positive.raw_neighbor_gradient_l2.item() == 0.0


def test_centered_longitudinal_guidance_uses_ten_hz_velocity_and_opposite_signs() -> None:
    current, reference = _straight_reference()
    state = torch.cat([current[:, :, None], reference], dim=2).reshape(1, 11, -1)
    sample = state.clone().requires_grad_(True)
    guidance = OrthogonalGuidance(_config(), _identity_normalizer())

    positive = guidance.gradient(sample, sample, reference, current, torch.tensor([[0.0, 1.0]]))
    negative = guidance.gradient(sample, sample, reference, current, torch.tensor([[0.0, -1.0]]))

    torch.testing.assert_close(
        positive.applied_gradient,
        -negative.applied_gradient,
        rtol=0.0,
        atol=0.0,
    )
    gradient = positive.applied_gradient.reshape(1, 11, 81, 4)
    assert gradient[0, 0, -1, 0].item() == -0.625
    assert torch.count_nonzero(gradient[0, 0, 1:-1, 0]) == 0
    assert positive.zero_speed_count.item() == 0
    target = guidance.longitudinal_target_speed_delta_mps(
        reference,
        current,
        torch.tensor([[0.0, 1.0]]),
    )
    torch.testing.assert_close(target, torch.full((1, 80), 2.5), rtol=0.0, atol=0.0)


def test_guidance_is_rigid_transform_equivariant() -> None:
    current, reference = _straight_reference()
    predicted = reference.clone()
    predicted[:, 0, :, 1] = 0.5
    state = torch.cat([current[:, :, None], predicted], dim=2)
    guidance = OrthogonalGuidance(_config(), _identity_normalizer())
    original_sample = state.reshape(1, 11, -1).clone().requires_grad_(True)
    original = guidance.gradient(
        original_sample,
        original_sample,
        reference,
        current,
        torch.tensor([[1.0, 0.0]]),
    )

    angle = math.pi / 3.0
    rotation = torch.tensor(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]],
        dtype=torch.float32,
    )
    translation = torch.tensor([13.0, -7.0])

    def transform(value: torch.Tensor) -> torch.Tensor:
        result = value.clone()
        result[..., :2] = value[..., :2] @ rotation.T + translation
        result[..., 2:4] = value[..., 2:4] @ rotation.T
        return result

    transformed_current = transform(current)
    transformed_reference = transform(reference)
    transformed_predicted = transform(predicted)
    transformed_state = torch.cat([transformed_current[:, :, None], transformed_predicted], dim=2)
    transformed_sample = transformed_state.reshape(1, 11, -1).requires_grad_(True)
    transformed = guidance.gradient(
        transformed_sample,
        transformed_sample,
        transformed_reference,
        transformed_current,
        torch.tensor([[1.0, 0.0]]),
    )

    original_xy = original.applied_gradient.reshape(1, 11, 81, 4)[..., :2]
    transformed_xy = transformed.applied_gradient.reshape(1, 11, 81, 4)[..., :2]
    torch.testing.assert_close(transformed_xy, original_xy @ rotation.T, atol=1e-6, rtol=0.0)
    torch.testing.assert_close(
        transformed.lateral_objective_delta,
        original.lateral_objective_delta,
        atol=5e-6,
        rtol=0.0,
    )


def test_zero_speed_reference_is_audited_without_longitudinal_fallback() -> None:
    current, reference = _straight_reference()
    reference[..., 0] = 0.0
    state = torch.cat([current[:, :, None], reference], dim=2).reshape(1, 11, -1)
    sample = state.clone().requires_grad_(True)

    result = OrthogonalGuidance(_config(), _identity_normalizer()).gradient(
        sample,
        sample,
        reference,
        current,
        torch.tensor([[0.0, 1.0]]),
    )

    assert result.zero_speed_count.item() == 80
    assert result.longitudinal_objective_delta.item() == 0.0
    assert torch.count_nonzero(result.applied_gradient) == 0


def test_guidance_rejects_degenerate_heading_and_invalid_action() -> None:
    current, reference = _straight_reference()
    state = torch.cat([current[:, :, None], reference], dim=2).reshape(1, 11, -1)
    guidance = OrthogonalGuidance(_config(), _identity_normalizer())

    bad_reference = reference.clone()
    bad_reference[:, 0, 3, 2:4] = 0.0
    with pytest.raises(ValueError, match="heading"):
        guidance.gradient(
            state.clone().requires_grad_(True),
            state.clone().requires_grad_(True),
            bad_reference,
            current,
            torch.zeros((1, 2)),
        )
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        sample = state.clone().requires_grad_(True)
        guidance.gradient(
            sample,
            sample,
            reference,
            current,
            torch.tensor([[1.01, 0.0]]),
        )
