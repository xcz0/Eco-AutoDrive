"""Research-level model checks for the official planner boundary."""

from __future__ import annotations

import pytest
import torch

from eco_planner.models import Dpm10SamplerConfig, load_official_diffusion_planner


@pytest.mark.slow
def test_official_checkpoint_loads_and_generates_a_finite_trajectory(
    baseline_checkpoint_dir, baseline_observation: dict[str, torch.Tensor]
) -> None:
    planner, report = load_official_diffusion_planner(
        baseline_checkpoint_dir / "args.json",
        baseline_checkpoint_dir / "model.pth",
        Dpm10SamplerConfig(),
    )
    noise = torch.randn((1, 11, 80, 4), generator=torch.Generator().manual_seed(7))

    with torch.inference_mode():
        result = planner(baseline_observation, noise)

    assert report.ema_tensor_count > 0
    assert report.parameter_count > 0
    assert result.prediction.shape == (1, 11, 80, 4)
    assert torch.isfinite(result.prediction).all()
    assert not planner.training
    assert all(not parameter.requires_grad for parameter in planner.parameters())
