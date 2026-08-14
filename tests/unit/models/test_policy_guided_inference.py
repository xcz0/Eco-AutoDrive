from __future__ import annotations

from unittest.mock import patch

import torch

from eco_planner.models.config import Ddim5SamplerConfig, OrthogonalPolicyGuidanceConfig
from eco_planner.models.network import DiffusionPlanner
from eco_planner.models.planner import PretrainedDiffusionPlanner


def test_policy_guidance_prepares_one_frozen_encoding_and_reuses_the_reference(
    official_model_config, stage0_observation
) -> None:
    model = DiffusionPlanner(official_model_config)
    planner = PretrainedDiffusionPlanner(
        official_model_config,
        model,
        Ddim5SamplerConfig(
            name="ddim5",
            num_steps=5,
            timesteps=(1.0, 0.8, 0.6, 0.4, 0.2, 0.0),
            initial_noise_scale=1.0,
            ddim_stochasticity=0.0,
            parity_label="plannerrft_paper_text",
        ),
        OrthogonalPolicyGuidanceConfig(
            name="orthogonal_policy",
            formula_label="centered_energy_gradient_delta_v1",
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
        ),
    )
    before = {name: value.clone() for name, value in planner.model.state_dict().items()}
    noise = torch.randn((1, 11, 80, 4), generator=torch.Generator().manual_seed(7))
    generator = torch.Generator().manual_seed(8)

    with patch.object(
        planner.model, "encode_policy_features", wraps=planner.model.encode_policy_features
    ) as encode:
        prepared = planner.prepare_policy_guidance(stage0_observation, noise, generator)
        result = planner.complete_policy_guidance(
            prepared, torch.zeros((1, 2), dtype=torch.float32)
        )

    assert encode.call_count == 1
    assert prepared.policy_context.reference_trajectory.shape == (1, 80, 4)
    assert result.reference_prediction is not None
    assert result.guidance_action is not None
    torch.testing.assert_close(result.guidance_action, torch.zeros((1, 2)))
    assert all(parameter.grad is None for parameter in planner.model.parameters())
    for name, value in planner.model.state_dict().items():
        assert torch.equal(value, before[name])
