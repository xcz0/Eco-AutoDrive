"""Policy input adapter and neutral diffusion representation contract."""

from __future__ import annotations

from dataclasses import fields

import torch

from eco_planner.planning import diffusion
from eco_planner.planning.diffusion import DiffusionRepresentations
from eco_planner.planning.diffusion.planner import PreparedPrediction
from eco_planner.planning.policy import ExplorationPolicyContext, build_policy_inputs


def _representations() -> DiffusionRepresentations:
    return DiffusionRepresentations(
        scene_tokens=torch.arange(8, dtype=torch.float32).reshape(1, 2, 4),
        scene_padding_mask=torch.tensor([[False, True]]),
        navigation_tokens=torch.arange(8, 16, dtype=torch.float32).reshape(1, 2, 4),
        navigation_padding_mask=torch.tensor([[True, False]]),
        route_encoding=torch.arange(4, dtype=torch.float32).reshape(1, 4),
    )


def test_build_policy_inputs_maps_each_field_verbatim() -> None:
    representations = _representations()
    reference_prediction = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)

    inputs = build_policy_inputs(representations, reference_prediction)

    assert isinstance(inputs, ExplorationPolicyContext)
    assert inputs.scene_tokens is representations.scene_tokens
    assert inputs.scene_padding_mask is representations.scene_padding_mask
    assert inputs.navigation_tokens is representations.navigation_tokens
    assert inputs.navigation_padding_mask is representations.navigation_padding_mask
    assert torch.equal(inputs.reference_trajectory, reference_prediction[:, 0])


def test_diffusion_exposes_neutral_representations_without_policy_schema() -> None:
    assert {field.name for field in fields(DiffusionRepresentations)} == {
        "scene_tokens",
        "scene_padding_mask",
        "navigation_tokens",
        "navigation_padding_mask",
        "route_encoding",
    }
    assert not hasattr(diffusion, "PlannerPolicyContext")
    assert "representations" in {field.name for field in fields(PreparedPrediction)}
