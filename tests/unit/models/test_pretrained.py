from __future__ import annotations

import pytest
import torch

from eco_planner.models.config import OfficialDiffusionPlannerConfig
from eco_planner.models.diffusion_planner import DiffusionPlanner
from eco_planner.models.pretrained import PretrainedDiffusionPlanner


def test_pretrained_planner_remains_frozen(
    official_model_config: OfficialDiffusionPlannerConfig,
) -> None:
    planner = PretrainedDiffusionPlanner(
        official_model_config, DiffusionPlanner(official_model_config), torch.device("cpu")
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
        official_model_config, DiffusionPlanner(official_model_config), torch.device("cpu")
    )
    planner.model.train()

    with pytest.raises(RuntimeError, match="remain in eval mode"):
        planner.predict({}, torch.empty(0))


def test_pretrained_planner_tracks_actual_runtime_device(
    official_model_config: OfficialDiffusionPlannerConfig,
) -> None:
    planner = PretrainedDiffusionPlanner(
        official_model_config, DiffusionPlanner(official_model_config), torch.device("cpu")
    )
    planner.to(torch.device("cpu"))
    assert planner._runtime_device == torch.device("cpu")


@pytest.mark.gpu
def test_pretrained_planner_tracks_cuda_device(
    official_model_config: OfficialDiffusionPlannerConfig,
) -> None:
    planner = PretrainedDiffusionPlanner(
        official_model_config, DiffusionPlanner(official_model_config), torch.device("cpu")
    )
    planner.to(torch.device("cuda"))
    assert planner._runtime_device.type == "cuda"
