from __future__ import annotations

import pytest
import torch

from eco_planner.models.pretrained import (
    CheckpointLoadReport,
    PretrainedDiffusionPlanner,
)


@pytest.mark.slow
def test_stage0_planner_rejects_invalid_observation(
    stage0_planner: tuple[PretrainedDiffusionPlanner, CheckpointLoadReport],
    stage0_observation: dict[str, torch.Tensor],
) -> None:
    planner, _ = stage0_planner
    device = torch.device("cpu")
    stage0_observation["ego_current_state"] = torch.zeros(
        (1, 9), dtype=torch.float32, device=device
    )
    noise = torch.zeros((1, 11, 80, 4), dtype=torch.float32, device=device)

    with pytest.raises(ValueError, match="shape"):
        planner(stage0_observation, noise)


@pytest.mark.slow
def test_stage0_planner_rejects_nonfinite_noise(
    stage0_planner: tuple[PretrainedDiffusionPlanner, CheckpointLoadReport],
    stage0_observation: dict[str, torch.Tensor],
) -> None:
    planner, _ = stage0_planner
    device = torch.device("cpu")
    noise = torch.zeros((1, 11, 80, 4), dtype=torch.float32, device=device)
    noise[0, 0, 0, 0] = torch.nan

    with pytest.raises(ValueError, match="finite"):
        planner(stage0_observation, noise)
