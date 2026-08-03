from __future__ import annotations

import pytest
import torch

from eco_planner.models.pretrained import CheckpointLoadReport, PretrainedDiffusionPlanner
from eco_planner.models.synthetic import make_stage0_observation


@pytest.mark.slow
def test_stage0_planner_rejects_invalid_observation(
    stage0_planner: tuple[PretrainedDiffusionPlanner, CheckpointLoadReport],
) -> None:
    planner, _ = stage0_planner
    device = torch.device("cpu")
    observation = make_stage0_observation(device)
    observation["ego_current_state"] = torch.zeros((1, 9), dtype=torch.float32, device=device)
    noise = torch.zeros((1, 11, 80, 4), dtype=torch.float32, device=device)

    with pytest.raises(ValueError, match="shape"):
        planner.predict(observation, noise)


@pytest.mark.slow
def test_stage0_planner_rejects_nonfinite_noise(
    stage0_planner: tuple[PretrainedDiffusionPlanner, CheckpointLoadReport],
) -> None:
    planner, _ = stage0_planner
    device = torch.device("cpu")
    observation = make_stage0_observation(device)
    noise = torch.zeros((1, 11, 80, 4), dtype=torch.float32, device=device)
    noise[0, 0, 0, 0] = torch.nan

    with pytest.raises(ValueError, match="finite"):
        planner.predict(observation, noise)
