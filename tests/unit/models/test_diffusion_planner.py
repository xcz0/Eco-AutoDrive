from __future__ import annotations

import torch

from eco_planner.models.checkpoint.config import OfficialDiffusionPlannerConfig
from eco_planner.models.checkpoint.loader import OFFICIAL_EMA_TENSOR_COUNT, OFFICIAL_PARAMETER_COUNT
from eco_planner.models.diffusion.model import DiffusionPlanner


def test_model_hierarchy_matches_official_checkpoint(
    official_model_config: OfficialDiffusionPlannerConfig,
) -> None:
    model = DiffusionPlanner(official_model_config)
    state_dict = model.state_dict()

    assert len(state_dict) == OFFICIAL_EMA_TENSOR_COUNT
    assert sum(value.numel() for value in state_dict.values()) == OFFICIAL_PARAMETER_COUNT
    assert "encoder.encoder.neighbor_encoder.type_emb.weight" in state_dict
    assert "decoder.decoder.dit.route_encoder.Mixer.norm1.weight" in state_dict
    assert "decoder.decoder.dit.final_layer.proj.4.weight" in state_dict


def test_model_encode_and_denoise_shapes(
    official_model_config: OfficialDiffusionPlannerConfig,
    stage0_observation: dict[str, torch.Tensor],
) -> None:
    model = DiffusionPlanner(official_model_config).eval()
    inputs = official_model_config.observation_normalizer(stage0_observation)

    with torch.no_grad():
        encoding = model.encode(inputs)
        prediction = model.denoise(
            torch.zeros((1, 11, 324)),
            torch.full((1,), 0.5),
            encoding,
            inputs["route_lanes"],
            torch.zeros((1, 10), dtype=torch.bool),
        )

    assert tuple(encoding.shape) == (1, 107, 192)
    assert tuple(prediction.shape) == (1, 11, 324)
    assert torch.isfinite(prediction).all()
