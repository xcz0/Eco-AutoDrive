from __future__ import annotations

import math
from unittest.mock import patch

import torch
from torch import nn

from eco_planner.models.config import OfficialDiffusionPlannerConfig
from eco_planner.models.network import DiffusionPlanner, TimestepEmbedder


def test_timestep_embedder_caches_frequency_basis_without_checkpoint_state() -> None:
    embedder = TimestepEmbedder(hidden_size=6, frequency_embedding_size=6)
    embedder.mlp = nn.Identity()
    checkpoint_embedder = TimestepEmbedder(hidden_size=6, frequency_embedding_size=6)
    timestep = torch.tensor([0, 1, 7], dtype=torch.int64)
    expected_frequencies = torch.exp(-math.log(10000) * torch.arange(0, 3, dtype=torch.float32) / 3)
    expected = torch.cat(
        [
            torch.cos(timestep[:, None].float() * expected_frequencies[None]),
            torch.sin(timestep[:, None].float() * expected_frequencies[None]),
        ],
        dim=-1,
    )

    with (
        patch("eco_planner.models.network.torch.arange") as arange,
        patch("eco_planner.models.network.torch.exp") as exp,
    ):
        first = embedder(timestep)
        cached_basis = embedder.frequencies.data_ptr()
        second = embedder(timestep)

    torch.testing.assert_close(first, expected)
    torch.testing.assert_close(second, expected)
    assert embedder.frequencies.data_ptr() == cached_basis
    arange.assert_not_called()
    exp.assert_not_called()
    checkpoint = checkpoint_embedder.state_dict()
    assert "frequencies" not in checkpoint
    TimestepEmbedder(hidden_size=6, frequency_embedding_size=6).load_state_dict(
        checkpoint, strict=True
    )


def test_model_hierarchy_matches_official_checkpoint(
    official_model_config: OfficialDiffusionPlannerConfig,
) -> None:
    model = DiffusionPlanner(official_model_config)
    state_dict = model.state_dict()

    assert len(state_dict) == 276
    assert sum(value.numel() for value in state_dict.values()) == 6_042_628
    assert "encoder.neighbor_encoder.type_emb.weight" in state_dict
    assert "decoder.dit.route_encoder.Mixer.norm1.weight" in state_dict
    assert "decoder.dit.final_layer.proj.4.weight" in state_dict


def test_model_encode_and_denoise_shapes(
    official_model_config: OfficialDiffusionPlannerConfig,
    baseline_observation: dict[str, torch.Tensor],
) -> None:
    model = DiffusionPlanner(official_model_config).eval()
    inputs = official_model_config.observation_normalizer(baseline_observation)

    with torch.no_grad():
        encoding = model.encode(inputs)
        route_encoding = model.encode_route(inputs)
        prediction = model.denoise(
            torch.zeros((1, 11, 324)),
            torch.full((1,), 0.5),
            encoding,
            route_encoding,
            torch.zeros((1, 10), dtype=torch.bool),
        )

    assert tuple(encoding.shape) == (1, 107, 192)
    assert tuple(prediction.shape) == (1, 11, 324)
    assert torch.isfinite(prediction).all()
