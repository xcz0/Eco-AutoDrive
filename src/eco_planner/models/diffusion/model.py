"""Diffusion Planner network."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from eco_planner.models.checkpoint.config import OfficialDiffusionPlannerConfig
from eco_planner.models.diffusion.decoder import Decoder
from eco_planner.models.diffusion.encoder import Encoder
from eco_planner.models.diffusion.initialization import initialize_module


class DiffusionPlannerEncoder(nn.Module):
    def __init__(self, config: OfficialDiffusionPlannerConfig) -> None:
        super().__init__()
        self.encoder = Encoder(config)
        self.apply(initialize_module)
        nn.init.normal_(self.encoder.pos_emb.weight, std=0.02)
        nn.init.normal_(self.encoder.neighbor_encoder.type_emb.weight, std=0.02)
        nn.init.normal_(self.encoder.lane_encoder.speed_limit_emb.weight, std=0.02)
        nn.init.normal_(self.encoder.lane_encoder.traffic_emb.weight, std=0.02)

    def forward(self, inputs: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return self.encoder(inputs)


class DiffusionPlannerDecoder(nn.Module):
    def __init__(self, config: OfficialDiffusionPlannerConfig) -> None:
        super().__init__()
        self.decoder = Decoder(config)
        self.apply(initialize_module)
        nn.init.normal_(self.decoder.dit.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.decoder.dit.t_embedder.mlp[2].weight, std=0.02)
        for block in self.decoder.dit.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.decoder.dit.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.decoder.dit.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.decoder.dit.final_layer.proj[-1].weight, 0)
        nn.init.constant_(self.decoder.dit.final_layer.proj[-1].bias, 0)


class DiffusionPlanner(nn.Module):
    """Model wrapper whose state-dict hierarchy matches the official implementation."""

    def __init__(self, config: OfficialDiffusionPlannerConfig) -> None:
        super().__init__()
        self.encoder = DiffusionPlannerEncoder(config)
        self.decoder = DiffusionPlannerDecoder(config)

    def encode(self, inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return self.encoder(inputs)["encoding"]

    def encode_policy_features(self, inputs: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Encode frozen scene and navigation context once for an exploration policy."""

        scene = self.encoder(inputs)
        navigation, navigation_padding_mask = (
            self.decoder.decoder.dit.route_encoder.encode_with_mask(inputs["route_lanes"])
        )
        return {
            "scene_tokens": scene["encoding"],
            "scene_padding_mask": scene["padding_mask"],
            "navigation_tokens": navigation[:, None, :],
            "navigation_padding_mask": navigation_padding_mask[:, None],
        }

    def denoise(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        encoding: torch.Tensor,
        route_lanes: torch.Tensor,
        current_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.decoder.decoder.denoise(sample, timestep, encoding, route_lanes, current_mask)
