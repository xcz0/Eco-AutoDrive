"""Route conditioning and trajectory decoder for Diffusion Planner."""

from __future__ import annotations

import torch
from timm.layers import Mlp
from torch import nn

from eco_planner.models.checkpoint.config import OfficialDiffusionPlannerConfig
from eco_planner.models.diffusion.dit import DiTBlock, FinalLayer, TimestepEmbedder
from eco_planner.models.diffusion.mixer import MixerBlock


class RouteEncoder(nn.Module):
    def __init__(self, config: OfficialDiffusionPlannerConfig) -> None:
        super().__init__()
        self._channel = 64
        self.channel_pre_project = Mlp(4, self._channel, self._channel, act_layer=nn.GELU, drop=0.0)
        self.token_pre_project = Mlp(
            config.route_num * config.lane_len, 32, 32, act_layer=nn.GELU, drop=0.0
        )
        self.Mixer = MixerBlock(32, self._channel, config.encoder_drop_path_rate)
        self.norm = nn.LayerNorm(self._channel)
        self.emb_project = Mlp(
            self._channel,
            config.hidden_dim,
            config.hidden_dim,
            act_layer=nn.GELU,
            drop=config.encoder_drop_path_rate,
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.encode_with_mask(value)[0]

    def encode_with_mask(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        states = value[..., :4]
        batch, routes, horizon, _ = states.shape
        mask_time = torch.sum(torch.ne(states, 0), dim=-1) == 0
        mask_route = torch.sum(~mask_time, dim=-1) == 0
        mask_batch = torch.sum(~mask_route, dim=-1) == 0
        states = states.view(batch, routes * horizon, -1)
        valid = ~mask_batch.view(-1)
        encoded = states[valid]
        encoded = self.channel_pre_project(encoded).permute(0, 2, 1)
        encoded = self.token_pre_project(encoded).permute(0, 2, 1)
        encoded = torch.mean(self.Mixer(encoded), dim=1)
        encoded = self.emb_project(self.norm(encoded))
        result = encoded.new_zeros((batch, encoded.shape[-1]))
        result[valid] = encoded
        return result, mask_batch


class DiT(nn.Module):
    def __init__(self, config: OfficialDiffusionPlannerConfig) -> None:
        super().__init__()
        output_dim = (config.future_len + 1) * 4
        self.route_encoder = RouteEncoder(config)
        self.agent_embedding = nn.Embedding(2, config.hidden_dim)
        self.preproj = Mlp(output_dim, 512, config.hidden_dim, act_layer=nn.GELU, drop=0.0)
        self.t_embedder = TimestepEmbedder(config.hidden_dim)
        self.blocks = nn.ModuleList(
            [
                DiTBlock(config.hidden_dim, config.num_heads, config.decoder_drop_path_rate)
                for _ in range(config.decoder_depth)
            ]
        )
        self.final_layer = FinalLayer(config.hidden_dim, output_dim)
        self._model_type = config.diffusion_model_type

    @property
    def model_type(self) -> str:
        return self._model_type

    def forward(
        self,
        value: torch.Tensor,
        timestep: torch.Tensor,
        cross_context: torch.Tensor,
        route_lanes: torch.Tensor,
        neighbor_current_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, participants, _ = value.shape
        value = self.preproj(value)
        embedding = torch.cat(
            [
                self.agent_embedding.weight[0][None],
                self.agent_embedding.weight[1][None].expand(participants - 1, -1),
            ],
            dim=0,
        )
        value = value + embedding[None].expand(batch, -1, -1)
        condition = self.route_encoder(route_lanes) + self.t_embedder(timestep)
        mask = torch.zeros((batch, participants), dtype=torch.bool, device=value.device)
        mask[:, 1:] = neighbor_current_mask
        for block in self.blocks:
            value = block(value, cross_context, condition, mask)
        return self.final_layer(value, condition)


class Decoder(nn.Module):
    def __init__(self, config: OfficialDiffusionPlannerConfig) -> None:
        super().__init__()
        self._predicted_neighbor_num = config.predicted_neighbor_num
        self._future_len = config.future_len
        self.dit = DiT(config)

    def denoise(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        encoding: torch.Tensor,
        route_lanes: torch.Tensor,
        current_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.dit(sample, timestep, encoding, route_lanes, current_mask)
