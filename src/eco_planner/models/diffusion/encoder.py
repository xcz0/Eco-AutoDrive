"""Scene token encoders for the checkpoint-compatible Diffusion Planner."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from timm.layers import DropPath, Mlp
from torch import nn

from eco_planner.models.checkpoint.config import OfficialDiffusionPlannerConfig
from eco_planner.models.diffusion.mixer import MixerBlock


class AgentFusionEncoder(nn.Module):
    def __init__(self, config: OfficialDiffusionPlannerConfig) -> None:
        super().__init__()
        self._channel = 128
        self.type_emb = nn.Linear(3, self._channel)
        self.channel_pre_project = Mlp(9, self._channel, self._channel, act_layer=nn.GELU, drop=0.0)
        self.token_pre_project = Mlp(config.time_len, 64, 64, act_layer=nn.GELU, drop=0.0)
        self.blocks = nn.ModuleList(
            [
                MixerBlock(64, self._channel, config.encoder_drop_path_rate)
                for _ in range(config.encoder_depth)
            ]
        )
        self.norm = nn.LayerNorm(self._channel)
        self.emb_project = Mlp(
            self._channel,
            config.hidden_dim,
            config.hidden_dim,
            act_layer=nn.GELU,
            drop=config.encoder_drop_path_rate,
        )

    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        agent_type = value[:, :, -1, 8:]
        states = value[..., :8]
        position = states[:, :, -1, :7].clone()
        position[..., -3:] = 0.0
        position[..., -3] = 1.0
        batch, participants, horizon, _ = states.shape
        mask_time = torch.sum(torch.ne(states, 0), dim=-1) == 0
        mask_participant = torch.sum(~mask_time, dim=-1) == 0
        states = torch.cat([states, (~mask_time).float().unsqueeze(-1)], dim=-1).view(
            batch * participants, horizon, -1
        )
        valid = ~mask_participant.view(-1)
        encoded = states[valid]
        encoded = self.channel_pre_project(encoded).permute(0, 2, 1)
        encoded = self.token_pre_project(encoded).permute(0, 2, 1)
        for block in self.blocks:
            encoded = block(encoded)
        encoded = torch.mean(encoded, dim=1)
        encoded = encoded + self.type_emb(agent_type.view(batch * participants, -1)[valid])
        encoded = self.emb_project(self.norm(encoded))
        result = torch.zeros((batch * participants, encoded.shape[-1]), device=value.device)
        result[valid] = encoded
        return result.view(batch, participants, -1), mask_participant, position


class StaticFusionEncoder(nn.Module):
    def __init__(self, config: OfficialDiffusionPlannerConfig) -> None:
        super().__init__()
        self._hidden_dim = config.hidden_dim
        self.projection = Mlp(
            config.static_objects_state_dim,
            config.hidden_dim,
            config.hidden_dim,
            act_layer=nn.GELU,
            drop=config.encoder_drop_path_rate,
        )

    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, participants, _ = value.shape
        position = value[:, :, :7].clone()
        position[..., -3:] = 0.0
        position[..., -2] = 1.0
        mask = torch.sum(torch.ne(value[..., :10], 0), dim=-1) == 0
        valid = ~mask.view(-1)
        result = torch.zeros((batch * participants, self._hidden_dim), device=value.device)
        if valid.any():
            result[valid] = self.projection(value.view(batch * participants, -1)[valid])
        return result.view(batch, participants, -1), mask, position


class LaneFusionEncoder(nn.Module):
    def __init__(self, config: OfficialDiffusionPlannerConfig) -> None:
        super().__init__()
        self._lane_len = config.lane_len
        self._channel = 128
        self.speed_limit_emb = nn.Linear(1, self._channel)
        self.unknown_speed_emb = nn.Embedding(1, self._channel)
        self.traffic_emb = nn.Linear(4, self._channel)
        self.channel_pre_project = Mlp(8, self._channel, self._channel, act_layer=nn.GELU, drop=0.0)
        self.token_pre_project = Mlp(config.lane_len, 64, 64, act_layer=nn.GELU, drop=0.0)
        self.blocks = nn.ModuleList(
            [
                MixerBlock(64, self._channel, config.encoder_drop_path_rate)
                for _ in range(config.encoder_depth)
            ]
        )
        self.norm = nn.LayerNorm(self._channel)
        self.emb_project = Mlp(
            self._channel,
            config.hidden_dim,
            config.hidden_dim,
            act_layer=nn.GELU,
            drop=config.encoder_drop_path_rate,
        )

    def forward(
        self, value: torch.Tensor, speed_limit: torch.Tensor, has_speed_limit: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        traffic = value[:, :, 0, 8:]
        states = value[..., :8]
        position = states[:, :, self._lane_len // 2, :7].clone()
        heading = torch.atan2(position[..., 3], position[..., 2])
        position[..., 2] = torch.cos(heading)
        position[..., 3] = torch.sin(heading)
        position[..., -3:] = 0.0
        position[..., -1] = 1.0
        batch, participants, horizon, _ = states.shape
        mask_time = torch.sum(torch.ne(states, 0), dim=-1) == 0
        mask_participant = torch.sum(~mask_time, dim=-1) == 0
        valid = ~mask_participant.view(-1)
        encoded = states.view(batch * participants, horizon, -1)[valid]
        encoded = self.channel_pre_project(encoded).permute(0, 2, 1)
        encoded = self.token_pre_project(encoded).permute(0, 2, 1)
        for block in self.blocks:
            encoded = block(encoded)
        encoded = torch.mean(encoded, dim=1)
        flat_has_limit = has_speed_limit.view(batch * participants, 1)[valid].squeeze(-1)
        flat_limit = speed_limit.view(batch * participants, 1)[valid].squeeze(-1)
        speed_embedding = torch.zeros((flat_limit.shape[0], self._channel), device=value.device)
        if flat_has_limit.any():
            speed_embedding[flat_has_limit] = self.speed_limit_emb(
                flat_limit[flat_has_limit].unsqueeze(-1)
            )
        if (~flat_has_limit).any():
            speed_embedding[~flat_has_limit] = self.unknown_speed_emb.weight.expand(
                (~flat_has_limit).sum().item(), -1
            )
        encoded = (
            encoded
            + speed_embedding
            + self.traffic_emb(traffic.view(batch * participants, -1)[valid])
        )
        encoded = self.emb_project(self.norm(encoded))
        result = torch.zeros((batch * participants, encoded.shape[-1]), device=value.device)
        result[valid] = encoded
        return result.view(batch, participants, -1), mask_participant, position


class SelfAttentionBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.drop_path = DropPath(dropout) if dropout > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * 4.0), act_layer=nn.GELU, drop=dropout)

    def forward(self, value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        value = value + self.drop_path(
            self.attn(self.norm1(value), value, value, key_padding_mask=mask)[0]
        )
        return value + self.drop_path(self.mlp(self.norm2(value)))


class FusionEncoder(nn.Module):
    def __init__(self, config: OfficialDiffusionPlannerConfig) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                SelfAttentionBlock(
                    config.hidden_dim, config.num_heads, config.encoder_drop_path_rate
                )
                for _ in range(config.encoder_depth)
            ]
        )
        self.norm = nn.LayerNorm(config.hidden_dim)

    def forward(self, value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask[:, 0] = False
        for block in self.blocks:
            value = block(value, mask)
        return self.norm(value)


class Encoder(nn.Module):
    def __init__(self, config: OfficialDiffusionPlannerConfig) -> None:
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.token_num = config.agent_num + config.static_objects_num + config.lane_num
        self.neighbor_encoder = AgentFusionEncoder(config)
        self.static_encoder = StaticFusionEncoder(config)
        self.lane_encoder = LaneFusionEncoder(config)
        self.fusion = FusionEncoder(config)
        self.pos_emb = nn.Linear(7, config.hidden_dim)

    def forward(self, inputs: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        batch = inputs["neighbor_agents_past"].shape[0]
        neighbors, neighbors_mask, neighbors_position = self.neighbor_encoder(
            inputs["neighbor_agents_past"]
        )
        static, static_mask, static_position = self.static_encoder(inputs["static_objects"])
        lanes, lanes_mask, lanes_position = self.lane_encoder(
            inputs["lanes"], inputs["lanes_speed_limit"], inputs["lanes_has_speed_limit"]
        )
        value = torch.cat([neighbors, static, lanes], dim=1)
        positions = torch.cat([neighbors_position, static_position, lanes_position], dim=1).view(
            batch * self.token_num, -1
        )
        mask = torch.cat([neighbors_mask, static_mask, lanes_mask], dim=1).view(-1)
        projected = self.pos_emb(positions[~mask])
        position_result = torch.zeros(
            (batch * self.token_num, self.hidden_dim), device=value.device
        )
        position_result[~mask] = projected
        return {
            "encoding": self.fusion(
                value + position_result.view(batch, self.token_num, -1), mask.view(batch, -1)
            )
        }
