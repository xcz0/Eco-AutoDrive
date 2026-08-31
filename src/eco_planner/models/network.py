"""Diffusion Planner network architecture."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import cast

import torch
from timm.layers import (
    DropPath,  # pyright: ignore[reportPrivateImportUsage]
    Mlp,  # pyright: ignore[reportPrivateImportUsage]
)
from torch import nn

from eco_planner.models.config import OfficialDiffusionPlannerConfig


class MixerBlock(nn.Module):
    def __init__(self, tokens_mlp_dim: int, channels_mlp_dim: int, drop_path_rate: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(channels_mlp_dim)
        self.channels_mlp = Mlp(
            in_features=channels_mlp_dim,
            hidden_features=channels_mlp_dim,
            act_layer=nn.GELU,
            drop=drop_path_rate,
        )
        self.norm2 = nn.LayerNorm(channels_mlp_dim)
        self.tokens_mlp = Mlp(
            in_features=tokens_mlp_dim,
            hidden_features=tokens_mlp_dim,
            act_layer=nn.GELU,
            drop=drop_path_rate,
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        update = self.norm1(value).permute(0, 2, 1)
        update = self.tokens_mlp(update).permute(0, 2, 1)
        value = value + update
        return value + self.channels_mlp(self.norm2(value))


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
        result = encoded.new_zeros((batch * participants, encoded.shape[-1]))
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
        projected = self.projection(value.view(batch * participants, -1))
        result = torch.where(valid[:, None], projected, torch.zeros_like(projected))
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
        known_speed_embedding = self.speed_limit_emb(flat_limit.unsqueeze(-1))
        unknown_speed_embedding = self.unknown_speed_emb.weight.expand(flat_limit.shape[0], -1)
        speed_embedding = torch.where(
            flat_has_limit[:, None], known_speed_embedding, unknown_speed_embedding
        )
        encoded = (
            encoded
            + speed_embedding
            + self.traffic_emb(traffic.view(batch * participants, -1)[valid])
        )
        encoded = self.emb_project(self.norm(encoded))
        result = encoded.new_zeros((batch * participants, encoded.shape[-1]))
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
        position_result = projected.new_zeros((batch * self.token_num, self.hidden_dim))
        position_result[~mask] = projected
        padding_mask = mask.view(batch, -1)
        encoding = self.fusion(
            value + position_result.view(batch, self.token_num, -1), padding_mask
        )
        return {"encoding": encoding, "padding_mask": padding_mask}


def _modulate(value: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return value * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    frequencies: torch.Tensor

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size
        half = frequency_embedding_size // 2
        frequencies = torch.exp(
            -math.log(10000) * torch.arange(0, half, dtype=torch.float32) / half
        )
        self.register_buffer("frequencies", frequencies, persistent=False)

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        angles = timestep[:, None].float() * self.frequencies[None]
        embedding = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)
        return self.mlp(embedding)


class _ApproximateGELU(nn.GELU):
    def __init__(self) -> None:
        super().__init__(approximate="tanh")


class DiTBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp1 = Mlp(dim, int(dim * mlp_ratio), act_layer=_ApproximateGELU, drop=0.0)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        self.norm3 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.norm4 = nn.LayerNorm(dim)
        self.mlp2 = Mlp(dim, int(dim * mlp_ratio), act_layer=_ApproximateGELU, drop=0.0)

    def forward(
        self,
        value: torch.Tensor,
        cross_context: torch.Tensor,
        condition: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(
            condition
        ).chunk(6, dim=1)
        modulated = _modulate(self.norm1(value), shift_msa, scale_msa)
        value = (
            value
            + gate_msa.unsqueeze(1)
            * self.attn(modulated, modulated, modulated, key_padding_mask=mask)[0]
        )
        value = value + gate_mlp.unsqueeze(1) * self.mlp1(
            _modulate(self.norm2(value), shift_mlp, scale_mlp)
        )
        value = self.cross_attn(self.norm3(value), cross_context, cross_context)[0]
        return self.mlp2(self.norm4(value))


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size)
        self.proj = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(approximate="tanh"),
            nn.LayerNorm(hidden_size * 4),
            nn.Linear(hidden_size * 4, output_size),
        )
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))

    def forward(self, value: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(condition).chunk(2, dim=1)
        return self.proj(_modulate(self.norm_final(value), shift, scale))


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
        route_encoding: torch.Tensor,
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
        condition = route_encoding + self.t_embedder(timestep)
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
        route_encoding: torch.Tensor,
        current_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.dit(sample, timestep, encoding, route_encoding, current_mask)


def initialize_module(module: nn.Module) -> None:
    """Apply the official initializer for standard PyTorch layers."""

    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.bias, 0)
        nn.init.constant_(module.weight, 1.0)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)


class DiffusionPlanner(nn.Module):
    """Checkpoint-backed scene encoder and trajectory denoiser."""

    def __init__(self, config: OfficialDiffusionPlannerConfig) -> None:
        super().__init__()
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.encoder.apply(initialize_module)
        self.decoder.apply(initialize_module)
        nn.init.normal_(self.encoder.pos_emb.weight, std=0.02)
        nn.init.normal_(self.encoder.neighbor_encoder.type_emb.weight, std=0.02)
        nn.init.normal_(self.encoder.lane_encoder.speed_limit_emb.weight, std=0.02)
        nn.init.normal_(self.encoder.lane_encoder.traffic_emb.weight, std=0.02)
        timestep_input = cast(nn.Linear, self.decoder.dit.t_embedder.mlp[0])
        timestep_output = cast(nn.Linear, self.decoder.dit.t_embedder.mlp[2])
        nn.init.normal_(timestep_input.weight, std=0.02)
        nn.init.normal_(timestep_output.weight, std=0.02)
        for block in self.decoder.dit.blocks:
            modulation = cast(nn.Linear, cast(DiTBlock, block).adaLN_modulation[-1])
            nn.init.constant_(modulation.weight, 0)
            nn.init.constant_(modulation.bias, 0)
        final_modulation = cast(nn.Linear, self.decoder.dit.final_layer.adaLN_modulation[-1])
        final_projection = cast(nn.Linear, self.decoder.dit.final_layer.proj[-1])
        nn.init.constant_(final_modulation.weight, 0)
        nn.init.constant_(final_modulation.bias, 0)
        nn.init.constant_(final_projection.weight, 0)
        nn.init.constant_(final_projection.bias, 0)

    def encode(self, inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return self.encoder(inputs)["encoding"]

    def encode_route(self, inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return self.decoder.dit.route_encoder(inputs["route_lanes"])

    def encode_policy_features(self, inputs: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        scene = self.encoder(inputs)
        navigation, navigation_padding_mask = self.decoder.dit.route_encoder.encode_with_mask(
            inputs["route_lanes"]
        )
        return {
            "scene_tokens": scene["encoding"],
            "scene_padding_mask": scene["padding_mask"],
            "route_encoding": navigation,
            "navigation_tokens": navigation[:, None, :],
            "navigation_padding_mask": navigation_padding_mask[:, None],
        }

    def denoise(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        encoding: torch.Tensor,
        route_encoding: torch.Tensor,
        current_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.decoder.denoise(sample, timestep, encoding, route_encoding, current_mask)
