"""Diffusion Transformer blocks used by the trajectory decoder."""

from __future__ import annotations

import math

import torch
from timm.layers import Mlp
from torch import nn


def _modulate(value: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return value * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        half = self.frequency_embedding_size // 2
        frequencies = torch.exp(
            -math.log(10000)
            * torch.arange(0, half, dtype=torch.float32, device=timestep.device)
            / half
        )
        angles = timestep[:, None].float() * frequencies[None]
        embedding = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)
        return self.mlp(embedding)


class DiTBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp1 = Mlp(
            dim, int(dim * mlp_ratio), act_layer=lambda: nn.GELU(approximate="tanh"), drop=0.0
        )
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        self.norm3 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.norm4 = nn.LayerNorm(dim)
        self.mlp2 = Mlp(
            dim, int(dim * mlp_ratio), act_layer=lambda: nn.GELU(approximate="tanh"), drop=0.0
        )

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
