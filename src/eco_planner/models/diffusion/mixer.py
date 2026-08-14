"""MLP-Mixer building blocks shared by the planner encoders."""

from __future__ import annotations

import torch
from timm.layers import Mlp
from torch import nn


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
