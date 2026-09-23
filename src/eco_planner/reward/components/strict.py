"""Shared strict pydantic base for reward component configuration schemas."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class StrictRewardModel(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid", allow_inf_nan=False)


__all__ = ["StrictRewardModel"]
