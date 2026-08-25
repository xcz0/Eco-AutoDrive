"""Typed configuration for shared single-device runtimes."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt


class RuntimeConfig(BaseModel):
    """Requested accelerator, precision, and global random seed."""

    model_config = ConfigDict(
        strict=True,
        frozen=True,
        extra="forbid",
        arbitrary_types_allowed=True,
        allow_inf_nan=False,
    )

    accelerator: Literal["auto", "cpu", "cuda"]
    precision: Literal["auto", "32-true", "16-mixed", "bf16-mixed"]
    seed: StrictInt = Field(ge=0)
