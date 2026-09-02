"""Typed configuration owned by the MetaDrive backend."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictFloat


class _StrictMetaDriveConfig(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid", allow_inf_nan=False)


class MetaDriveBuiltinRewardConfig(_StrictMetaDriveConfig):
    """MetaDrive's native reward parameters, passed directly to the simulator."""

    name: Literal["metadrive_builtin_v1"]
    driving_reward: StrictFloat = Field(ge=0.0)
    speed_reward: StrictFloat = Field(ge=0.0)
    success_reward: StrictFloat = Field(ge=0.0)
    out_of_road_penalty: StrictFloat = Field(ge=0.0)
    crash_vehicle_penalty: StrictFloat = Field(ge=0.0)
    crash_object_penalty: StrictFloat = Field(ge=0.0)
    crash_sidewalk_penalty: StrictFloat = Field(ge=0.0)
    use_lateral_reward: StrictBool


__all__ = ["MetaDriveBuiltinRewardConfig"]
