"""Configuration for one update-0 batch collection."""

from pydantic import BaseModel, ConfigDict, Field, StrictInt


class CollectionConfig(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid", allow_inf_nan=False)
    protocol: str
    training_seed: StrictInt = Field(ge=0)
    overrides: list[str]
