"""Versioned host-scheduling resource profiles resolved by Hydra."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, StrictInt


class ResourceProfileConfig(BaseModel):
    """Physical execution budgets, kept separate from experiment semantics."""

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid", allow_inf_nan=False)

    name: str = Field(min_length=1)
    rollout_worker_count: StrictInt = Field(gt=0)
    evaluation_job_worker_count: StrictInt = Field(gt=0)
    evaluation_vector_env_slots: StrictInt = Field(gt=0)
    torch_threads_per_worker: StrictInt = Field(gt=0)


def require_resource_profile(
    resources: ResourceProfileConfig | None,
) -> ResourceProfileConfig:
    """Require host scheduling budgets at an execution boundary."""

    if resources is None:
        raise ValueError(
            "execution requires a resource profile; set MACHINE_NAME or override "
            "components/resources=<profile>"
        )
    return resources
