from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt


class _ArtifactModel(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid", allow_inf_nan=False)


class PolicyEvaluationSummary(_ArtifactModel):
    """Acceptance inputs projected from one ordinary generic evaluation job."""

    checkpoint_label: Literal["initial", "final"]
    checkpoint_path: str
    policy_hash: str = Field(min_length=64, max_length=64)
    evaluation_seed: StrictInt = Field(ge=0)
    scenarios: tuple[str, ...]
    noise_seeds: tuple[StrictInt, ...]
    transition_count: StrictInt = Field(gt=0)
    episode_count: StrictInt = Field(gt=0)
    mean_episode_length: StrictFloat = Field(gt=0.0)
    collision_count: StrictInt = Field(ge=0)
    out_of_road_count: StrictInt = Field(ge=0)
    route_completion_delta: StrictFloat
    distance_m: StrictFloat = Field(ge=0.0)
    mean_speed_mps: StrictFloat = Field(ge=0.0)
    stopped_fraction: StrictFloat = Field(ge=0.0, le=1.0)


class PolicyEvaluationComparison(_ArtifactModel):
    initial: PolicyEvaluationSummary
    final: PolicyEvaluationSummary
    episode_length_retention: StrictFloat = Field(ge=0.0)
    route_progress_retention: StrictFloat = Field(ge=0.0)
    collision_count_not_increased: bool
    out_of_road_count_not_increased: bool
    passed: bool


def compare_policy_evaluations(
    initial: PolicyEvaluationSummary,
    final: PolicyEvaluationSummary,
    *,
    minimum_retention: float = 0.9,
) -> PolicyEvaluationComparison:
    """Apply PPO's retention and safety gates; generic execution owns all metrics."""

    if (
        initial.evaluation_seed != final.evaluation_seed
        or initial.scenarios != final.scenarios
        or initial.noise_seeds != final.noise_seeds
    ):
        raise ValueError("policy evaluations must use identical scenarios and diffusion seeds")
    if initial.route_completion_delta <= 0.0:
        raise ValueError("initial policy evaluation must make positive route progress")
    episode_retention = final.mean_episode_length / initial.mean_episode_length
    route_retention = final.route_completion_delta / initial.route_completion_delta
    collision_ok = final.collision_count <= initial.collision_count
    out_of_road_ok = final.out_of_road_count <= initial.out_of_road_count
    return PolicyEvaluationComparison(
        initial=initial,
        final=final,
        episode_length_retention=float(episode_retention),
        route_progress_retention=float(route_retention),
        collision_count_not_increased=collision_ok,
        out_of_road_count_not_increased=out_of_road_ok,
        passed=(
            collision_ok
            and out_of_road_ok
            and episode_retention >= minimum_retention
            and route_retention >= minimum_retention
        ),
    )
