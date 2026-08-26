"""Typed reward profiles and pure PlannerRFT-style MetaDrive scoring."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Annotated, Literal, TypeAlias

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictFloat, model_validator

from eco_planner.envs.domain.traffic import TrafficFrame


class _StrictRewardModel(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid", allow_inf_nan=False)


class MetaDriveBuiltinRewardConfig(_StrictRewardModel):
    name: Literal["metadrive_builtin_v1"]
    driving_reward: StrictFloat = Field(ge=0.0)
    speed_reward: StrictFloat = Field(ge=0.0)
    success_reward: StrictFloat = Field(ge=0.0)
    out_of_road_penalty: StrictFloat = Field(ge=0.0)
    crash_vehicle_penalty: StrictFloat = Field(ge=0.0)
    crash_object_penalty: StrictFloat = Field(ge=0.0)
    crash_sidewalk_penalty: StrictFloat = Field(ge=0.0)
    use_lateral_reward: StrictBool


class RewardWeightsConfig(_StrictRewardModel):
    ttc: StrictFloat = Field(gt=0.0)
    progress: StrictFloat = Field(gt=0.0)
    comfort: StrictFloat = Field(gt=0.0)
    speed: StrictFloat = Field(gt=0.0)
    energy: StrictFloat = Field(gt=0.0)

    @property
    def total(self) -> float:
        return self.ttc + self.progress + self.comfort + self.speed + self.energy


class RewardGatesConfig(_StrictRewardModel):
    collision_vehicle: StrictBool
    collision_object: StrictBool
    collision_building: StrictBool
    collision_human: StrictBool
    collision_sidewalk: StrictBool
    wrong_direction_max_heading_error_rad: StrictFloat = Field(gt=0.0, le=math.pi)


class TTCRewardConfig(_StrictRewardModel):
    critical_ttc_s: StrictFloat = Field(ge=0.0)
    safe_ttc_s: StrictFloat = Field(gt=0.0)
    maximum_ttc_s: StrictFloat = Field(gt=0.0)
    minimum_closing_speed_mps: StrictFloat = Field(gt=0.0)
    lateral_margin_m: StrictFloat = Field(ge=0.0)
    longitudinal_margin_m: StrictFloat = Field(ge=0.0)

    @model_validator(mode="after")
    def validate_ttc_bounds(self) -> TTCRewardConfig:
        if self.safe_ttc_s <= self.critical_ttc_s:
            raise ValueError("ttc.safe_ttc_s must exceed critical_ttc_s")
        if self.maximum_ttc_s < self.safe_ttc_s:
            raise ValueError("ttc.maximum_ttc_s must cover safe_ttc_s")
        return self


class ProgressRewardConfig(_StrictRewardModel):
    full_score_delta_m: StrictFloat = Field(gt=0.0)


class ComfortRewardConfig(_StrictRewardModel):
    longitudinal_acceleration_limit_mps2: StrictFloat = Field(gt=0.0)
    lateral_acceleration_limit_mps2: StrictFloat = Field(gt=0.0)
    jerk_limit_mps3: StrictFloat = Field(gt=0.0)
    yaw_rate_limit_radps: StrictFloat = Field(gt=0.0)


class SpeedRewardConfig(_StrictRewardModel):
    overspeed_margin_mps: StrictFloat = Field(ge=0.0)
    zero_score_overspeed_mps: StrictFloat = Field(gt=0.0)

    @model_validator(mode="after")
    def validate_speed_bounds(self) -> SpeedRewardConfig:
        if self.zero_score_overspeed_mps <= self.overspeed_margin_mps:
            raise ValueError("speed.zero_score_overspeed_mps must exceed overspeed_margin_mps")
        return self


class EnergyRewardConfig(_StrictRewardModel):
    reference_ml_per_km: StrictFloat = Field(gt=0.0)
    minimum_step_distance_m: StrictFloat = Field(gt=0.0)


class PlannerRFTEnergyRewardConfig(_StrictRewardModel):
    name: Literal["plannerrft_energy_v1"]
    weights: RewardWeightsConfig
    gates: RewardGatesConfig
    ttc: TTCRewardConfig
    progress: ProgressRewardConfig
    comfort: ComfortRewardConfig
    speed: SpeedRewardConfig
    energy: EnergyRewardConfig


RewardProfileConfig: TypeAlias = Annotated[
    MetaDriveBuiltinRewardConfig | PlannerRFTEnergyRewardConfig,
    Field(discriminator="name"),
]


@dataclass(frozen=True, slots=True)
class RewardStepInput:
    previous_position_xy_m: tuple[float, float]
    position_xy_m: tuple[float, float]
    previous_velocity_xy_mps: tuple[float, float]
    velocity_xy_mps: tuple[float, float]
    previous_acceleration_xy_mps2: tuple[float, float]
    heading_rad: float
    yaw_rate_radps: float
    route_progress_delta_m: float
    route_heading_rad: float
    speed_limit_mps: float
    ego_width_m: float
    ego_length_m: float
    traffic_frame: TrafficFrame
    crash_vehicle: bool
    crash_object: bool
    crash_building: bool
    crash_human: bool
    crash_sidewalk: bool
    out_of_road: bool
    native_step_energy_ml: float
    native_episode_energy_ml: float
    timestep_s: float


@dataclass(frozen=True, slots=True)
class PlannerRFTEnergyRewardAudit:
    profile_name: Literal["plannerrft_energy_v1"]
    reward_total: float
    reward_ungated: float
    reward_gate: float
    collision_score: float
    drivable_score: float
    wrong_direction_score: float
    ttc_score: float
    progress_score: float
    comfort_score: float
    speed_score: float
    energy_score: float
    has_ttc_candidate: bool
    min_ttc_s: float
    route_progress_delta_m: float
    speed_mps: float
    speed_limit_mps: float
    overspeed_mps: float
    longitudinal_acceleration_mps2: float
    lateral_acceleration_mps2: float
    jerk_mps3: float
    yaw_rate_radps: float
    step_distance_m: float
    native_step_energy_ml: float
    native_episode_energy_ml: float
    executed_fuel_proxy_step_energy_ml: float
    executed_fuel_proxy_ml_per_km: float
    energy_distance_valid: bool


@dataclass(frozen=True, slots=True)
class MetaDriveBuiltinRewardAudit:
    profile_name: Literal["metadrive_builtin_v1"]
    reward_total: float
    dense_reward: float
    terminal_override: float
    step_distance_m: float
    native_step_energy_ml: float
    native_episode_energy_ml: float
    executed_fuel_proxy_step_energy_ml: float
    executed_fuel_proxy_ml_per_km: float
    energy_distance_valid: bool


RewardAudit: TypeAlias = MetaDriveBuiltinRewardAudit | PlannerRFTEnergyRewardAudit


def score_plannerrft_energy_step(
    config: PlannerRFTEnergyRewardConfig, step: RewardStepInput
) -> PlannerRFTEnergyRewardAudit:
    """Score one executed 0.1 s transition without simulator-specific objects."""

    previous_position = _vector(step.previous_position_xy_m, "previous position")
    position = _vector(step.position_xy_m, "position")
    previous_velocity = _vector(step.previous_velocity_xy_mps, "previous velocity")
    velocity = _vector(step.velocity_xy_mps, "velocity")
    previous_acceleration = _vector(
        step.previous_acceleration_xy_mps2, "previous acceleration"
    )
    scalars = {
        "heading": step.heading_rad,
        "yaw rate": step.yaw_rate_radps,
        "route progress": step.route_progress_delta_m,
        "route heading": step.route_heading_rad,
        "speed limit": step.speed_limit_mps,
        "ego width": step.ego_width_m,
        "ego length": step.ego_length_m,
        "native step energy": step.native_step_energy_ml,
        "native episode energy": step.native_episode_energy_ml,
        "timestep": step.timestep_s,
    }
    if not all(math.isfinite(value) for value in scalars.values()):
        raise ValueError("reward step scalars must be finite")
    if step.speed_limit_mps <= 0.0 or step.ego_width_m <= 0.0 or step.ego_length_m <= 0.0:
        raise ValueError("speed limit and ego dimensions must be positive")
    if step.native_step_energy_ml < 0.0 or step.native_episode_energy_ml < 0.0:
        raise ValueError("native MetaDrive energy must be non-negative")
    if step.timestep_s <= 0.0:
        raise ValueError("reward timestep must be positive")

    acceleration = (velocity - previous_velocity) / step.timestep_s
    jerk_mps3 = float(np.linalg.norm(acceleration - previous_acceleration) / step.timestep_s)
    forward = np.asarray([math.cos(step.heading_rad), math.sin(step.heading_rad)])
    left = np.asarray([-forward[1], forward[0]])
    longitudinal_acceleration = float(np.dot(acceleration, forward))
    lateral_acceleration = float(np.dot(acceleration, left))
    speed_mps = float(np.linalg.norm(velocity))

    collision = any(
        (
            config.gates.collision_vehicle and step.crash_vehicle,
            config.gates.collision_object and step.crash_object,
            config.gates.collision_building and step.crash_building,
            config.gates.collision_human and step.crash_human,
            config.gates.collision_sidewalk and step.crash_sidewalk,
        )
    )
    collision_score = float(not collision)
    drivable_score = float(not step.out_of_road)
    heading_error = abs(
        math.atan2(
            math.sin(step.heading_rad - step.route_heading_rad),
            math.cos(step.heading_rad - step.route_heading_rad),
        )
    )
    wrong_direction_score = float(
        heading_error <= config.gates.wrong_direction_max_heading_error_rad
    )
    reward_gate = collision_score * drivable_score * wrong_direction_score

    min_ttc_s, has_ttc_candidate = _minimum_ttc_s(config, step, position, velocity, forward)
    ttc_score = _linear_score(
        min_ttc_s, config.ttc.critical_ttc_s, config.ttc.safe_ttc_s
    )
    progress_score = float(
        np.clip(
            max(0.0, step.route_progress_delta_m) / config.progress.full_score_delta_m,
            0.0,
            1.0,
        )
    )
    comfort_score = min(
        _comfort_component(
            abs(longitudinal_acceleration),
            config.comfort.longitudinal_acceleration_limit_mps2,
        ),
        _comfort_component(
            abs(lateral_acceleration), config.comfort.lateral_acceleration_limit_mps2
        ),
        _comfort_component(jerk_mps3, config.comfort.jerk_limit_mps3),
        _comfort_component(abs(step.yaw_rate_radps), config.comfort.yaw_rate_limit_radps),
    )
    overspeed_mps = max(0.0, speed_mps - step.speed_limit_mps)
    speed_score = float(
        np.clip(
            1.0
            - max(0.0, overspeed_mps - config.speed.overspeed_margin_mps)
            / (
                config.speed.zero_score_overspeed_mps
                - config.speed.overspeed_margin_mps
            ),
            0.0,
            1.0,
        )
    )

    step_distance_m = float(np.linalg.norm(position - previous_position))
    fuel_proxy_step_energy_ml = executed_fuel_proxy_step_energy_ml(
        step.previous_position_xy_m,
        step.position_xy_m,
        speed_mps,
    )
    energy_distance_valid = step_distance_m >= config.energy.minimum_step_distance_m
    fuel_proxy_ml_per_km = (
        fuel_proxy_step_energy_ml * 1000.0 / step_distance_m
        if energy_distance_valid
        else 0.0
    )
    energy_score = (
        math.exp(-fuel_proxy_ml_per_km / config.energy.reference_ml_per_km)
        if energy_distance_valid
        else 0.0
    )

    weights = config.weights
    reward_ungated = (
        weights.ttc * ttc_score
        + weights.progress * progress_score
        + weights.comfort * comfort_score
        + weights.speed * speed_score
        + weights.energy * energy_score
    ) / weights.total
    reward_total = reward_gate * reward_ungated
    output = PlannerRFTEnergyRewardAudit(
        profile_name="plannerrft_energy_v1",
        reward_total=reward_total,
        reward_ungated=reward_ungated,
        reward_gate=reward_gate,
        collision_score=collision_score,
        drivable_score=drivable_score,
        wrong_direction_score=wrong_direction_score,
        ttc_score=ttc_score,
        progress_score=progress_score,
        comfort_score=comfort_score,
        speed_score=speed_score,
        energy_score=energy_score,
        has_ttc_candidate=has_ttc_candidate,
        min_ttc_s=min_ttc_s,
        route_progress_delta_m=step.route_progress_delta_m,
        speed_mps=speed_mps,
        speed_limit_mps=step.speed_limit_mps,
        overspeed_mps=overspeed_mps,
        longitudinal_acceleration_mps2=longitudinal_acceleration,
        lateral_acceleration_mps2=lateral_acceleration,
        jerk_mps3=jerk_mps3,
        yaw_rate_radps=step.yaw_rate_radps,
        step_distance_m=step_distance_m,
        native_step_energy_ml=step.native_step_energy_ml,
        native_episode_energy_ml=step.native_episode_energy_ml,
        executed_fuel_proxy_step_energy_ml=fuel_proxy_step_energy_ml,
        executed_fuel_proxy_ml_per_km=fuel_proxy_ml_per_km,
        energy_distance_valid=energy_distance_valid,
    )
    values = [
        value
        for item in fields(output)
        if isinstance((value := getattr(output, item.name)), float)
    ]
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError("PlannerRFT energy reward produced a non-finite audit value")
    return output


def _minimum_ttc_s(
    config: PlannerRFTEnergyRewardConfig,
    step: RewardStepInput,
    ego_position: np.ndarray,
    ego_velocity: np.ndarray,
    forward: np.ndarray,
) -> tuple[float, bool]:
    candidates: list[float] = []
    objects = [
        (
            np.asarray(item.position_xy_m, dtype=np.float64),
            np.asarray(item.velocity_xy_mps, dtype=np.float64),
            item.heading_rad,
            item.width_m,
            item.length_m,
        )
        for item in step.traffic_frame.participants
    ]
    objects.extend(
        (
            np.asarray(item.position_xy_m, dtype=np.float64),
            np.zeros(2, dtype=np.float64),
            item.heading_rad,
            item.width_m,
            item.length_m,
        )
        for item in step.traffic_frame.static_objects
    )
    left = np.asarray([-forward[1], forward[0]])
    for object_position, object_velocity, object_heading, width_m, length_m in objects:
        relative = object_position - ego_position
        longitudinal = float(np.dot(relative, forward))
        if longitudinal <= 0.0:
            continue
        heading_delta = object_heading - step.heading_rad
        projected_half_length = 0.5 * (
            length_m * abs(math.cos(heading_delta)) + width_m * abs(math.sin(heading_delta))
        )
        projected_half_width = 0.5 * (
            length_m * abs(math.sin(heading_delta)) + width_m * abs(math.cos(heading_delta))
        )
        lateral = abs(float(np.dot(relative, left)))
        lateral_bound = step.ego_width_m / 2 + projected_half_width + config.ttc.lateral_margin_m
        if lateral > lateral_bound:
            continue
        closing_speed = float(np.dot(ego_velocity - object_velocity, forward))
        if closing_speed < config.ttc.minimum_closing_speed_mps:
            continue
        clearance = longitudinal - (
            step.ego_length_m / 2
            + projected_half_length
            + config.ttc.longitudinal_margin_m
        )
        candidates.append(max(0.0, clearance) / closing_speed)
    if not candidates:
        return config.ttc.maximum_ttc_s, False
    return min(min(candidates), config.ttc.maximum_ttc_s), True


def _linear_score(value: float, lower: float, upper: float) -> float:
    return float(np.clip((value - lower) / (upper - lower), 0.0, 1.0))


def _comfort_component(value: float, limit: float) -> float:
    return float(np.clip(1.0 - max(0.0, value - limit) / limit, 0.0, 1.0))


def executed_fuel_proxy_step_energy_ml(
    start_position_xy_m: tuple[float, float] | np.ndarray,
    end_position_xy_m: tuple[float, float] | np.ndarray,
    speed_mps: float,
) -> float:
    """Recompute MetaDrive's proxy from the actual kinematic execution trace."""

    start = np.asarray(start_position_xy_m, dtype=np.float64)
    end = np.asarray(end_position_xy_m, dtype=np.float64)
    if start.shape != (2,) or end.shape != (2,) or not np.isfinite([*start, *end]).all():
        raise ValueError("fuel proxy positions must be finite 2D vectors")
    if not math.isfinite(speed_mps) or speed_mps < 0.0:
        raise ValueError("fuel proxy speed must be finite and non-negative")
    distance_m = float(np.linalg.norm(end - start))
    return (
        3.25
        * math.exp(0.01 * speed_mps * 3.6)
        * (distance_m / 1000.0)
        / 100.0
        * 1000.0
    )


def _vector(value: tuple[float, float], name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (2,) or not np.isfinite(result).all():
        raise ValueError(f"reward {name} must be a finite 2D vector")
    return result


__all__ = [
    "MetaDriveBuiltinRewardConfig",
    "MetaDriveBuiltinRewardAudit",
    "PlannerRFTEnergyRewardAudit",
    "PlannerRFTEnergyRewardConfig",
    "RewardProfileConfig",
    "RewardAudit",
    "RewardStepInput",
    "executed_fuel_proxy_step_energy_ml",
    "score_plannerrft_energy_step",
]
