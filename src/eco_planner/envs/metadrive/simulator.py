"""MetaDrive-native simulator lifecycle and single-step backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import gymnasium as gym
import numpy as np
from metadrive.constants import DEFAULT_AGENT, TerminationState
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.obs.observation_base import BaseObservation
from metadrive.utils import Config, concat_step_infos, merge_dicts

from eco_planner.contracts import (
    METADRIVE_DECISION_REPEAT,
    METADRIVE_PHYSICS_STEP_S,
    validate_metadrive_timestep,
)

from ..array_types import PlannerOnlyObservationArray
from ..domain import WorldTrajectory
from .lane_speed import ProgrammaticLaneSpeedAdapter
from .map import navigation_route_roads
from .policy import KinematicTrajectoryPolicy

_BACKEND_OBSERVATION: PlannerOnlyObservationArray = np.zeros(1, dtype=np.float32)


class _BackendObservation(BaseObservation):
    """Minimal Gym placeholder; planner observations are built by MetaDriveEnvSlot."""

    @property
    def observation_space(self) -> gym.spaces.Box:
        return gym.spaces.Box(0.0, 0.0, shape=(1,), dtype=np.float32)

    def observe(self, *args: Any, **kwargs: Any) -> PlannerOnlyObservationArray:
        return _BACKEND_OBSERVATION.copy()


@dataclass(frozen=True, slots=True)
class MetaDriveStepResult:
    """Typed MetaDrive-native outcome for one actual 0.1 s transition."""

    native_step_energy_ml: float
    native_episode_energy_ml: float
    terminated: bool
    truncated: bool
    route_completion: float
    arrive_dest: bool
    out_of_road: bool
    crash_vehicle: bool
    crash_object: bool
    crash_building: bool
    crash_human: bool
    crash_sidewalk: bool
    max_step: bool


class MetaDriveBackend(MetaDriveEnv):
    """Single-agent MetaDrive backend for project-owned kinematic execution."""

    @classmethod
    def default_config(cls) -> Config:
        config = super().default_config()
        config.update(
            {
                "programmatic_lane_speed_limit_kmh": None,
                "programmatic_lane_speed_limit_profile_kmh": None,
            },
            allow_add_new_key=True,
        )
        return config

    def __init__(
        self,
        config: dict[str, Any] | None = None,
    ) -> None:
        if config is None:
            raise ValueError("MetaDriveBackend requires an explicit configuration")
        required = {"programmatic_lane_speed_limit_kmh"}
        missing = sorted(required - set(config))
        if missing:
            raise ValueError(f"MetaDriveBackend configuration is missing: {missing}")
        if config.get("image_observation", False) is not False:
            raise ValueError("MetaDriveBackend requires image_observation=false")
        if config.get("agent_observation") is not None:
            raise ValueError("MetaDriveBackend does not accept a custom agent_observation")
        supplied_policy = config.get("agent_policy")
        if supplied_policy is not None and supplied_policy is not KinematicTrajectoryPolicy:
            raise ValueError("agent_policy must be KinematicTrajectoryPolicy")

        configured = dict(config)
        configured["physics_world_step_size"] = METADRIVE_PHYSICS_STEP_S
        configured["decision_repeat"] = METADRIVE_DECISION_REPEAT
        validate_metadrive_timestep(
            configured["physics_world_step_size"], configured["decision_repeat"]
        )
        configured["agent_policy"] = KinematicTrajectoryPolicy
        configured["agent_observation"] = _BackendObservation
        self._terminal_out_of_road = False
        super().__init__(configured)
        if self.config["is_multi_agent"]:
            raise ValueError("MetaDriveBackend supports only single-agent operation")
        if self.config["manual_control"]:
            raise ValueError("MetaDriveBackend does not support manual control")
        self._programmatic_lane_speed_adapter = ProgrammaticLaneSpeedAdapter(
            self.config["programmatic_lane_speed_limit_kmh"],
            self.config["programmatic_lane_speed_limit_profile_kmh"],
        )

    def _post_process_config(self, config: Config) -> Config:
        processed = super()._post_process_config(config)
        if not processed["use_render"]:
            processed["sensors"] = (
                {"lidar": processed["sensors"]["lidar"]}
                if float(processed["traffic_density"]) > 0.0
                else {}
            )
        return processed

    @property
    def programmatic_lane_speed_limit_audit(self) -> dict[str, object]:
        """Return the verified lane-speed metadata written during the latest reset."""

        return self._programmatic_lane_speed_adapter.audit

    @property
    def route_completion(self) -> float:
        """Return the current finite route-completion fraction."""

        value = float(self.agent.navigation.route_completion)
        if not np.isfinite(value):
            raise RuntimeError("MetaDrive returned non-finite route completion")
        return value

    @property
    def route_length_m(self) -> float:
        """Return the total length of the current navigation route."""

        checkpoints = list(self.agent.navigation.checkpoints)
        if len(checkpoints) < 2:
            raise RuntimeError("MetaDrive navigation did not expose a complete route")
        current_map = cast(Any, self.current_map)
        graph = current_map.road_network.graph
        lengths: list[float] = []
        for start, end in zip(checkpoints[:-1], checkpoints[1:], strict=True):
            lanes = graph.get(start, {}).get(end, [])
            if not lanes:
                raise RuntimeError(f"route edge {(start, end)!r} has no lane")
            length = getattr(lanes[0], "length", None)
            if isinstance(length, (bool, np.bool_)) or not isinstance(
                length, (int, float, np.integer, np.floating)
            ):
                raise RuntimeError(f"route edge {(start, end)!r} has an invalid length")
            if not np.isfinite(length) or float(length) <= 0.0:
                raise RuntimeError(f"route edge {(start, end)!r} has an invalid length")
            lengths.append(float(length))
        return float(sum(lengths))

    @property
    def is_out_of_road_terminal(self) -> bool:
        """Whether the latest transition ended because the ego is out of road."""

        return self._terminal_out_of_road

    def reset(self, *args: Any, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        self._terminal_out_of_road = False
        self._programmatic_lane_speed_adapter.clear_audit()
        if self.engine is not None:
            # BaseEnv.reset advances taskMgr once; do not replay the prior episode's trajectory.
            self.engine.external_actions = None
        result = super().reset(*args, **kwargs)
        current_map = self.current_map
        if current_map is None:
            raise RuntimeError("MetaDrive did not create a current map during reset")
        self._programmatic_lane_speed_adapter.apply(current_map)
        return result

    def step_world_trajectory(self, action: WorldTrajectory | None) -> MetaDriveStepResult:
        """Advance one project-level simulator transition using a prepared world trajectory."""

        engine_info = self._step_planner_simulator({DEFAULT_AGENT: action})
        agent_id, agent = next(iter(self.agents.items()))
        self.episode_lengths[agent_id] += 1
        done, done_info = self.done_function(agent_id)
        _, cost_info = self.cost_function(agent_id)
        self.dones[agent_id] = done or self.dones[agent_id]
        info = concat_step_infos(
            [engine_info, {agent_id: done_info}, {agent_id: cost_info}]
        )
        agent_info = info.pop(agent_id)
        info.update(agent_info)
        truncated = bool(info.get(TerminationState.MAX_STEP, False))
        terminated = bool(self.dones[agent_id])
        if self.config["horizon"] and self.episode_step > 5 * self.config["horizon"]:
            truncated = True
            if self.config["truncate_as_terminate"]:
                self.dones[agent_id] = terminated = True
        self._terminal_out_of_road = terminated and bool(info[TerminationState.OUT_OF_ROAD])
        if agent is not self.agent:
            raise RuntimeError("single-agent environment changed its active agent")
        return MetaDriveStepResult(
            native_step_energy_ml=_finite_info_scalar(info, "step_energy"),
            native_episode_energy_ml=_finite_info_scalar(info, "episode_energy"),
            terminated=terminated,
            truncated=truncated,
            route_completion=self.route_completion,
            arrive_dest=bool(info[TerminationState.SUCCESS]),
            out_of_road=bool(info[TerminationState.OUT_OF_ROAD]),
            crash_vehicle=bool(info[TerminationState.CRASH_VEHICLE]),
            crash_object=bool(info[TerminationState.CRASH_OBJECT]),
            crash_building=bool(info[TerminationState.CRASH_BUILDING]),
            crash_human=bool(info[TerminationState.CRASH_HUMAN]),
            crash_sidewalk=bool(info[TerminationState.CRASH_SIDEWALK]),
            max_step=bool(info[TerminationState.MAX_STEP]),
        )

    def _is_out_of_road(self, vehicle: Any) -> bool:
        """Treat a drivable lane outside the complete navigation route as off-road."""

        if super()._is_out_of_road(vehicle):
            return True
        lane = vehicle.lane
        lane_index = getattr(lane, "index", None)
        if not isinstance(lane_index, tuple) or len(lane_index) < 2:
            raise RuntimeError(f"vehicle lane has invalid index: {lane_index!r}")
        lane_road = (lane_index[0], lane_index[1])
        return lane_road not in navigation_route_roads(vehicle.navigation)

    def _step_planner_simulator(self, actions: dict[str, WorldTrajectory | None]) -> dict:
        before_info = self.engine.before_step(actions)
        if self.config["_render_mode"] != "none" or self.config["record_episode"]:
            self.engine.step(self.config["decision_repeat"])
        else:
            repeats = self.config["decision_repeat"]
            for _ in range(repeats):
                for name, manager in self.engine.managers.items():
                    if name != "record_manager":
                        manager.step()
            physics_step = float(self.config["physics_world_step_size"])
            dynamic_world = cast(Any, self.engine.physics_world.dynamic_world)
            dynamic_world.doPhysics(physics_step * repeats, repeats, physics_step)
        after_info = self.engine.after_step()
        return merge_dicts(after_info, before_info, allow_new_keys=True, without_copy=True)


def _finite_info_scalar(info: dict[str, Any], name: str) -> float:
    value: float = info[name]
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)
