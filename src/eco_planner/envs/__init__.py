"""MetaDrive integration interfaces."""

import torch as _torch  # noqa: F401

from eco_planner.envs.execution import KinematicTrajectoryPolicy, TrajectoryExecutionRecord
from eco_planner.envs.map_adapter import MetaDriveMapAdapter
from eco_planner.envs.metadrive_env import TrajectoryMetaDriveEnv
from eco_planner.envs.observation import collate_observations
from eco_planner.envs.observation_adapter import (
    MetaDriveObservationAdapter,
    NoTrafficMetaDriveObservationAdapter,
    TrafficObservationAudit,
)
from eco_planner.envs.traffic_state import (
    StaticTrafficObjectState,
    TrafficFrame,
    TrafficParticipantState,
)
from eco_planner.envs.vector_metadrive import (
    VectorEnvReset,
    VectorEnvScenario,
    VectorEnvStep,
    VectorEnvTiming,
    VectorMetaDriveEnv,
    VectorMetaDriveWorkerError,
)

__all__ = [
    "KinematicTrajectoryPolicy",
    "collate_observations",
    "MetaDriveMapAdapter",
    "MetaDriveObservationAdapter",
    "NoTrafficMetaDriveObservationAdapter",
    "StaticTrafficObjectState",
    "TrafficFrame",
    "TrafficObservationAudit",
    "TrafficParticipantState",
    "TrajectoryMetaDriveEnv",
    "TrajectoryExecutionRecord",
    "VectorEnvReset",
    "VectorEnvScenario",
    "VectorEnvStep",
    "VectorEnvTiming",
    "VectorMetaDriveEnv",
    "VectorMetaDriveWorkerError",
]
