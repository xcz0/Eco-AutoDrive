"""Planner-facing MetaDrive interfaces."""

# ruff: noqa: I001

# Torch must load before MetaDrive/Panda3D on Windows to avoid DLL initialization failures.
from eco_planner.envs.domain import ExecutionResult, ExecutionTrace, EpisodeStatus, WorldTrajectory
from eco_planner.envs.observation import PlannerObservationSpec, collate_observations
from eco_planner.envs.metadrive.observation import TrafficObservationAudit
from eco_planner.envs.metadrive.execution import TrajectoryExecutionRecord
from eco_planner.envs.metadrive.simulator import TrajectoryMetaDriveEnv
from eco_planner.envs.metadrive.slot import (
    EnvSlotObservation,
    EnvSlotReset,
    EnvSlotStep,
    MetaDriveEnvSlot,
)
from eco_planner.envs.torchrl.adapter import TorchRLMetaDriveEnv
from eco_planner.envs.runtime.vector import (
    VectorEnvReset,
    VectorEnvScenario,
    VectorEnvStep,
    VectorEnvTiming,
    VectorMetaDriveEnv,
    VectorMetaDriveWorkerError,
)

__all__ = [
    "collate_observations",
    "EpisodeStatus",
    "ExecutionResult",
    "ExecutionTrace",
    "PlannerObservationSpec",
    "TrafficObservationAudit",
    "TrajectoryMetaDriveEnv",
    "TrajectoryExecutionRecord",
    "WorldTrajectory",
    "EnvSlotObservation",
    "EnvSlotReset",
    "EnvSlotStep",
    "MetaDriveEnvSlot",
    "TorchRLMetaDriveEnv",
    "VectorEnvReset",
    "VectorEnvScenario",
    "VectorEnvStep",
    "VectorEnvTiming",
    "VectorMetaDriveEnv",
    "VectorMetaDriveWorkerError",
]
