"""TorchRL-backed parallel MetaDrive environments and operation sidecars."""

from eco_planner.envs.parallel.torchrl import TorchRLMetaDriveEnv
from eco_planner.envs.parallel.vector import (
    VectorMetaDriveEnv,
    VectorMetaDriveWorkerError,
    operation_results,
)
from eco_planner.envs.parallel.worker import (
    VectorEnvScenario,
    VectorEnvTiming,
    WorkerResetResult,
    WorkerStepResult,
)

__all__ = [
    "TorchRLMetaDriveEnv",
    "VectorEnvScenario",
    "VectorEnvTiming",
    "VectorMetaDriveEnv",
    "VectorMetaDriveWorkerError",
    "WorkerResetResult",
    "WorkerStepResult",
    "operation_results",
]
