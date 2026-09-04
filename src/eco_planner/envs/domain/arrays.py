"""Array shape and dtype contracts for environment domain data."""

# jaxtyping shape strings are runtime metadata, not Python forward annotations.
# ruff: noqa: F722, F821, UP037

from __future__ import annotations

from typing import TypeAlias

import numpy as np
from jaxtyping import Bool, Float32, Float64

from eco_planner.contracts import PLANNER_HORIZON

TrajectoryArray: TypeAlias = Float32[np.ndarray, f"{PLANNER_HORIZON} 4"]
WorldVectorArray: TypeAlias = Float64[np.ndarray, "2"]
WorldPointArray: TypeAlias = Float64[np.ndarray, "points 2"]
WorldHeadingArray: TypeAlias = Float64[np.ndarray, "points"]
WorldVelocityArray: TypeAlias = Float64[np.ndarray, f"{PLANNER_HORIZON} 2"]
WorldAngularVelocityArray: TypeAlias = Float64[np.ndarray, f"{PLANNER_HORIZON}"]
ExecutionStateArray: TypeAlias = Float64[np.ndarray, "execution_steps 7"]
ExecutionPointArray: TypeAlias = Float64[np.ndarray, "execution_steps 2"]
ExecutionScalarArray: TypeAlias = Float64[np.ndarray, "execution_steps"]
ExecutionBooleanArray: TypeAlias = Bool[np.ndarray, "execution_steps"]
Float64Array: TypeAlias = Float64[np.ndarray, "*shape"]
