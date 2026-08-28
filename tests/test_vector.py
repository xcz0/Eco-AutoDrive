from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict
from torchrl.data import Binary, Composite, Unbounded

from eco_planner.envs.runtime.vector import (
    VectorMetaDriveEnv,
    VectorMetaDriveWorkerError,
    _TorchRLScenarioMetaDriveEnv,
    _WorkerFailure,
)


def test_step_failure_output_preserves_the_original_worker_traceback() -> None:
    worker = SimpleNamespace(
        observation_spec=Composite(
            observation=Unbounded(shape=(2,), dtype=torch.float32), shape=(), device="cpu"
        ),
        done_spec=Composite(
            done=Binary(1, shape=(1,), dtype=torch.bool, device="cpu"),
            terminated=Binary(1, shape=(1,), dtype=torch.bool, device="cpu"),
            truncated=Binary(1, shape=(1,), dtype=torch.bool, device="cpu"),
            shape=(),
            device="cpu",
        ),
        reward_spec=Unbounded(shape=(1,), dtype=torch.float32, device="cpu"),
        _operation_result=None,
    )
    try:
        raise RuntimeError("sentinel worker failure")
    except RuntimeError:
        output = _TorchRLScenarioMetaDriveEnv._failure_output(worker, "step")

    assert isinstance(output, TensorDict)
    assert output["reward"].shape == (1,)
    assert output["reward"].dtype is torch.float32
    assert isinstance(worker._operation_result, _WorkerFailure)
    assert worker._operation_result.operation == "step"
    assert "sentinel worker failure" in worker._operation_result.traceback_text
    assert "AttributeError" not in worker._operation_result.traceback_text

    closed = False

    def close() -> None:
        nonlocal closed
        closed = True

    facade = SimpleNamespace(close=close)
    with pytest.raises(VectorMetaDriveWorkerError, match="slot 3 failed during step") as error:
        VectorMetaDriveEnv._raise_worker_failure(facade, 3, worker._operation_result)

    assert closed is True
    assert "sentinel worker failure" in str(error.value)
