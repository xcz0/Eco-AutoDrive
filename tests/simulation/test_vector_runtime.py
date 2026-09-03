from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict
from torchrl.data import Binary, Composite, Unbounded

from eco_planner.runtime.envs.vector import (
    VectorMetaDriveEnv,
    VectorMetaDriveWorkerError,
    operation_results,
)
from eco_planner.runtime.envs.worker import (
    TorchRLScenarioMetaDriveEnv,
    WorkerFailure,
)


def test_operation_results_follow_the_tensor_batch_dimension() -> None:
    first = WorkerFailure("reset", "first")
    second = WorkerFailure("step", "second")
    output = TensorDict({"value": torch.zeros((2, 1))}, batch_size=[2]).set_non_tensor(
        "operation_results", (first, second)
    )

    assert operation_results(output, WorkerFailure) == (first, second)

    invalid = TensorDict({"value": torch.zeros((2, 1))}, batch_size=[2]).set_non_tensor(
        "operation_results", (first,)
    )
    with pytest.raises(TypeError, match="batch dimension"):
        operation_results(invalid, WorkerFailure)


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
        output = TorchRLScenarioMetaDriveEnv._failure_output(worker, "step")

    assert isinstance(output, TensorDict)
    assert output["reward"].shape == (1,)
    assert output["reward"].dtype is torch.float32
    assert isinstance(worker._operation_result, WorkerFailure)
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
