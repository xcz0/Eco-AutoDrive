from __future__ import annotations

import json

import numpy as np
import pytest

from eco_planner.evaluation import (
    EpisodeFailure,
    parse_evaluation_config,
    run_evaluation,
    runner,
)


def _patch_runtime(monkeypatch: pytest.MonkeyPatch, fake_runtime: object) -> None:
    monkeypatch.setattr(runner, "create_fabric_inference_runtime", lambda *args: fake_runtime)
    monkeypatch.setattr(runner, "write_runtime_metadata", lambda *args: None)


def test_episode_failure_preserves_original_cause() -> None:
    cause = RuntimeError("planner produced a non-finite trajectory")
    failure = EpisodeFailure("inference", cause)

    assert failure.stage == "inference"
    assert failure.cause is cause
    assert str(failure) == "inference: planner produced a non-finite trajectory"


def test_run_evaluation_writes_job_summary(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    evaluation_config: object,
    fake_runtime: object,
    patch_episode_dependencies,
) -> None:
    patch_episode_dependencies()
    _patch_runtime(monkeypatch, fake_runtime)
    summary = runner.run_evaluation(parse_evaluation_config(evaluation_config), tmp_path)

    assert summary.status == "completed"
    assert summary.runtime.seed == 7
    assert summary.episodes[0].sampler == summary.sampler
    assert json.loads((tmp_path / "summary.json").read_text()) == summary.model_dump(mode="json")
    assert (tmp_path / "resolved_config.yaml").is_file()


def test_run_evaluation_persists_episode_failure_and_continues(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    evaluation_config: object,
    patch_episode_dependencies,
    fake_env_class: object,
    fake_runtime: object,
) -> None:
    class FailureEnv(fake_env_class):  # type: ignore[misc, valid-type]
        def reset(self, seed: int) -> tuple[None, dict[str, object]]:
            raise EpisodeFailure("reset", RuntimeError("injected episode failure"))

    def environment(config: dict[str, object]) -> object:
        return FailureEnv(config) if config["map"] == "FAIL" else fake_env_class(config)

    evaluation_config.scenarios = [  # type: ignore[attr-defined]
        {"name": "failed", "map": "FAIL", "seed": 3},
        {"name": "completed", "map": "S", "seed": 3},
    ]
    patch_episode_dependencies(environment)
    _patch_runtime(monkeypatch, fake_runtime)
    summary = run_evaluation(parse_evaluation_config(evaluation_config), tmp_path)

    assert [item.status for item in summary.episodes] == ["failed", "completed"]
    failure = summary.episodes[0]
    assert failure.termination.model_dump() == {"type": "runtime_error", "detail": "reset"}
    with np.load(tmp_path / "failed" / "trace.npz") as trace:
        assert trace["trace_status"].item() == "empty"


def test_run_evaluation_propagates_unclassified_errors(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    evaluation_config: object,
    patch_episode_dependencies,
    fake_env_class: object,
    fake_runtime: object,
) -> None:
    class BrokenEnv(fake_env_class):  # type: ignore[misc, valid-type]
        def reset(self, seed: int) -> tuple[None, dict[str, object]]:
            raise RuntimeError("unclassified")

    patch_episode_dependencies(BrokenEnv)
    _patch_runtime(monkeypatch, fake_runtime)
    with pytest.raises(RuntimeError, match="unclassified"):
        run_evaluation(parse_evaluation_config(evaluation_config), tmp_path)
    assert not (tmp_path / "summary.json").exists()
