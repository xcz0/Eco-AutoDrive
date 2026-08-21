from __future__ import annotations

import json

import numpy as np
import pytest

from eco_planner.evaluation import episode, runner
from eco_planner.evaluation.artifacts.models import FailurePhase
from eco_planner.evaluation.config import parse_evaluation_config
from eco_planner.evaluation.failures import EpisodeFailure
from eco_planner.evaluation.runner import run_evaluation


def _patch_runtime(monkeypatch: pytest.MonkeyPatch, fake_runtime: object) -> None:
    monkeypatch.setattr(runner, "create_fabric_inference_runtime", lambda *args: fake_runtime)
    monkeypatch.setattr(runner, "write_runtime_metadata", lambda *args: None)


def test_episode_failure_preserves_original_cause() -> None:
    cause = RuntimeError("planner produced a non-finite trajectory")
    failure = EpisodeFailure(FailurePhase.INFERENCE, cause)

    assert failure.phase is FailurePhase.INFERENCE
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
            raise EpisodeFailure(FailurePhase.RESET, RuntimeError("injected episode failure"))

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


def test_run_evaluation_selects_vector_runner_when_slots_are_configured(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    evaluation_config: object,
    fake_runtime: object,
    patch_episode_dependencies,
) -> None:
    evaluation_config.evaluation.execution.vector_env_slots = 1  # type: ignore[attr-defined]
    patch_episode_dependencies()
    _patch_runtime(monkeypatch, fake_runtime)
    calls: list[tuple[str, ...]] = []

    def vector_runner(specs, runtime, config, output_dir):
        calls.append(tuple(spec.name for spec in specs))
        return tuple(episode.run_scenario(spec, runtime, config, output_dir) for spec in specs)

    monkeypatch.setattr(runner, "run_vector_scenarios", vector_runner)

    summary = runner.run_evaluation(parse_evaluation_config(evaluation_config), tmp_path)

    assert summary.status == "completed"
    assert calls == [("fake",)]


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
