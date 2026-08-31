from __future__ import annotations

import importlib
import os
import sys

import pytest

from eco_planner import configuration
from eco_planner._repository import LOCAL_ENVIRONMENT_PATH

HYDRA_ENTRYPOINTS = ("scripts.evaluate", "scripts.train", "scripts.benchmark")


@pytest.mark.parametrize("module_name", HYDRA_ENTRYPOINTS)
def test_importing_hydra_entrypoint_does_not_bootstrap_environment(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
) -> None:
    load_calls: list[object] = []
    monkeypatch.setattr(
        configuration,
        "load_local_environment",
        lambda path: load_calls.append(path),
    )
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    sys.modules.pop(module_name, None)

    try:
        importlib.import_module(module_name)
        assert load_calls == []
        assert "CUBLAS_WORKSPACE_CONFIG" not in os.environ
    finally:
        sys.modules.pop(module_name, None)


@pytest.mark.parametrize("module_name", HYDRA_ENTRYPOINTS)
def test_main_bootstraps_environment_before_hydra(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
) -> None:
    module = importlib.import_module(module_name)
    calls: list[tuple[str, object | None]] = []

    def record_environment_load(path: object) -> None:
        if module_name == "scripts.train":
            assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
        calls.append(("load", path))

    monkeypatch.setattr(
        module,
        "load_local_environment",
        record_environment_load,
    )
    monkeypatch.setattr(module, "_hydra_main", lambda: calls.append(("hydra", None)))
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)

    module.main()

    assert calls == [("load", LOCAL_ENVIRONMENT_PATH), ("hydra", None)]
    if module_name == "scripts.train":
        assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    else:
        assert "CUBLAS_WORKSPACE_CONFIG" not in os.environ
