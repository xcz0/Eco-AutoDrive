from __future__ import annotations

import importlib
import os
import subprocess
import sys
from importlib import import_module

import pytest

from eco_planner import configuration
from eco_planner._repository import LOCAL_ENVIRONMENT_PATH, REPOSITORY_ROOT
from scripts import experiments as cli


@pytest.mark.parametrize("key", list(cli.COMMANDS))
def test_current_operations_parse_and_route(monkeypatch, key):
    spec = cli.COMMANDS[key]
    args = [*key, "--output-dir", "out"]
    if spec.source:
        args += ["--source-dir", "source"]
    if key in (("compare", "analyze"), ("training", "diagnose"), ("training", "eval")):
        args += ["--config", "comparison.yaml"]
    if key == ("compare", "train"):
        args += ["--arm", "r0", "--training-seed", "0"]
    if key == ("compare", "eval"):
        args += ["--arm", "a0"]
    parsed = cli.build_parser().parse_args(args)
    assert parsed.key == key
    cli.validate_arguments(cli.build_parser(), parsed)
    calls = []
    if parsed.action == "analyze":
        from eco_planner.analysis import runner

        monkeypatch.setattr(runner, "analyze", lambda *a, **k: calls.append((a, k)) or {})
        if parsed.domain == "compare":
            from eco_planner.experiments.comparison import inputs

            monkeypatch.setattr(inputs, "load_comparison", lambda *_: "validated")
    else:
        module = import_module("eco_planner.experiments." + spec.module)
        monkeypatch.setattr(module, spec.function, lambda *a, **k: calls.append((a, k)) or {})
    cli.dispatch(parsed)
    assert len(calls) == 1


def test_help_does_not_import_execution():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from scripts.experiments import build_parser; build_parser(); "
            "assert not any(m in sys.modules for m in ('torch', 'metadrive', 'panda3d'))",
        ],
        check=True,
        cwd=REPOSITORY_ROOT,
    )


def test_collection_bootstrap_sets_cuda_before_environment(monkeypatch):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    calls = []
    monkeypatch.setattr(
        configuration,
        "load_local_environment",
        lambda path: calls.append((path, os.environ["CUBLAS_WORKSPACE_CONFIG"])),
    )
    args = cli.build_parser().parse_args(["reward", "collect", "--output-dir", "out"])
    cli.bootstrap(args)
    assert calls == [(LOCAL_ENVIRONMENT_PATH, ":4096:8")]


@pytest.mark.parametrize(
    "adapter,module,function,command",
    [
        (
            "scripts.validation",
            "eco_planner.reward_validation",
            "run_sanity",
            "reward run --output-dir out --no-figures",
        ),
        (
            "scripts.benchmark_execution",
            "eco_planner.benchmarking.execution",
            "write_report",
            "report --serial-dir serial --job-level-dir job --vector-dir vector "
            "--serial-wall-s 1 --job-level-wall-s 2 --vector-wall-s 3 "
            "--output-dir out --no-figures",
        ),
    ],
)
def test_non_research_execution_routes(monkeypatch, adapter, module, function, command):
    entry = import_module(adapter)
    calls = []
    monkeypatch.setattr(
        import_module(module), function, lambda *a, **kw: calls.append((a, kw)) or {}
    )
    entry.dispatch(entry.build_parser().parse_args(command.split()))
    assert calls[0][1]["figures"] is False


@pytest.mark.parametrize(
    "adapter,command,evidence",
    [
        ("scripts.validation", "reward analyze", "reward-sanity"),
        ("scripts.benchmark_execution", "analyze", "execution-backend"),
    ],
)
def test_non_research_analysis_routes(monkeypatch, adapter, command, evidence):
    from eco_planner.analysis import runner

    calls = []
    monkeypatch.setattr(runner, "analyze", lambda *a, **kw: calls.append((a, kw)) or {})
    entry = import_module(adapter)
    entry.dispatch(
        entry.build_parser().parse_args(
            [
                *command.split(),
                "--source-dir",
                "in",
                "--output-dir",
                "out",
                "--no-figures",
            ]
        )
    )
    assert calls[0][0][0] == evidence
    assert calls[0][1]["figures"] is False


@pytest.mark.parametrize("status,code", [("passed", 0), ("failed", 1)])
def test_reward_validation_exit_status(monkeypatch, status, code, capsys):
    from scripts import validation

    monkeypatch.setattr(validation, "dispatch", lambda _: {"status": status})
    monkeypatch.setattr(sys, "argv", ["validation", "reward", "run", "--output-dir", "out"])
    with pytest.raises(SystemExit) as error:
        validation.main()
    assert error.value.code == code
    assert status in capsys.readouterr().out


HYDRA_ENTRYPOINTS = ("scripts.evaluation", "scripts.training", "scripts.benchmark")


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
        if module_name == "scripts.training":
            assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
        monkeypatch.setenv("MACHINE_NAME", "rtx3050_laptop")
        calls.append(("load", path))

    monkeypatch.setattr(
        module,
        "load_local_environment",
        record_environment_load,
    )
    monkeypatch.setattr(module, "_hydra_main", lambda: calls.append(("hydra", None)))
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.delenv("MACHINE_NAME", raising=False)
    monkeypatch.setattr(module.sys, "argv", [module_name, "runtime.seed=17"])

    module.main()

    assert calls == [("load", LOCAL_ENVIRONMENT_PATH), ("hydra", None)]
    assert module.sys.argv == [
        module_name,
        "runtime.seed=17",
        "components/resources=rtx3050_laptop",
    ]
    if module_name == "scripts.training":
        assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    else:
        assert "CUBLAS_WORKSPACE_CONFIG" not in os.environ
