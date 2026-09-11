"""Research command routing, required arguments, bootstrap and offline boundaries."""

import ast
import os
import subprocess
import sys
from importlib import import_module

import pytest

from eco_planner._repository import REPOSITORY_ROOT
from scripts import experiments as cli


@pytest.mark.parametrize(
    "command",
    [
        "reward lambda-identifiability run",
        "reward calibration run --source-dir in",
        "reward critic-gae-ablation run --source-dir in",
        "reward scalar run --operation train --arm a1",
        "reward scalar run --operation train --arm a0 --training-seed 1",
        "reward scalar run --operation train --arm a1 --training-seed 1 --checkpoint final",
        "reward scalar run --operation evaluate --arm a1 --checkpoint final",
        "reward scalar run --operation evaluate --arm a0 --checkpoint-path policy.pt",
        "reward scalar run --operation evaluate --arm a0 --training-seed 1",
        "reward scalar run --operation evaluate --arm a2 --override ppo.epochs=1",
        "reward scalar run --operation evaluate --arm a0 --reference-dir in",
        "training stability run",
        "training stability run --operation diagnostic",
        "training stability run --operation search --diagnostic gradient",
        "training reproducibility validate",
        "reward scalar analyze --source-dir in",
        "scalar-reward evaluate-a0",
        "training stability stage-a",
        "reward sanity run",
    ],
)
def test_invalid_arguments_fail_before_bootstrap(monkeypatch, command):
    monkeypatch.setattr(cli, "bootstrap", lambda _: pytest.fail("bootstrapped invalid command"))
    monkeypatch.setattr(sys, "argv", ["experiment", *command.split(), "--output-dir", "out"])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2


@pytest.mark.parametrize("status,code", [("completed", 0), ("failed", 1), (1, 1)])
def test_exit_status(monkeypatch, status, code):
    monkeypatch.setattr(cli, "bootstrap", lambda _: None)
    monkeypatch.setattr(
        cli, "dispatch", lambda _: status if isinstance(status, int) else {"status": status}
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "experiment",
            "guidance",
            "energy-sweep",
            "run",
            "--output-dir",
            "out",
        ],
    )
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == code


RUNS = [
    ("reward fixed-batch collect", ""),
    ("reward lambda-identifiability run", "--source-dir in"),
    ("reward calibration run", "--source-dir in --reference-dir ref"),
    ("reward objective-decomposition run", "--source-dir in"),
    ("reward critic-gae-ablation run", "--source-dir in --reference-dir ref"),
    ("guidance energy-sweep run", ""),
    ("guidance control-authority run", ""),
    ("training reproducibility validate", "--source-dir in"),
    ("reward scalar run", "--operation train --arm a1 --training-seed 2 --override ppo.epochs=1"),
    ("reward scalar run", "--operation evaluate --arm a0"),
    (
        "reward scalar run",
        "--operation evaluate --arm a2 --checkpoint final --checkpoint-path policy.pt",
    ),
    *[("training stability run", f"--operation {op}") for op in ("search", "confirm", "held-out")],
    ("training stability run", "--operation diagnostic --diagnostic guidance"),
]


@pytest.mark.parametrize("command,options", RUNS)
def test_operation_routing_and_no_figures(monkeypatch, command, options):
    argv = [*command.split(), *options.split(), "--output-dir", "out"]
    if not command.endswith("collect"):
        argv.append("--no-figures")
    parser = cli.build_parser()
    args = parser.parse_args(argv)
    cli.validate_arguments(parser, args)
    spec = args.command
    if spec.config:
        assert args.config.is_file()
    target = import_module("eco_planner.experiments." + spec.module)
    received = []
    monkeypatch.setattr(target, spec.function, lambda *a, **kw: received.append((a, kw)) or {})
    cli.dispatch(args)
    positional, keyword = received[0]
    if args.action == "collect":
        assert positional == (args.config, args.output_dir) and not keyword
    else:
        assert keyword["figures"] is False
    if spec.evidence == "scalar-reward":
        assert positional[0] == args.operation
        assert keyword["arm"] == args.arm
        assert keyword["training_seed"] == args.training_seed
        assert keyword["checkpoint_path"] == args.checkpoint_path
        assert keyword["overrides"] == args.override
    elif spec.evidence == "ppo-stability":
        assert positional == (args.operation, args.config, args.output_dir, args.diagnostic)
    elif spec.source:
        assert positional[0] == args.source_dir
        if spec.reference:
            assert positional[1] == args.reference_dir


@pytest.mark.parametrize(
    "key", [key for key, spec in cli.COMMANDS.items() if "analyze" in spec.actions]
)
def test_analysis_is_offline_and_scalar_inputs_are_validated(monkeypatch, key):
    from eco_planner import configuration
    from eco_planner.analysis import runner
    from eco_planner.experiments.reward.scalar import comparison

    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.setattr(
        configuration, "load_local_environment", lambda _: pytest.fail("loaded environment")
    )
    argv = [*key, "analyze", "--source-dir", "in", "--output-dir", "out", "--no-figures"]
    validated = object()
    loaded = []
    monkeypatch.setattr(
        comparison, "load_comparison", lambda path: loaded.append(path) or validated
    )
    if key == ("reward", "scalar"):
        argv += ["--config", "comparison.yaml"]
    args = cli.build_parser().parse_args(argv)
    calls = []
    monkeypatch.setattr(runner, "analyze", lambda *a, **kw: calls.append((a, kw)) or {})
    cli.bootstrap(args)
    cli.dispatch(args)
    assert "CUBLAS_WORKSPACE_CONFIG" not in os.environ
    assert calls[0][0] == (args.command.evidence, args.source_dir, args.output_dir)
    assert calls[0][1]["figures"] is False
    if key == ("reward", "scalar"):
        assert loaded == [args.config]
        assert calls[0][1]["scalar_comparison"] is validated
    else:
        assert not loaded


def test_collection_bootstrap_sets_cuda_before_environment(monkeypatch):
    from eco_planner import configuration

    calls = []
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.setattr(
        configuration,
        "load_local_environment",
        lambda _: calls.append(os.environ["CUBLAS_WORKSPACE_CONFIG"]),
    )
    args = cli.build_parser().parse_args(
        ["reward", "fixed-batch", "collect", "--output-dir", "out"]
    )
    cli.bootstrap(args)
    assert calls == [":4096:8"]


def test_help_does_not_import_execution():
    script = """
import sys
from scripts.experiments import build_parser
from scripts.validation import build_parser as validation
from scripts.benchmark_execution import build_parser as benchmark
for factory in (build_parser, validation, benchmark):
    try:
        factory().parse_args(['--help'])
    except SystemExit as error:
        assert error.code == 0
for root in ('torch', 'metadrive', 'panda3d', 'eco_planner.rl.trainer'):
    assert not any(k == root or k.startswith(root + '.') for k in sys.modules), root
"""
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True)


def test_analysis_and_core_do_not_import_experiments():
    root = REPOSITORY_ROOT / "src" / "eco_planner"
    for folder in ("analysis", "rl", "evaluation", "benchmarking", "runtime", "models", "envs"):
        for path in (root / folder).rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""]
                else:
                    continue
                assert not any(
                    name == "eco_planner.experiments" or name.startswith("eco_planner.experiments.")
                    for name in names
                ), path


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
