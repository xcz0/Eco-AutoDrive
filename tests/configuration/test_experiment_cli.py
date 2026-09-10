"""Required command arguments, bootstrap ordering and exit status."""

import os
import subprocess
import sys

import pytest

from scripts.experiments import __main__ as cli


@pytest.mark.parametrize(
    "command",
    [
        ["lambda-identifiability", "run"],
        ["reward-calibration", "run", "--source-dir", "in"],
        ["critic-gae-ablation", "run", "--source-dir", "in"],
        ["scalar-reward", "train", "--arm", "a1"],
        ["scalar-reward", "evaluate-policy", "--arm", "a1", "--checkpoint", "final"],
        ["ppo-stability", "diagnose"],
        ["scalar-reward", "analyze", "--source-dir", "in"],
    ],
)
def test_missing_arguments_fail_before_bootstrap(monkeypatch, command):
    monkeypatch.setattr(cli, "bootstrap", lambda _: pytest.fail("bootstrapped invalid command"))
    monkeypatch.setattr(sys, "argv", ["experiment", *command, "--output-dir", "out"])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2


@pytest.mark.parametrize("status,code", [("completed", 0), ("failed", 1), (1, 1)])
def test_exit_status(monkeypatch, status, code):
    monkeypatch.setattr(cli, "bootstrap", lambda _: None)
    monkeypatch.setattr(
        cli, "dispatch", lambda _: status if isinstance(status, int) else {"status": status}
    )
    monkeypatch.setattr(sys, "argv", ["experiment", "reward-sanity", "run", "--output-dir", "out"])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == code


def test_help_and_report_bootstrap_are_offline(monkeypatch):
    from eco_planner import configuration

    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.setattr(
        configuration, "load_local_environment", lambda _: pytest.fail("loaded env")
    )
    for command in (
        ["ppo-stability", "summarize"],
        ["ppo-reproducibility", "report", "--source-dir", "in"],
        ["lambda-identifiability", "analyze", "--source-dir", "in"],
    ):
        cli.bootstrap(cli.build_parser().parse_args([*command, "--output-dir", "out"]))
    assert "CUBLAS_WORKSPACE_CONFIG" not in os.environ
    script = """
import sys
from scripts.experiments.__main__ import build_parser
try:
    build_parser().parse_args(['--help'])
except SystemExit as error:
    assert error.code == 0
for root in ('torch', 'metadrive', 'panda3d', 'eco_planner.rl.trainer'):
    assert not any(k == root or k.startswith(root + '.') for k in sys.modules), root
"""
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True)


def test_collection_bootstrap_sets_cuda_before_local_environment(monkeypatch):
    from eco_planner import configuration

    calls = []
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.setattr(
        configuration,
        "load_local_environment",
        lambda _: calls.append(os.environ["CUBLAS_WORKSPACE_CONFIG"]),
    )
    cli.bootstrap(cli.build_parser().parse_args(["fixed-batch", "collect", "--output-dir", "out"]))
    assert calls == [":4096:8"]


@pytest.mark.parametrize(
    "experiment,action,module,function,options",
    [
        ("fixed-batch", "collect", "fixed_batch.collection", "collect", []),
        (
            "scalar-reward",
            "train",
            "scalar_reward.runner",
            "run_command",
            ["--arm", "a1", "--training-seed", "2", "--override", "ppo.epochs=1"],
        ),
        (
            "scalar-reward",
            "evaluate-policy",
            "scalar_reward.runner",
            "run_command",
            ["--arm", "a2", "--checkpoint", "final", "--checkpoint-path", "policy.pt"],
        ),
        ("ppo-stability", "stage-a", "ppo_stability.runner", "run_command", []),
        ("ppo-stability", "stage-b", "ppo_stability.runner", "run_command", []),
        ("ppo-stability", "stage-c", "ppo_stability.runner", "run_command", []),
        (
            "ppo-stability",
            "diagnose",
            "ppo_stability.runner",
            "run_command",
            ["--diagnostic", "guidance"],
        ),
    ],
)
def test_additional_workflow_actions(monkeypatch, experiment, action, module, function, options):
    from importlib import import_module

    target = import_module("eco_planner.experiments." + module)
    received = []
    monkeypatch.setattr(
        target, function, lambda *args, **kwargs: received.append((args, kwargs)) or {}
    )
    args = cli.build_parser().parse_args([experiment, action, "--output-dir", "out", *options])
    assert args.config.is_file()
    cli.dispatch(args)
    positional, keyword = received[0]
    if action == "collect":
        assert positional == (args.config, args.output_dir) and not keyword
    elif experiment == "ppo-stability":
        assert positional[0] == action
        assert positional[3] == ("guidance" if action == "diagnose" else None)
    elif action == "train":
        assert keyword["arm"] == "a1" and keyword["training_seed"] == 2
        assert keyword["overrides"] == ["ppo.epochs=1"]
    else:
        assert keyword["arm"] == "a2" and keyword["checkpoint_label"] == "final"
        assert keyword["checkpoint_path"] == args.checkpoint_path


def test_scalar_analysis_dispatch_uses_comparison_config(monkeypatch):
    from eco_planner.analysis import runner

    received = []
    monkeypatch.setattr(
        runner, "analyze", lambda *args, **kwargs: received.append((args, kwargs)) or {}
    )
    args = cli.build_parser().parse_args(
        [
            "scalar-reward",
            "analyze",
            "--source-dir",
            "in",
            "--output-dir",
            "out",
            "--config",
            "comparison.yaml",
            "--no-figures",
        ]
    )
    cli.dispatch(args)
    assert received == [
        (
            ("scalar-reward", args.source_dir, args.output_dir),
            {"figures": False, "comparison_config": args.config},
        )
    ]
