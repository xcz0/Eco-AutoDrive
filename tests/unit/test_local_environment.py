from __future__ import annotations

import os
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from local_environment import load_local_environment


def test_load_local_environment_reads_machine_name(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("MACHINE_NAME=rtx_a4000\n", encoding="utf-8")
    monkeypatch.delenv("MACHINE_NAME")

    load_local_environment(env_path)

    assert os.environ["MACHINE_NAME"] == "rtx_a4000"


def test_training_config_uses_machine_name_for_its_default_resource_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MACHINE_NAME", "rtx_a4000")
    config_dir = Path(__file__).parents[2] / "configs"

    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        config = compose(
            config_name="jobs/training/ppo_smoke",
            overrides=["runtime.seed=0", "training.replay_id=0"],
        )

    assert config.resources.name == "rtx_a4000"
