from __future__ import annotations

import os
from pathlib import Path

from eco_planner.configuration import (
    load_local_environment,
    with_machine_resource_override,
)


def test_missing_local_environment_is_optional(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("MACHINE_NAME", raising=False)

    load_local_environment(tmp_path / "missing.env")

    assert "MACHINE_NAME" not in os.environ


def test_process_environment_wins_over_local_environment(monkeypatch, tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("MACHINE_NAME=rtx3050_laptop\n", encoding="utf-8")
    monkeypatch.setenv("MACHINE_NAME", "rtx_a4000")

    load_local_environment(env_path)

    assert with_machine_resource_override([]) == ["components/resources=rtx_a4000"]


def test_explicit_resource_override_wins_over_machine_selection(monkeypatch) -> None:
    monkeypatch.setenv("MACHINE_NAME", "rtx3050_laptop")

    overrides = with_machine_resource_override(
        ["components/resources=rtx_a4000", "runtime.seed=17"]
    )

    assert overrides == ["components/resources=rtx_a4000", "runtime.seed=17"]
