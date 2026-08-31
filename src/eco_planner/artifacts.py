"""Shared JSON and reproducibility metadata helpers for research artifacts."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from collections.abc import Mapping
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pydantic import BaseModel


def write_json(path: Path, payload: BaseModel | dict[str, object]) -> None:
    """Persist stable, UTF-8 JSON for a typed research artifact."""

    value: Any = payload.model_dump(mode="json") if isinstance(payload, BaseModel) else payload
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )


def write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    """Persist named arrays without weakening each artifact schema to ``Any``."""

    # NumPy's stub treats every dynamic keyword as its reserved allow_pickle option.
    np.savez(path, **arrays)  # pyright: ignore[reportArgumentType]


def collect_repository_metadata(repository_root: Path) -> dict[str, object]:
    """Collect source and runtime facts shared by evaluation and training artifacts."""

    return {
        "git_head": _git_output(repository_root, "rev-parse", "HEAD").strip(),
        "git_status_short": tuple(_git_output(repository_root, "status", "--short").splitlines()),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "lightning": version("lightning"),
        "metadrive": version("metadrive-simulator"),
        "pydantic": version("pydantic"),
    }


def write_tracked_diff(path: Path, repository_root: Path) -> None:
    """Persist the tracked source diff associated with an artifact run."""

    path.write_text(
        _git_output(repository_root, "diff", "--binary", "--no-ext-diff"), encoding="utf-8"
    )


def _git_output(repository_root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout
