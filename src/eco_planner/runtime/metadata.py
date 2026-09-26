"""Collect repository and runtime provenance at execution boundaries."""

import platform
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

import torch


def collect_repository_metadata(repository_root: Path) -> dict[str, object]:
    """Collect source and runtime facts shared by evaluation and training artifacts."""

    return {
        "git_head": _git_output(repository_root, "rev-parse", "HEAD").strip(),
        "git_branch": _git_output(repository_root, "rev-parse", "--abbrev-ref", "HEAD").strip(),
        "git_status_short": tuple(_git_output(repository_root, "status", "--short").splitlines()),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "lightning": version("lightning"),
        "metadrive": version("metadrive-simulator"),
        "pydantic": version("pydantic"),
    }


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
