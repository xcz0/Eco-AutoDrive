"""Load the repository-local environment before Hydra resolves configuration."""

from __future__ import annotations

import os
import re
from pathlib import Path

_ASSIGNMENT = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$")


def load_local_environment(env_path: Path | None = None) -> None:
    """Load ``.env`` from the repository root without replacing shell variables."""
    env_path = env_path or Path(__file__).resolve().parents[1] / ".env"
    if not env_path.is_file():
        raise FileNotFoundError(f"missing local environment file: {env_path}")

    lines = env_path.read_text(encoding="utf-8").splitlines()
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _ASSIGNMENT.fullmatch(line)
        if match is None:
            raise ValueError(f"invalid .env assignment at {env_path}:{line_number}")
        name, value = match.groups()
        os.environ.setdefault(name, value)
