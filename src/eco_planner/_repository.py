"""Repository locations needed by internal application workflows."""

from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPOSITORY_ROOT / "configs"
LOCAL_ENVIRONMENT_PATH = REPOSITORY_ROOT / ".env"
