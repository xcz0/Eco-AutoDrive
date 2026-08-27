"""Repository paths shared by non-installed script modules."""

from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = REPOSITORY_ROOT / "configs"
LOCAL_ENVIRONMENT_PATH = REPOSITORY_ROOT / ".env"
