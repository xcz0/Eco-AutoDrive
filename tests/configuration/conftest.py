from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

CONFIG_ROOT = Path(__file__).resolve().parents[2] / "configs"
ComposeConfig = Callable[[str, list[str] | None], DictConfig]


@pytest.fixture(scope="module")
def config_root() -> Path:
    return CONFIG_ROOT


@pytest.fixture(scope="module")
def compose_config() -> Iterator[ComposeConfig]:
    def _compose(config_name: str, overrides: list[str] | None = None) -> DictConfig:
        return compose(config_name=config_name, overrides=overrides or [])

    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_ROOT)):
        yield _compose
