"""Static dependency-direction contract for evaluation and planning packages.

Evaluation must reuse planning-owned decisions without importing the RL rollout,
optimization, or policy packages. Planning must not depend on RL or evaluation.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[2] / "src" / "eco_planner"

_EVALUATION_FORBIDDEN = (
    "eco_planner.rl.rollout",
    "eco_planner.rl.optimization",
    "eco_planner.rl.policy",
)
_PLANNING_FORBIDDEN = ("eco_planner.rl", "eco_planner.evaluation")


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None and node.level == 0:
            modules.add(node.module)
    return modules


def _package_files(package: str) -> list[Path]:
    return sorted((_SRC / package).rglob("*.py"))


def _offenders(package: str, forbidden: tuple[str, ...]) -> list[tuple[Path, str]]:
    offenders: list[tuple[Path, str]] = []
    for path in _package_files(package):
        for module in _imported_modules(path):
            for prefix in forbidden:
                if module == prefix or module.startswith(prefix + "."):
                    offenders.append((path.relative_to(_SRC), module))
    return offenders


@pytest.mark.parametrize("package,forbidden", [("evaluation", _EVALUATION_FORBIDDEN)])
def test_evaluation_does_not_import_rl_runtime(package: str, forbidden: tuple[str, ...]) -> None:
    offenders = _offenders(package, forbidden)
    assert not offenders, f"{package} imports forbidden RL runtime modules: {offenders}"


@pytest.mark.parametrize("package,forbidden", [("planning", _PLANNING_FORBIDDEN)])
def test_planning_does_not_import_rl_or_evaluation(
    package: str, forbidden: tuple[str, ...]
) -> None:
    offenders = _offenders(package, forbidden)
    assert not offenders, f"{package} imports forbidden downstream modules: {offenders}"
