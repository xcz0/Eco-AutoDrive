"""Source separation and derived analysis artifact persistence."""

from pathlib import Path
from typing import Any

from eco_planner.artifacts import write_json


def separate_output(
    source: Path, output: Path, *, sources: tuple[Path, ...] = ()
) -> tuple[Path, Path]:
    source, output = source.resolve(strict=True), output.resolve()
    if any(
        s == output or s.is_relative_to(output) or output.is_relative_to(s)
        for s in (source, *(p.resolve(strict=True) for p in sources))
    ):
        raise ValueError("offline output must be separate from the source directory")
    return source, output


def write_analysis(
    output: Path,
    data: dict[str, Any],
    files: list[str],
    *,
    experiment: str | None = None,
    source: Path | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"evidence": data, "figures": files}
    if experiment is not None:
        payload["experiment"] = experiment
    if source is not None:
        payload["source"] = str(source.resolve())
    write_json(output / "analysis.json", payload)
    return payload
