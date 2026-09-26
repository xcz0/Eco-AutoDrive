"""Markdown formatting shared by independently owned research reports."""

import json
import os
from pathlib import Path
from typing import Any


def evidence_tables(value: Any, title: str = "Evidence") -> str:
    lines = []
    if isinstance(value, dict):
        scalars = [(k, v) for k, v in value.items() if not isinstance(v, (dict, list))]
        if scalars:
            lines += [f"### {title}", "", "| Field | Value |", "| --- | --- |"]
            for key, entry in scalars:
                cell = "undefined / unavailable" if entry is None else str(entry)
                lines.append(f"| {key} | {cell.replace('|', '&#124;').replace(chr(10), '<br>')} |")
            lines.append("")
        for key, entry in value.items():
            if isinstance(entry, (dict, list)):
                lines.append(evidence_tables(entry, f"{title} / {key}"))
    elif isinstance(value, list):
        if all(not isinstance(v, (dict, list)) for v in value):
            lines += [f"### {title}", "", json.dumps(value, ensure_ascii=False), ""]
        else:
            for i, entry in enumerate(value):
                lines.append(evidence_tables(entry, f"{title} / {i}"))
    return "\n".join(lines)


def number(value: float | None) -> str:
    return "undefined" if value is None else f"{value:.9g}"


def write_report(
    experiment: str,
    source: Path,
    output: Path,
    data: dict,
    files: list[str],
    *,
    body: str | None = None,
    description: str | None = None,
    evidence: dict | None = None,
) -> None:
    source_link = Path(os.path.relpath(source, output)).as_posix()
    lines = [
        f"# {experiment} analysis",
        "",
        "Descriptive evidence from saved artifacts. Experimental gates and acceptance "
        "decisions are recorded evidence, not re-executed by this report.",
        "",
        "Undefined vectors and unavailable failed-run metrics are not replaced with zero.",
        "",
        f"[Source artifacts](<{source_link}>) · [Analysis JSON](analysis.json)",
        "",
    ]
    if body is not None:
        lines.append(body)
    if description is not None:
        lines += [
            description,
            "",
            "Saved gates remain experimental decisions. Fixed-batch gradient evidence "
            "alone does not establish learned behavioral improvement.",
            "",
        ]
    for path in files:
        if path.endswith(".png"):
            lines += [
                f"![{Path(path).stem}](<{path}>)",
                "",
                f"[SVG](<{path[:-4]}.svg>) · [PNG](<{path}>)",
                "",
            ]
    if body is None:
        lines.append(evidence_tables(data if evidence is None else evidence))
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
