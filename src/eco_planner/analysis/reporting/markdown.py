"""Markdown presentation of structured evidence, including unavailable values."""

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


def write_report(experiment: str, source: Path, output: Path, data: dict, files: list[str]) -> None:
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
    scalar_comparison = experiment == "scalar-reward" and "contrasts" in data
    if scalar_comparison:
        from .scalar import render_scalar

        lines.append(render_scalar(data))
    chain = {
        "reward-sanity": "Reward: synthetic checks establish component correctness, not learning.",
        "reward-calibration": "Reward → credit assignment → actor gradient: compare original and "
        "calibrated component scales, advantage changes and gradient response on the same batch.",
        "lambda-identifiability": "Reward → credit assignment → actor gradient: follow lambda "
        "changes through raw/normalized advantages and actor-head/shared-trunk gradients.",
        "objective-decomposition": "Reward → credit assignment → actor gradient: attribute "
        "separation or attenuation to reward components and advantage normalization.",
        "critic-gae-ablation": "Credit assignment → actor gradient: compare value targets and "
        "advantage forms to locate attenuation before the actor backward pass.",
    }
    if experiment in chain:
        lines += [
            chain[experiment],
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
    if experiment in ("lambda-identifiability", "objective-decomposition", "critic-gae-ablation"):
        from .fixed import (
            render_ablation_report,
            render_decomposition_report,
            render_report,
        )

        renderer = {
            "lambda-identifiability": render_report,
            "objective-decomposition": render_decomposition_report,
            "critic-gae-ablation": render_ablation_report,
        }[experiment]
        body = (
            render_report(data, batch_origin="Reused fixed source batch")
            if experiment == "lambda-identifiability"
            else renderer(data)
        )
        for filename in ("summary.json", "diagnostics.npz", "sample_index.json"):
            body = body.replace(f"]({filename})", f"](<{source_link}/{filename}>)")
        lines.append(body)
    elif experiment == "reward-calibration":
        from .fixed import render_report

        lines.append(
            evidence_tables(
                {k: v for k, v in data["experiment"].items() if k != "comparisons"},
                "Recorded experiment",
            )
        )
        for label in ("original", "calibrated"):
            body = render_report(data[label], batch_origin=f"Reused batch: {label}")
            for filename in ("summary.json", "diagnostics.npz"):
                body = body.replace(f"]({filename})", f"](<{source_link}/{label}/{filename}>)")
            body = body.replace("](sample_index.json)", f"](<{source_link}/sample_index.json>)")
            lines.append(body)
        lines.append(
            evidence_tables(
                {k: v for k, v in data["audit"].items() if k in ("all", "interpretation")},
                "Recorded audit",
            )
        )
        lines.append(
            evidence_tables(
                {k: v["all"] for k, v in data["distributions"].items()},
                "Recomputed audit distributions",
            )
        )
        lines.append(
            "\nPer-scenario and per-planning-cycle evidence: [analysis.json](analysis.json)."
        )
    elif not scalar_comparison:
        lines.append(evidence_tables(report_evidence(experiment, data)))
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def report_evidence(experiment: str, data: dict) -> dict:
    """Keep reviewable summaries in Markdown; full structured evidence remains in JSON."""

    def comparison(entry: dict) -> dict:
        return {
            **{k: v for k, v in entry.items() if k != "pairs"},
            "unavailable_pairs": [r for r in entry.get("pairs", []) if not r["available"]],
        }

    if experiment == "energy-sweep":
        return {
            "runs": [{k: v for k, v in r.items() if k != "episodes"} for r in data["runs"]],
            "comparisons": {k: comparison(v) for k, v in data["comparisons"].items()},
        }
    if experiment == "ppo-stability":
        return {k: v for k, v in data.items() if k != "training_curves"}
    if experiment == "execution-backend":
        return {
            "modes": data["modes"],
            "recorded_provenance": data["recorded_comparison"].get("provenance"),
        }
    return data
