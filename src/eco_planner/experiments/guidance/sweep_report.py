from pathlib import Path


def plot(data: dict, output: Path) -> list[str]:
    from eco_planner.reporting.plots import curves

    comparisons = data["comparisons"]
    files = []
    for label, entry in comparisons.items():
        if "pairs" not in entry:
            continue
        rows = entry["pairs"]
        name = label.replace("/", "-")
        for metric in ("energy_ml", "route_completion"):
            files += curves(
                output,
                name + "-" + metric,
                {
                    side: ([str(r["key"][0]) for r in rows], [r[side][metric] for r in rows])
                    for side in ("reference", "comparison")
                },
                "matched scenario",
                metric,
            )
        files += curves(
            output,
            name + "-energy-progress",
            {
                side: (
                    [r[side]["route_completion"] for r in rows],
                    [r[side]["energy_ml"] for r in rows],
                )
                for side in ("reference", "comparison")
            },
            "route completion",
            "MetaDrive fuel proxy (mL)",
            scatter=True,
        )
    return files


def report_evidence(data: dict) -> dict:
    return {
        "runs": [{k: v for k, v in r.items() if k != "episodes"} for r in data["runs"]],
        "comparisons": {
            key: {
                **{k: v for k, v in entry.items() if k != "pairs"},
                "unavailable_pairs": [r for r in entry.get("pairs", []) if not r["available"]],
            }
            for key, entry in data["comparisons"].items()
        },
    }


def write_report(source: Path, output: Path, data: dict, files: list[str]) -> None:
    from eco_planner.reporting.markdown import write_report as write_markdown

    write_markdown("energy-sweep", source, output, data, files, evidence=report_evidence(data))
