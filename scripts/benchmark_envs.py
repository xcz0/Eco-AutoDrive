"""Measure the planner-facing MetaDrive environment cycle without model inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eco_planner.evaluation.environment_benchmark import (
    BenchmarkAcceptanceError,
    benchmark_environment,
)
from eco_planner.models import OfficialDiffusionPlannerConfig


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--args-path",
        type=Path,
        default=Path("checkpoints/DP-Origin/args.json"),
    )
    parser.add_argument("--traffic-baseline-ms", type=float)
    parser.add_argument("--no-traffic-baseline-ms", type=float)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    model_config = OfficialDiffusionPlannerConfig.from_json(args.args_path)
    try:
        result = benchmark_environment(
            model_config,
            traffic_baseline_ms=args.traffic_baseline_ms,
            no_traffic_baseline_ms=args.no_traffic_baseline_ms,
        )
    except BenchmarkAcceptanceError as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
