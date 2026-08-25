"""Validate and summarize the fixed 20-episode MetaDrive traffic matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eco_planner.evaluation.analysis import summarize_matrix


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matrix_root", type=Path)
    parser.add_argument(
        "--partial",
        action="store_true",
        help="summarize only jobs with complete job-level summaries without claiming full coverage",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = summarize_matrix(args.matrix_root, partial=args.partial)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
