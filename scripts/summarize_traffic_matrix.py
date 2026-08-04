"""Validate and summarize the fixed 20-episode MetaDrive traffic matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from eco_planner.evaluation.matrix import summarize_matrix as _summarize_matrix


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matrix_root", type=Path)
    parser.add_argument(
        "--partial",
        action="store_true",
        help="summarize only jobs with complete job-level summaries without claiming full coverage",
    )
    return parser.parse_args()


def summarize_matrix(matrix_root: Path, *, partial: bool = False) -> dict[str, Any]:
    """Compatibility wrapper for the importable matrix implementation."""

    return _summarize_matrix(matrix_root, partial=partial)


def main() -> None:
    args = _parse_args()
    report = summarize_matrix(args.matrix_root, partial=args.partial)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
