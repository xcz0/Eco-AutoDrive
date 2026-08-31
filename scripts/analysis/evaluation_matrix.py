"""CLI adapter for evaluation-matrix analysis."""

import argparse
import json
from pathlib import Path

from eco_planner.analysis.evaluation_matrix import summarize_evaluation_matrix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix_root", type=Path)
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args()
    report = summarize_evaluation_matrix(args.matrix_root, partial=args.partial)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
