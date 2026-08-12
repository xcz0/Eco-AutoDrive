"""Compare serial and parallel matrix artifacts exactly."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eco_planner.evaluation.comparison import compare_artifact_trees


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("serial_root", type=Path)
    parser.add_argument("parallel_root", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            compare_artifact_trees(args.serial_root, args.parallel_root),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
