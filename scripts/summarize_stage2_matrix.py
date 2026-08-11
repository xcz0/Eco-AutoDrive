"""Validate and summarize the pre-registered stage-2 guidance matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eco_planner.evaluation.stage2_matrix import summarize_stage2_matrix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix_root", type=Path)
    parser.add_argument("stage1_ddim_root", type=Path)
    args = parser.parse_args()
    report = summarize_stage2_matrix(args.matrix_root, args.stage1_ddim_root)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
