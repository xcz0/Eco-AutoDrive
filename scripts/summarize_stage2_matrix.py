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
    parser.add_argument("--expected-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--expected-accelerator", choices=("cpu", "cuda"), required=True)
    parser.add_argument(
        "--expected-precision",
        choices=("32-true", "16-mixed", "bf16-mixed"),
        required=True,
    )
    args = parser.parse_args()
    report = summarize_stage2_matrix(
        args.matrix_root,
        args.stage1_ddim_root,
        expected_seeds=tuple(args.expected_seeds),
        expected_accelerator=args.expected_accelerator,
        expected_precision=args.expected_precision,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
