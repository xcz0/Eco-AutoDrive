from __future__ import annotations

import json
from pathlib import Path

from omegaconf import OmegaConf

from eco_planner._repository import REPOSITORY_ROOT
from eco_planner.analysis.runner import publish
from eco_planner.artifacts import write_json
from eco_planner.experiments.reward_sanity.config import load_sanity_config
from eco_planner.experiments.reward_sanity.diagnostics import evaluate_sanity


def run_sanity(config_path: Path, output_root: Path, *, figures: bool = True) -> int:
    config = load_sanity_config(config_path)
    output_root.mkdir(parents=True, exist_ok=False)
    OmegaConf.save(OmegaConf.load(config_path), output_root / "sanity_manifest.yaml", resolve=True)
    reward_path = (REPOSITORY_ROOT / config.reward_config).resolve()
    OmegaConf.save(OmegaConf.load(reward_path), output_root / "resolved_reward.yaml", resolve=True)
    report = evaluate_sanity(config)
    write_json(output_root / "sanity_report.json", report)
    publish("reward-sanity", output_root, output_root, figures=figures)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" else 1
