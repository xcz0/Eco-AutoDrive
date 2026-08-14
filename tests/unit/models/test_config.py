from __future__ import annotations

import json
from pathlib import Path

from eco_planner.models.config import OfficialDiffusionPlannerConfig


def test_official_config_builds_normalizers_from_checkpoint_args(
    tmp_path: Path,
    official_config_args: dict[str, object],
) -> None:
    path = tmp_path / "args.json"
    path.write_text(json.dumps(official_config_args), encoding="utf-8")

    config = OfficialDiffusionPlannerConfig.from_json(path)

    assert config.future_len == 80
    assert config.predicted_neighbor_num == 10
    assert config.checkpoint_device == "cuda"
    assert config.observation_feature_dimensions["lanes"] == 12
    assert config.state_normalizer.mean.shape == (11, 1, 4)
