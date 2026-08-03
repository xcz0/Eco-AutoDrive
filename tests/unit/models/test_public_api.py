from eco_planner.models import (
    OfficialDiffusionPlannerConfig,
    PretrainedDiffusionPlanner,
    load_official_diffusion_planner,
)
from eco_planner.models.diffusion_planner import DiffusionPlanner
from eco_planner.models.pretrained import CheckpointLoadReport


def test_models_public_api_remains_importable() -> None:
    assert OfficialDiffusionPlannerConfig.__name__ == "OfficialDiffusionPlannerConfig"
    assert PretrainedDiffusionPlanner.__name__ == "PretrainedDiffusionPlanner"
    assert load_official_diffusion_planner.__name__ == "load_official_diffusion_planner"
    assert DiffusionPlanner.__name__ == "DiffusionPlanner"
    assert CheckpointLoadReport.__name__ == "CheckpointLoadReport"
