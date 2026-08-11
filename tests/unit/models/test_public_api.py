from eco_planner.models import (
    Ddim5SamplerConfig,
    Dpm10SamplerConfig,
    NoGuidanceConfig,
    OfficialDiffusionPlannerConfig,
    OrthogonalReferenceGuidanceConfig,
    PlannerInferenceResult,
    PretrainedDiffusionPlanner,
    load_official_diffusion_planner,
    parse_guidance_config,
    parse_sampler_config,
)
from eco_planner.models.diffusion_planner import DiffusionPlanner
from eco_planner.models.pretrained import CheckpointLoadReport


def test_models_public_api_remains_importable() -> None:
    assert OfficialDiffusionPlannerConfig.__name__ == "OfficialDiffusionPlannerConfig"
    assert Ddim5SamplerConfig.__name__ == "Ddim5SamplerConfig"
    assert Dpm10SamplerConfig.__name__ == "Dpm10SamplerConfig"
    assert NoGuidanceConfig.__name__ == "NoGuidanceConfig"
    assert OrthogonalReferenceGuidanceConfig.__name__ == "OrthogonalReferenceGuidanceConfig"
    assert PlannerInferenceResult.__name__ == "PlannerInferenceResult"
    assert PretrainedDiffusionPlanner.__name__ == "PretrainedDiffusionPlanner"
    assert load_official_diffusion_planner.__name__ == "load_official_diffusion_planner"
    assert parse_sampler_config.__name__ == "parse_sampler_config"
    assert parse_guidance_config.__name__ == "parse_guidance_config"
    assert DiffusionPlanner.__name__ == "DiffusionPlanner"
    assert CheckpointLoadReport.__name__ == "CheckpointLoadReport"
