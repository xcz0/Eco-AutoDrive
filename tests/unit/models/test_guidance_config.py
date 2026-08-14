from omegaconf import OmegaConf

from eco_planner.models.config import (
    NoGuidanceConfig,
    OrthogonalPolicyGuidanceConfig,
    OrthogonalReferenceGuidanceConfig,
    parse_guidance_config,
)


def _orthogonal_profile(name: str) -> dict[str, object]:
    profile: dict[str, object] = {
        "name": name,
        "formula_label": "centered_energy_gradient_delta_v1",
        "lateral_max_offset_m": 2.5,
        "longitudinal_max_speed_fraction": 0.25,
        "trajectory_dt_s": 0.1,
        "gradient_step_coefficient": 1.0,
        "reference_refresh_cycles": 1,
        "share_scene_encoding": True,
        "share_initial_noise": True,
        "share_transition_noise": True,
        "heading_norm_epsilon": 1e-6,
        "zero_speed_tolerance_mps": 1e-6,
    }
    if name == "orthogonal_reference":
        profile.update(lateral_scale=0.0, longitudinal_scale=0.0)
    return profile


def test_guidance_config_parses_canonical_profiles() -> None:
    none = parse_guidance_config(OmegaConf.create({"name": "none"}))
    reference = parse_guidance_config(OmegaConf.create(_orthogonal_profile("orthogonal_reference")))
    policy = parse_guidance_config(OmegaConf.create(_orthogonal_profile("orthogonal_policy")))

    assert isinstance(none, NoGuidanceConfig)
    assert isinstance(reference, OrthogonalReferenceGuidanceConfig)
    assert reference.fixed_action == (0.0, 0.0)
    assert isinstance(policy, OrthogonalPolicyGuidanceConfig)
