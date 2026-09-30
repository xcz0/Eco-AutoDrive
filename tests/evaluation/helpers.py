"""Synthetic evaluation and training summaries for artifact/report tests."""

from __future__ import annotations

from eco_planner.evaluation import (
    CompletedEpisodeSummary,
    EnergySummary,
    EpisodeMetrics,
    ErrorValues,
    ExecutionErrorSummary,
    MapInputAudit,
    NoGuidanceSummary,
    SamplerSummary,
    ScenarioSummary,
    SpeedSummary,
    TerminationSummary,
    TrafficObservationSummary,
    WarmupSummary,
)
from eco_planner.rl.artifacts import PolicyProbeSummary, TrainingRunSummary, TrainingUpdateSummary


def _episode(*, seed: int, distance_m: float, energy_ml: float) -> CompletedEpisodeSummary:
    return CompletedEpisodeSummary(
        scenario=ScenarioSummary(name="traffic", map_sequence="S", seed=seed),
        evaluation_mode="traffic",
        traffic_density=0.2,
        route_length_m=2_500.0,
        noise_seed=seed,
        sampler=SamplerSummary(
            name="ddim5",
            implementation="diffusers",
            num_steps=5,
            timesteps=None,
            initial_noise_scale=1.0,
            ddim_stochasticity=0.0,
            parity_label="fixture",
        ),
        guidance=NoGuidanceSummary(name="none"),
        plan_cycles=2,
        simulator_steps=10,
        environment_steps_including_warmup=10,
        metrics=EpisodeMetrics(
            simulated_seconds=1.0,
            distance_m=distance_m,
            energy=EnergySummary(
                metric="metadrive_fuel_proxy",
                total_ml=energy_ml,
                distance_m=distance_m,
                ml_per_km=energy_ml * 1_000.0 / distance_m,
            ),
            speed_mps=SpeedSummary(minimum=5.0, mean=6.0 + seed, maximum=7.0),
            stopped_fraction=0.0,
            route_completion=0.4 + 0.1 * seed,
            arrive_dest=seed == 1,
            collision=False,
            out_of_road=False,
            wrong_direction=False,
            wrong_direction_fraction=0.0,
        ),
        crash_vehicle=False,
        crash_object=False,
        crash_building=False,
        crash_human=False,
        crash_sidewalk=False,
        terminated=True,
        truncated=False,
        terminal_reason="arrive_dest",
        termination=TerminationSummary(type="arrive_dest", detail="fixture"),
        map_input_audit=MapInputAudit(
            speed_limit_sentinel_replaced_count=0,
            speed_limit_existing_preserved_count=1,
            configured_programmatic_lane_speed_limit_kmh=60.0,
            lane_speed_limit_kmh_counts={"60": 1},
            valid_lane_count_min=1,
            valid_lane_count_max=1,
            speed_limit_valid_count_min=1,
            speed_limit_valid_count_max=1,
            speed_limit_mps_min=60.0 / 3.6,
            speed_limit_mps_max=60.0 / 3.6,
            speed_limit_mps_unique_values=(60.0 / 3.6,),
        ),
        history_warmup=WarmupSummary(
            simulator_steps=0,
            simulated_seconds=0.0,
            ego_displacement_m_maximum=0.0,
            participant_count_minimum=0,
            participant_count_maximum=0,
        ),
        traffic_observation=TrafficObservationSummary(
            planning_frames=2,
            frames_with_participants=1,
            frames_with_participants_fraction=0.5,
            participant_count_minimum=0,
            participant_count_maximum=1,
        ),
        trajectory_execution_error=ExecutionErrorSummary(
            position_m=ErrorValues(maximum=0.0, mean=0.0, final=0.0),
            heading_rad=ErrorValues(maximum=0.0, mean=0.0, final=0.0),
        ),
    )


def _update(
    index: int,
    reward: float,
    *,
    action: float = 0.0,
    fuel: float = 2.0,
    distance: float = 20.0,
    progress: float = 2.0,
    speed: float = 10.0,
    collision: int = 0,
    out_of_road: int = 0,
) -> TrainingUpdateSummary:
    return TrainingUpdateSummary.model_construct(
        update_index=index,
        sample_count=2,
        mean_episode_length=2.0,
        total_reward=reward,
        route_completion_delta=progress,
        mean_speed_mps=speed,
        stopped_fraction=0.0,
        collision_count=collision,
        out_of_road_count=out_of_road,
        action_mean=(0.0, action),
        action_std=(0.0, 0.0),
        executed_fuel_proxy_total_ml=fuel,
        executed_fuel_proxy_distance_m=distance,
        maximum_pre_clip_gradient_norm=1.0,
        mean_policy_loss=-0.1,
        mean_value_loss=0.2,
        mean_entropy=0.3,
        mean_approximate_kl=0.01,
    )


def _training_summary(
    seed: int,
    replay: int,
    *,
    post_update: TrainingUpdateSummary | None = None,
) -> TrainingRunSummary:
    probe_before = PolicyProbeSummary.model_construct(
        alpha=((1.0, 1.0), (1.0, 1.0)),
        beta=((1.0, 1.0), (1.0, 1.0)),
        guidance_mean=((0.0, 0.0), (0.0, 0.0)),
        boundary_mass=((0.0, 0.0), (0.0, 0.0)),
    )
    probe_after = PolicyProbeSummary.model_construct(
        alpha=((1.1, 1.0), (1.0, 1.1)),
        beta=((1.0, 1.1), (1.1, 1.0)),
        guidance_mean=((0.1, 0.0), (0.0, 0.1)),
        boundary_mass=((0.0, 0.0), (0.0, 0.0)),
    )
    return TrainingRunSummary.model_construct(
        status="completed",
        training_seed=seed,
        replay_id=replay,
        noise_seeds=(seed * 10 + 1,),
        policy_action_seeds=(seed * 10 + 2,),
        total_transitions=4,
        initial_policy_hash="a" * 64,
        final_policy_hash="b" * 64,
        frozen_planner_hash_before="c" * 64,
        frozen_planner_hash_after="c" * 64,
        probe_before=probe_before,
        probe_after=probe_after,
        updates=(_update(0, 1.0), post_update or _update(1, 2.0)),
        reward_profile="plannerrft_energy_v1",
    )
