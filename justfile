# Eco-AutoDrive developer and experiment entrypoints.
# Keep experiment semantics in configs/ and domain logic in src/eco_planner/.

set shell := ["pwsh.exe", "-NoLogo", "-NoProfile", "-Command"]

python := if os_family() == "windows" {
    ".venv/Scripts/python.exe"
} else {
    ".venv/bin/python"
}

pytest := if os_family() == "windows" {
    ".venv/Scripts/pytest.exe"
} else {
    ".venv/bin/pytest"
}

ruff := if os_family() == "windows" {
    ".venv/Scripts/ruff.exe"
} else {
    ".venv/bin/ruff"
}


# Show available commands.
default:
    @just --list


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

# Create/update the local development environment.
# Not intended as a routine Agent preflight.
setup:
    uv sync --all-groups


# ---------------------------------------------------------------------------
# Static checks
# ---------------------------------------------------------------------------

lint:
    {{ruff}} check .

format-check:
    {{ruff}} format --check .

format:
    {{ruff}} check --fix .
    {{ruff}} format .


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

# Default fast test set.
test:
    {{pytest}} -m "not gpu and not simulator and not slow"

# Run one or more specific test files, directories, or nodes.
test-target +targets:
    {{pytest}} {{targets}}

# Simulator tests only.
test-sim:
    {{pytest}} -m "simulator and not gpu and not slow"

# GPU tests only.
test-gpu:
    {{pytest}} -m "gpu and not slow"

# Full lightweight repository validation.
check: lint format-check test


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

# Fast no-traffic smoke evaluation. Append Hydra overrides as needed.
eval-smoke *overrides:
    {{python}} scripts/evaluate.py --config-name experiment/evaluate_no_traffic_smoke {{overrides}}

# Fast traffic smoke evaluation. Append Hydra overrides as needed.
eval-traffic-smoke *overrides:
    {{python}} scripts/evaluate.py --config-name experiment/evaluate_traffic_smoke {{overrides}}

# Standard no-traffic evaluation. Append Hydra overrides as needed.
eval *overrides:
    {{python}} scripts/evaluate.py --config-name experiment/evaluate_no_traffic_full {{overrides}}

# Standard traffic evaluation. Append Hydra overrides as needed.
eval-traffic *overrides:
    {{python}} scripts/evaluate.py --config-name experiment/evaluate_traffic_full {{overrides}}

# Predefined no-traffic matrix. Append Hydra overrides as needed.
eval-no-traffic-matrix *overrides:
    {{python}} scripts/evaluate.py --config-name experiment/evaluate_no_traffic_matrix --multirun {{overrides}}

# Predefined traffic matrix. Append Hydra overrides as needed.
eval-matrix *overrides:
    {{python}} scripts/evaluate.py --config-name experiment/evaluate_traffic_matrix --multirun {{overrides}}

# Smoke test for reference-centered guidance. Append scale or other overrides as needed.
eval-guidance *overrides:
    {{python}} scripts/evaluate.py --config-name experiment/evaluate_no_traffic_smoke planner/sampler=ddim5 planner/guidance=orthogonal_reference {{overrides}}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

# Closed-loop PPO smoke training. Optional arguments: seed, replay ID.
train-smoke seed="0" replay="0":
    {{python}} scripts/train.py --config-name experiment/train_ppo_smoke runtime.seed={{seed}} training.replay_id={{replay}}


# ---------------------------------------------------------------------------
# Analysis / diagnostics
# ---------------------------------------------------------------------------

benchmark-env *overrides:
    {{python}} scripts/benchmark_envs.py {{overrides}}

# Reference-only planner batch and vector-environment scaling. Append Hydra overrides.
benchmark-throughput *overrides:
    {{python}} scripts/benchmark_throughput.py {{overrides}}

benchmark-throughput-traffic *overrides:
    {{python}} scripts/benchmark_throughput.py --config-name experiment/benchmark_throughput_traffic {{overrides}}

# Policy-guided fixed-slot rollout and PPO-update scaling. Append Hydra overrides.
benchmark-rollout *overrides:
    {{python}} scripts/benchmark_rollout.py {{overrides}}

# Consolidate completed serial, Hydra job-level, and vector evaluation measurements.
benchmark-eval-report serial job_level vector serial_wall_s job_level_wall_s vector_wall_s:
    {{python}} scripts/report_evaluation_benchmark.py {{serial}} {{job_level}} {{vector}} --serial-wall-s {{serial_wall_s}} --job-level-wall-s {{job_level_wall_s}} --vector-wall-s {{vector_wall_s}}

summarize-training root:
    {{python}} scripts/summarize_training.py {{root}}

summarize-matrix root:
    {{python}} scripts/summarize_traffic_matrix.py {{root}}

compare-eval serial parallel:
    {{python}} scripts/compare_evaluation_artifacts.py {{serial}} {{parallel}}
