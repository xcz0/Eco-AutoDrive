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

pyright := if os_family() == "windows" {
    ".venv/Scripts/pyright.exe"
} else {
    ".venv/bin/pyright"
}

smoke_test_files := "tests/benchmarking/test_rollout.py tests/configuration/test_jobs.py tests/evaluation/test_runner.py tests/planning/test_guidance.py tests/planning/test_sampling.py tests/simulation/test_geometry.py tests/simulation/test_reward.py tests/training/test_ppo.py"


# Show available commands.
default:
    @just --list


# Create/update the local development environment. Not intended as a routine Agent preflight.
[group('development')]
setup:
    uv sync --all-groups

[group('development')]
lint:
    {{ruff}} check .

[group('development')]
typecheck:
    {{pyright}}

[group('development')]
format:
    {{ruff}} check --fix .
    {{ruff}} format .

# Core cross-workflow smoke set.
[group('development')]
test:
    {{pytest}} {{smoke_test_files}} -m "smoke and not gpu and not simulator and not slow"

# Complete CPU test set excluding simulator and slow cases.
[group('development')]
test-all-cpu:
    {{pytest}} -m "not gpu and not simulator and not slow"

# Run one research workflow's CPU tests.
[group('development')]
test-workflow workflow:
    {{pytest}} tests/{{workflow}} -m "not gpu and not simulator and not slow"

# Run one or more specific test files, directories, or nodes.
[group('development')]
test-target +targets:
    {{pytest}} {{targets}}

# Simulator tests only.
[group('development')]
test-sim:
    {{pytest}} -m "simulator and not gpu and not slow"

# GPU tests only.
[group('development')]
test-gpu:
    {{pytest}} -m "gpu and not slow"

# Full lightweight repository validation.
[group('development')]
check: lint format test-all-cpu


# Run one evaluation job profile. Append Hydra overrides as needed.
[group('evaluation')]
evaluate profile="no_traffic/full" *overrides:
    {{python}} -m scripts.evaluate --config-name jobs/evaluation/{{profile}} {{overrides}}

# Run a no-traffic or traffic Hydra matrix.
[group('evaluation')]
evaluate-matrix mode="traffic" *overrides:
    {{python}} -m scripts.evaluate --config-name jobs/evaluation/{{mode}}/matrix --multirun {{overrides}}

# Run the fixed-seed energy matrix.
[group('evaluation')]
energy output_root *options:
    {{python}} -m scripts.studies.energy_matrix --output-root "{{output_root}}" {{options}}

# Audit fixed synthetic PlannerRFT-style reward cases without running PPO.
[group('evaluation')]
reward-sanity output_root config="configs/studies/reward/sanity.yaml":
    {{python}} -m scripts.studies.reward_sanity --config "{{config}}" --output-root "{{output_root}}"


# Run one training profile with explicit seed and replay identity.
[group('training')]
train profile="ppo/smoke" seed="0" replay="0" *overrides:
    {{python}} -m scripts.train --config-name jobs/training/{{profile}} runtime.seed={{seed}} training.replay_id={{replay}} {{overrides}}

# Run the matched builtin/energy PPO A/B and produce a pending human-review report.
[group('training')]
ppo-reward-ab output_root study="configs/studies/reward/ppo_ab.yaml":
    {{python}} -m scripts.studies.ppo_reward_ab --study "{{study}}" --output-root "{{output_root}}"

# Run Issue #76 PPO stability search and staged validation on the experiment host.
[group('training')]
ppo-stability-stage-a output_root study="configs/studies/ppo/stability.yaml":
    {{python}} -m scripts.studies.ppo_stability stage-a --study "{{study}}" --output-root "{{output_root}}"

[group('training')]
ppo-stability-stage-b output_root study="configs/studies/ppo/stability.yaml":
    {{python}} -m scripts.studies.ppo_stability stage-b --study "{{study}}" --output-root "{{output_root}}"

[group('training')]
ppo-stability-stage-c output_root study="configs/studies/ppo/stability.yaml":
    {{python}} -m scripts.studies.ppo_stability stage-c --study "{{study}}" --output-root "{{output_root}}"

[group('training')]
ppo-stability-diagnose output_root diagnostic study="configs/studies/ppo/stability.yaml":
    {{python}} -m scripts.studies.ppo_stability diagnose --study "{{study}}" --output-root "{{output_root}}" --diagnostic "{{diagnostic}}"


# Run one reusable benchmark profile.
[group('benchmark')]
benchmark profile="throughput" *overrides:
    {{python}} -m scripts.benchmark --config-name jobs/benchmark/{{profile}} {{overrides}}

# Consolidate serial, job-level, and vector evaluation measurements.
[group('benchmark')]
benchmark-report serial job_level vector serial_wall_s job_level_wall_s vector_wall_s:
    {{python}} -m scripts.benchmarking.evaluation_report "{{serial}}" "{{job_level}}" "{{vector}}" --serial-wall-s {{serial_wall_s}} --job-level-wall-s {{job_level_wall_s}} --vector-wall-s {{vector_wall_s}}


[group('analysis')]
summarize-matrix root *options:
    {{python}} -m scripts.analysis.evaluation_matrix "{{root}}" {{options}}

[group('analysis')]
summarize-training root:
    {{python}} -m scripts.analysis.training "{{root}}"

[group('analysis')]
summarize-ppo-stability root study="configs/studies/ppo/stability.yaml":
    {{python}} -m scripts.studies.ppo_stability summarize --study "{{study}}" --output-root "{{root}}"

[group('analysis')]
review-ppo-reward-ab root:
    {{python}} -m scripts.analysis.ppo_reward_ab "{{root}}"
