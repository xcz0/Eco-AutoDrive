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


# Create/update the local development environment. Not intended as a routine Agent preflight.
[group('development')]
setup:
    uv sync --all-groups

[group('development')]
lint:
    {{ruff}} check .

[group('development')]
format:
    {{ruff}} check --fix .
    {{ruff}} format .

# Default fast test set.
[group('development')]
test:
    {{pytest}} -m "not gpu and not simulator and not slow"

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
check: lint format test


# Run one evaluation job profile. Append Hydra overrides as needed.
[group('evaluation')]
evaluate profile="no_traffic" *overrides:
    {{python}} scripts/evaluate.py --config-name jobs/evaluation/{{profile}} {{overrides}}

# Run a no-traffic or traffic Hydra matrix.
[group('evaluation')]
evaluate-matrix mode="traffic" *overrides:
    {{python}} scripts/evaluate.py --config-name jobs/evaluation/{{mode}}_matrix --multirun {{overrides}}

# Run the fixed-seed energy matrix.
[group('evaluation')]
energy output_root *options:
    {{python}} scripts/energy_matrix.py --output-root "{{output_root}}" {{options}}


# Run one training profile with explicit seed and replay identity.
[group('training')]
train profile="ppo_smoke" seed="0" replay="0" *overrides:
    {{python}} scripts/train.py --config-name jobs/training/{{profile}} runtime.seed={{seed}} training.replay_id={{replay}} {{overrides}}


# Run one reusable benchmark profile.
[group('benchmark')]
benchmark profile="throughput" *overrides:
    {{python}} scripts/benchmark.py --config-name jobs/benchmark/{{profile}} {{overrides}}

# Consolidate serial, job-level, and vector evaluation measurements.
[group('benchmark')]
benchmark-report serial job_level vector serial_wall_s job_level_wall_s vector_wall_s:
    {{python}} scripts/benchmark_report.py "{{serial}}" "{{job_level}}" "{{vector}}" --serial-wall-s {{serial_wall_s}} --job-level-wall-s {{job_level_wall_s}} --vector-wall-s {{vector_wall_s}}


[group('analysis')]
summarize-matrix root *options:
    {{python}} scripts/summarize_matrix.py "{{root}}" {{options}}

[group('analysis')]
summarize-training root:
    {{python}} scripts/summarize_training.py "{{root}}"
