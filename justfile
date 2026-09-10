# Eco-AutoDrive Windows developer and experiment entrypoints.
# Keep experiment semantics in configs/ and application logic in src/eco_planner/.

set shell := ["pwsh.exe", "-NoLogo", "-NoProfile", "-Command"]

python := ".venv/Scripts/python.exe"
pytest := ".venv/Scripts/pytest.exe"
ruff := ".venv/Scripts/ruff.exe"
pyright := ".venv/Scripts/pyright.exe"

# Show available commands.
default:
    @just --list

# Create/update the local development environment. Not intended as a routine Agent preflight.
[group('development')]
setup:
    uv sync --all-groups

[group('development')]
lint:
    {{ ruff }} check .

[group('development')]
typecheck:
    {{ pyright }}

[group('development')]
format:
    {{ ruff }} check --fix .
    {{ ruff }} format .
    just --fmt

[group('development')]
format-check:
    {{ ruff }} format --check .
    just --fmt --check

# Core cross-workflow smoke set. Membership is defined only by the pytest marker.
[group('development')]
test:
    {{ pytest }} -m "smoke and not gpu and not simulator and not slow"

# Complete CPU test set excluding simulator and slow cases.
[group('development')]
test-all-cpu:
    {{ pytest }} -m "not gpu and not simulator and not slow"

# Run one research workflow's CPU tests.
[group('development')]
test-workflow workflow:
    {{ pytest }} tests/{{ workflow }} -m "not gpu and not simulator and not slow"

# Run one or more specific test files, directories, or nodes.
[group('development')]
test-target +targets:
    {{ pytest }} {{ targets }}

# Simulator tests only.
[group('development')]
test-sim:
    {{ pytest }} -m "simulator and not gpu and not slow"

# GPU tests only.
[group('development')]
test-gpu:
    {{ pytest }} -m "gpu and not slow"

# Full read-only repository validation.
[group('development')]
check: lint format-check typecheck test-all-cpu

# Run a configured evaluation job or summarize an evaluation matrix.
[group('evaluation')]
evaluation action *arguments:
    if ("{{ action }}" -eq "run") { & {{ python }} -m scripts.evaluation {{ arguments }}; exit $LASTEXITCODE } elseif ("{{ action }}" -eq "matrix-report") { & {{ python }} -m scripts.evaluation_matrix {{ arguments }}; exit $LASTEXITCODE } else { throw "evaluation action must be run or matrix-report" }

# Run a configured PPO job.
[group('training')]
training action *arguments:
    if ("{{ action }}" -ne "run") { throw "training action must be run" } else { & {{ python }} -m scripts.training {{ arguments }}; exit $LASTEXITCODE }

# Forward MLflow CLI commands for tracking and the local UI.
[group('training')]
mlflow action *arguments:
    & .venv/Scripts/mlflow.exe {{ action }} {{ arguments }}
    exit $LASTEXITCODE

# Run a configured benchmark.
[group('benchmark')]
benchmark action *arguments:
    if ("{{ action }}" -ne "run") { throw "benchmark action must be run" } else { & {{ python }} -m scripts.benchmark {{ arguments }}; exit $LASTEXITCODE }

# Forward experiment selection, actions and options to the unified CLI.
[group('experiments')]
experiment experiment action *arguments:
    & {{ python }} -m scripts.experiments {{ experiment }} {{ action }} {{ arguments }}
    exit $LASTEXITCODE
