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

# Run a configured PPO job or summarize reproducibility artifacts.
[group('training')]
training action *arguments:
    if ("{{ action }}" -eq "run") { & {{ python }} -m scripts.training {{ arguments }}; exit $LASTEXITCODE } elseif ("{{ action }}" -eq "reproducibility-report") { & {{ python }} -m scripts.experiments.ppo_reproducibility {{ arguments }}; exit $LASTEXITCODE } else { throw "training action must be run or reproducibility-report" }

# Run a configured benchmark or consolidate evaluation-backend measurements.
[group('benchmark')]
benchmark action *arguments:
    if ("{{ action }}" -eq "run") { & {{ python }} -m scripts.benchmark {{ arguments }}; exit $LASTEXITCODE } elseif ("{{ action }}" -eq "evaluation-report") { & {{ python }} -m scripts.experiments.execution_backend {{ arguments }}; exit $LASTEXITCODE } else { throw "benchmark action must be run or evaluation-report" }

# Run the fixed energy-sweep experiment.
[group('experiments')]
energy action *arguments:
    if ("{{ action }}" -ne "run") { throw "energy action must be run" } else { & {{ python }} -m scripts.experiments.energy_sweep {{ arguments }}; exit $LASTEXITCODE }

# Audit fixed synthetic PlannerRFT-style reward cases without running PPO.
[group('experiments')]
reward-sanity action *arguments:
    if ("{{ action }}" -ne "run") { throw "reward-sanity action must be run" } else { & {{ python }} -m scripts.experiments.reward_sanity {{ arguments }}; exit $LASTEXITCODE }

# Run or report the matched builtin/energy PPO A/B experiment.
[group('experiments')]
ppo-reward-ab action *arguments:
    & {{ python }} -m scripts.experiments.ppo_reward_ab {{ action }} {{ arguments }}
    exit $LASTEXITCODE

# Run staged PPO stability search, validation, diagnostics, or summary.
[group('experiments')]
ppo-stability action *arguments:
    & {{ python }} -m scripts.experiments.ppo_stability {{ action }} {{ arguments }}
    exit $LASTEXITCODE
