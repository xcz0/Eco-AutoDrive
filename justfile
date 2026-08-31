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

# Run the default evaluation profile or pass Hydra arguments unchanged.
[group('evaluation')]
evaluate *arguments:
    {{ python }} -m scripts.evaluate {{ arguments }}

# Run the fixed energy study CLI.
[group('evaluation')]
energy *arguments:
    {{ python }} -m scripts.studies.energy_matrix {{ arguments }}

# Audit fixed synthetic PlannerRFT-style reward cases without running PPO.
[group('evaluation')]
reward-sanity *arguments:
    {{ python }} -m scripts.studies.reward_sanity {{ arguments }}

# Run the default PPO training profile or pass Hydra arguments unchanged.
[group('training')]
train *arguments:
    {{ python }} -m scripts.train {{ arguments }}

# Run the matched builtin/energy PPO A/B study.
[group('training')]
ppo-reward-ab *arguments:
    {{ python }} -m scripts.studies.ppo_reward_ab {{ arguments }}

# Run Issue #76 PPO stability search and staged validation.
[group('training')]
ppo-stability-stage-a *arguments:
    {{ python }} -m scripts.studies.ppo_stability stage-a {{ arguments }}

[group('training')]
ppo-stability-stage-b *arguments:
    {{ python }} -m scripts.studies.ppo_stability stage-b {{ arguments }}

[group('training')]
ppo-stability-stage-c *arguments:
    {{ python }} -m scripts.studies.ppo_stability stage-c {{ arguments }}

[group('training')]
ppo-stability-diagnose *arguments:
    {{ python }} -m scripts.studies.ppo_stability diagnose {{ arguments }}

# Run the default benchmark profile or pass Hydra arguments unchanged.
[group('benchmark')]
benchmark *arguments:
    {{ python }} -m scripts.benchmark {{ arguments }}

# Consolidate serial, job-level, and vector evaluation measurements.
[group('benchmark')]
benchmark-report *arguments:
    {{ python }} -m scripts.benchmarking.evaluation_report {{ arguments }}

[group('analysis')]
summarize-matrix *arguments:
    {{ python }} -m scripts.analysis.evaluation_matrix {{ arguments }}

[group('analysis')]
summarize-training *arguments:
    {{ python }} -m scripts.analysis.training {{ arguments }}

[group('analysis')]
summarize-ppo-stability *arguments:
    {{ python }} -m scripts.studies.ppo_stability summarize {{ arguments }}

[group('analysis')]
review-ppo-reward-ab *arguments:
    {{ python }} -m scripts.analysis.ppo_reward_ab {{ arguments }}
