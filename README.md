# Eco-AutoDrive

本项目研究如何在预训练 Diffusion Planner 中加入道路预瞄信息，并在 MetaDrive 闭环中分析和优化能耗表现。领域术语、当前行为、设计理由、active work、未决研究和实验事实分别由下方文档维护。

## 环境准备

项目固定使用 Python 3.10 和 uv。官方权重放在 `checkpoints/DP-Origin/`；本地 MetaDrive 源码位于 `third_party/metadrive/`，由 `pyproject.toml` 以 editable dependency 使用。

Windows 快速验证：

```powershell
uv sync --all-groups
uv run pytest -m "not gpu and not simulator and not slow"
uv run ruff check .
uv run ruff format --check .
```

## 运行无交通闭环

```powershell
uv run python scripts/evaluate.py model.device=cpu env.horizon=5 `
  video.enabled=false 'scenarios=[{name:straight,map:S,seed:0}]'
```

该入口拒绝背景车辆和静态交通物体，不能用于有交通评测。

## 运行交通闭环

```powershell
uv run python scripts/evaluate.py --config-name evaluation/traffic `
  evaluation.evaluated_horizon_steps=100 env.horizon=120 video.enabled=false
```

## 文档导航

| 文件 | 内容 |
| --- | --- |
| [CONTEXT.md](CONTEXT.md) | 项目领域语言和规范术语 |
| [system-contract.md](docs/agents/system-contract.md) | 当前已实现系统的数据与执行契约 |
| [docs/adr/](docs/adr/) | 重要设计和实验方法决定及其理由 |
| [GitHub Issues](https://github.com/xcz0/Eco-AutoDrive/issues) | active work |
| [issue-tracker.md](docs/agents/issue-tracker.md) | GitHub Issue 工作流 |
| [research/README.md](docs/research/README.md) | 未决假设、候选方法和开放问题 |
| [experiments/README.md](experiments/README.md) | 实际评测配置、结果和产物索引 |
| [AGENTS.md](AGENTS.md) | 工作规则和验证要求 |
