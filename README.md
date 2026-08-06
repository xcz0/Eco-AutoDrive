# Eco-AutoDrive

本项目研究如何在预训练 Diffusion Planner 中加入道路预瞄信息，并在 MetaDrive 闭环中分析和优化能耗表现。当前仍处于基础代码构建与方案分析阶段，强化学习方法、预瞄注入方式和最终能耗模型尚未确定。

当前可运行部分包括：官方 EMA 权重推理、MetaDrive 无交通与低密度交通观测适配，以及轨迹级运动学闭环。准确进度见 [STATUS.md](docs/STATUS.md)，未确定的研究方案见 [RESEARCH.md](docs/RESEARCH.md)。

## 当前成果

- 与官方 checkpoint 兼容的 Diffusion Planner、归一化和 10 步 DPM-Solver++ baseline sampler；
- MetaDrive 程序化地图、路线、邻车历史和静态物体到官方固定形状张量的适配；
- 2 Hz 滚动规划与 10 Hz 轨迹级运动学执行；
- 无交通 `S`/`SC`、噪声 seeds `0..4` 的 20 s Windows CPU 闭环验证；
- 有交通观测、2 s 历史预热、评测审计和可复现产物；
- PGMap 未设置限速哨兵和轨迹执行时序错误的回归修正。

长时高交通量实验中，背景仿真车辆在持续运行后出现异常行为，并进一步触发自车碰撞或出界。这些结果不用于判断规划器是否具备基本驾驶能力；中程、低密度场景可正常运行。后续评测优先采用较短或中等时长、较低背景车辆数量的场景，只需覆盖过弯、变道、限速变化等主要能耗影响因素。

## 当前研究边界

当前闭环使用运动学方式直接执行轨迹点，不生成 steering、throttle 或 brake，也不验证低层车辆动力学可执行性。该限制需要在解释能耗结果时明确说明，但现阶段不阻塞轨迹规划和相对能耗研究。

强化学习、奖励定义、预瞄长度与注入结构均属于待研究内容，不应从当前依赖或占位代码推断为已实现方案。

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

交通评测应优先使用能够稳定表达研究因素的短程或中程配置，并显式控制背景车辆数量、评测时长和场景特征。现有长路线配置仅保留为历史诊断，不作为默认能耗评测基线。

```powershell
uv run python scripts/evaluate.py --config-name evaluation/traffic `
  evaluation.evaluated_horizon_steps=100 env.horizon=120 video.enabled=false
```

## 文档导航

| 文件 | 内容 |
| --- | --- |
| [STATUS.md](docs/STATUS.md) | 当前完成度、已验证结论和下一步 |
| [LOGIC.md](docs/LOGIC.md) | 已实现系统的数据与执行契约 |
| [DECISIONS.md](docs/DECISIONS.md) | 已确定的重要方案选择 |
| [CORRECTIONS.md](docs/CORRECTIONS.md) | 历史错误、修正和证据 |
| [RESEARCH.md](docs/RESEARCH.md) | 尚未确定或尚未实现的研究设想与局限 |
| [experiments/README.md](experiments/README.md) | 实际评测配置、结果和产物索引 |
| [AGENTS.md](AGENTS.md) | 工作规则和验证要求 |
