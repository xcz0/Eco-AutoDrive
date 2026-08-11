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
.\scripts\run_evaluation.ps1 -Mode no-traffic -Profile smoke
```

该入口拒绝背景车辆和静态交通物体，不能用于有交通评测。

默认 sampler 是保持官方语义的 `dpm10`。阶段 1 的确定性 DDIM 与初始尺度隔离变体必须显式选择：

```powershell
.\scripts\run_evaluation.ps1 -Mode no-traffic -Profile smoke -Sampler ddim5
.\scripts\run_evaluation.ps1 -Mode no-traffic -Profile smoke -Sampler ddim5_project_noise
```

`ddim5` 使用标准高斯初始噪声；`ddim5_project_noise` 使用 `0.5` 倍噪声且只用于项目隔离对照。
两者都使用本项目明确选择的均匀连续时间五步表，不能据此声称作者未公开的 sampler parity。

阶段 2 的固定 reference guidance 必须显式搭配标准高斯 `ddim5`。以下 action 分别表示左移与相对
reference 加速；scale 必须在 `[-1,1]`，不会被裁剪：

```powershell
.\scripts\run_evaluation.ps1 -Mode no-traffic -Profile smoke -Sampler ddim5 `
  -Guidance orthogonal_reference -LateralScale 1 -LongitudinalScale 0
```

`(0,0)` 精确退化为同次 unguided reference。reference-centered energy、10 Hz 速度差分、单位
梯度系数和 ego-only gradient scope 是 ADR 0012 的项目复现决定，不能声称为 PlannerRFT 作者未公开
实现。guidance 不做中心线投影、平滑、裁剪或失败回退。

## 运行交通闭环

```powershell
.\scripts\run_evaluation.ps1 -Mode traffic -Profile smoke
```

## 场景评测预设

`scripts/run_evaluation.ps1` 只编排已有 Hydra 配置，默认关闭视频以缩短本机验证时间；使用
`-Video` 才生成 GIF。所有运行仍由 Hydra 在 `outputs/` 下创建独立产物目录。

```powershell
# 读取将执行的命令，不启动仿真。
.\scripts\run_evaluation.ps1 -Mode no-traffic -Profile smoke -DryRun

# 默认 S/SC 无交通场景，各最多 20 s；指定全局/噪声 seed。
.\scripts\run_evaluation.ps1 -Mode no-traffic -Profile full -RuntimeSeed 3

# 无交通多噪声 seed 检查。
.\scripts\run_evaluation.ps1 -Mode no-traffic -Profile matrix -RuntimeSeeds 0,1,2,3,4

# 两条长路线的交通检查：2 s 预热，正式评测各 10 s。
.\scripts\run_evaluation.ps1 -Mode traffic -Profile smoke

# 交通密度和 paired seed 的矩阵；该运行不是稳定能耗基线。
.\scripts\run_evaluation.ps1 -Mode traffic -Profile matrix `
  -RuntimeSeeds 0,1,2 -TrafficDensities 0.05,0.10
```

脚本会校验 checkpoint、MetaDrive 源码与 `uv` 是否存在。`no-traffic` 的 `env.horizon` 必须等于正式评测步数；`traffic` 必须额外包含固定的 20 步 history warmup，脚本预设已满足这些约束。

推理运行时使用单设备 Lightning Fabric。默认 `-Accelerator auto -Precision auto`：有 CUDA 时使用 BF16 mixed precision（硬件不支持 BF16 时使用 FP16），否则使用 CPU FP32。严格复现官方 FP32 数值基线时显式传入 `-Precision 32-true`。无交通 smoke 可用 `-ScenarioSeed` 单独指定地图 seed；交通预设则显式配对地图 seed 与 `RuntimeSeed`。

## 文档导航

| 文件 | 内容 |
| --- | --- |
| [CONTEXT.md](CONTEXT.md) | 项目领域语言和规范术语 |
| [system-contract.md](docs/agents/system-contract.md) | 当前已实现系统的数据与执行契约 |
| [docs/adr/](docs/adr/) | 重要设计和实验方法决定及其理由 |
| [GitHub Issues](https://github.com/xcz0/Eco-AutoDrive/issues) | active work |
| [issue-tracker.md](docs/agents/issue-tracker.md) | GitHub Issue 工作流 |
| [research/README.md](docs/research/README.md) | 未决假设、候选方法和开放问题 |
| [docs/experiments/README.md](docs/experiments/README.md) | 实际评测配置、结果和产物索引 |
| [AGENTS.md](AGENTS.md) | 工作规则和验证要求 |


## 主要参考资源

- [Diffusion Planner](https://github.com/ZhengYinan-AIR/Diffusion-Planner)：提供扩散模型的基础框架与初始权重
- [metadrive](https://github.com/metadriverse/metadrive)：生成特定场景，远距离仿真环境
-
