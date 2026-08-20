# Eco-AutoDrive

Eco-AutoDrive 研究如何在 MetaDrive 闭环中利用预训练 Diffusion Planner，并进一步探索通过 guidance、强化学习和道路预瞄信息改善能耗表现。

## 当前定位

仓库当前已经包含：

- 冻结官方 EMA Diffusion Planner 的 MetaDrive 闭环评测；
- 无交通和有交通 observation / execution 路径；
- DPM-10 baseline、DDIM-5 变体和 reference-centered orthogonal guidance；
- Exploration Policy、policy-guided rollout、GAE/PPO update 与 closed-loop smoke training；
- 严格的 evaluation / RL 产物契约和实验记录体系。

这些能力不等于最终研究结论。道路预瞄应如何表示、是否最终采用 PPO、最终能耗 reward/指标以及精细车辆能耗模型仍属于研究问题，见 [docs/research/README.md](docs/research/README.md)。当前轨迹按运动学方式直接执行，也不代表低层 steering/throttle/brake 的动力学可执行性。

## 环境准备

项目使用 Python 3.10 和 uv。官方权重放在 `checkpoints/DP-Origin/`；本地 MetaDrive 源码位于 `third_party/metadrive/`，由 `pyproject.toml` 以 editable dependency 使用。

本地 Windows 环境：

```powershell
uv sync --all-groups
uv run pytest -m "not gpu and not simulator and not slow"
uv run ruff check .
uv run ruff format --check .
```

编码智能体使用已准备好的 `.venv`，其读取顺序、实现约束和验证命令见 [AGENTS.md](AGENTS.md)。

## 闭环评测

最常用的 smoke 入口：

```powershell
# 无交通
.venv\Scripts\python.exe scripts\evaluate.py --config-name experiment/evaluate_no_traffic_smoke

# 有交通
.venv\Scripts\python.exe scripts\evaluate.py --config-name experiment/evaluate_traffic_smoke
```

`no-traffic` 会拒绝背景车辆和静态交通物体，不能用于有交通评测。

默认 sampler 为保持官方语义的 `dpm10`。项目额外提供：

```powershell
.venv\Scripts\python.exe scripts\evaluate.py --config-name experiment/evaluate_no_traffic_smoke planner/sampler=ddim5
.venv\Scripts\python.exe scripts\evaluate.py --config-name experiment/evaluate_no_traffic_smoke planner/sampler=ddim5_project_noise
```

`ddim5` 使用标准高斯初始噪声；`ddim5_project_noise` 使用 `0.5` 倍噪声，仅作为项目隔离对照。两者均采用本项目选择的五步时间表，不代表上游未公开的 sampler parity。

固定 reference guidance 必须与标准高斯 `ddim5` 搭配：

```powershell
.venv\Scripts\python.exe scripts\evaluate.py --config-name experiment/evaluate_no_traffic_smoke `
  planner/sampler=ddim5 planner/guidance=orthogonal_reference `
  guidance.lateral_scale=1 guidance.longitudinal_scale=0
```

scale 必须位于 `[-1,1]`；`(0,0)` 精确退化为同次 unguided reference。reference-centered energy、10 Hz 速度差分、单位梯度系数和 ego-only gradient scope 属于项目复现决定；guidance 不做中心线投影、平滑、裁剪或失败回退。完整数学与随机性契约见 [system-contract.md](docs/agents/system-contract.md) 和相关 ADR。

## 评测预设

`configs/experiment/` 提供完整、可运行的 smoke、full 和 matrix 评测预设；它们默认关闭视频。需要录制 GIF 时，在任一命令后追加 `video.enabled=true`。每次运行写入 `outputs/` 下的独立目录。

常用示例：

```powershell
# 无交通完整预设
.venv\Scripts\python.exe scripts\evaluate.py --config-name experiment/evaluate_no_traffic_full runtime.seed=3

# 无交通多 seed
.venv\Scripts\python.exe scripts\evaluate.py --multirun `
  --config-name experiment/evaluate_no_traffic_matrix

# 交通矩阵
.venv\Scripts\python.exe scripts\evaluate.py --multirun `
  --config-name experiment/evaluate_traffic_matrix
```

traffic matrix 默认使用两个 Joblib `loky` 进程；串行对照可通过 `hydra/launcher=basic evaluation.execution.mode=serial evaluation.execution.launcher=basic evaluation.execution.worker_count=1` 覆盖。CUDA 运行使用 `runtime.accelerator=cuda`，CPU 对照使用 `runtime.accelerator=cpu`。多 GPU 调度不属于该入口。

推理使用单设备 Lightning Fabric。默认 `runtime.accelerator=auto runtime.precision=auto`：CUDA 上优先 BF16 mixed precision，不支持 BF16 时使用 FP16；CPU 使用 FP32。严格复现 FP32 数值基线时传入 `runtime.precision=32-true`。

具体 horizon、warmup、产物字段与数组契约、终止类型以及其他运行不变量以 [system-contract.md](docs/agents/system-contract.md) 为准。

## PPO closed-loop smoke training

当前 smoke profile 主要用于验证 Exploration Policy → rollout → GAE/PPO → checkpoint 的闭环数据流：

```powershell
.venv\Scripts\python.exe scripts\train.py runtime.seed=0 training.replay_id=0
```

现有 `metadrive_builtin_v1` reward 只用于训练链路验证，不是 PlannerRFT parity，也不是最终能耗 reward。规模化训练、learned-policy evaluation 和最终 energy-oriented objective 仍需按研究问题和 GitHub Issues 推进。

## 结果与实验记录

运行产物保存在被 Git 忽略的 `outputs/`。用于研究结论的运行应在 `docs/experiments/records/` 中登记，并更新 [实验索引](docs/experiments/README.md)；记录实际 commit/diff、resolved config、Hydra overrides、seed、环境、结果和产物位置。未运行的计划不要写成实验结果。

## 文档导航

| 文件 | 职责 |
| --- | --- |
| [AGENTS.md](AGENTS.md) | 编码智能体的读取、实现、验证和写回规则 |
| [docs/agents/domain.md](docs/agents/domain.md) | 高风险领域语义区分与术语使用规则 |
| [CONTEXT.md](CONTEXT.md) | 领域术语的完整规范定义 |
| [docs/agents/system-contract.md](docs/agents/system-contract.md) | 当前已实现系统的数据与执行契约 |
| [docs/adr/](docs/adr/) | 已接受的重要设计选择及理由 |
| [GitHub Issues](https://github.com/xcz0/Eco-AutoDrive/issues) | active implementation work 与验收标准 |
| [docs/research/README.md](docs/research/README.md) | 未决假设、候选方法和开放问题 |
| [docs/experiments/README.md](docs/experiments/README.md) | 实际运行的配置、结果、结论边界和产物索引 |

## 主要参考

- [Diffusion Planner](https://github.com/ZhengYinan-AIR/Diffusion-Planner)：基础框架与初始权重
- [MetaDrive](https://github.com/metadriverse/metadrive)：闭环仿真环境
