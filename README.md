# Eco-AutoDrive

Eco-AutoDrive 研究如何通过强化学习微调预训练 Diffusion Planner 优化能耗表现，在 MetaDrive 闭环中训练和分析。

## 环境准备

项目使用 Python 3.10 和 uv。官方权重放在 `checkpoints/DP-Origin/`；本地 MetaDrive 源码位于 `third_party/metadrive/`，由 `pyproject.toml` 以 editable dependency 使用。

本地 Windows 环境：

```powershell
uv sync --all-groups
uv run pytest -m "not gpu and not simulator and not slow"
uv run ruff check .
uv run ruff format --check .
```

编码智能体的沙箱执行约定见 [AGENTS.md](AGENTS.md)。

## 闭环评测

无交通：

```powershell
.\scripts\run_evaluation.ps1 -Mode no-traffic -Profile smoke
```

有交通：

```powershell
.\scripts\run_evaluation.ps1 -Mode traffic -Profile smoke
```

`no-traffic` 会拒绝背景车辆和静态交通物体，不能用于有交通评测。

默认 sampler 为保持官方语义的 `dpm10`。项目额外提供：

```powershell
.\scripts\run_evaluation.ps1 -Mode no-traffic -Profile smoke -Sampler ddim5
.\scripts\run_evaluation.ps1 -Mode no-traffic -Profile smoke -Sampler ddim5_project_noise
```

`ddim5` 使用标准高斯初始噪声；`ddim5_project_noise` 使用 `0.5` 倍噪声，仅作为项目隔离对照。两者均采用本项目选择的五步时间表，不代表上游未公开的 sampler parity。

固定 reference guidance 必须与标准高斯 `ddim5` 搭配：

```powershell
.\scripts\run_evaluation.ps1 -Mode no-traffic -Profile smoke -Sampler ddim5 `
  -Guidance orthogonal_reference -LateralScale 1 -LongitudinalScale 0
```

scale 必须位于 `[-1,1]`；`(0,0)` 精确退化为同次 unguided reference。reference-centered energy、10 Hz 速度差分、单位梯度系数和 ego-only gradient scope 属于项目复现决定，详见 ADR 0013；guidance 不做中心线投影、平滑、裁剪或失败回退。

## 评测预设

`scripts/run_evaluation.ps1` 编排已有 Hydra 配置，默认关闭视频；使用 `-Video` 生成 GIF。所有运行结果写入 `outputs/` 下的独立目录。

常用示例：

```powershell
# 只查看将执行的命令
.\scripts\run_evaluation.ps1 -Mode no-traffic -Profile smoke -DryRun

# 无交通完整预设
.\scripts\run_evaluation.ps1 -Mode no-traffic -Profile full -RuntimeSeed 3

# 无交通多 seed
.\scripts\run_evaluation.ps1 -Mode no-traffic -Profile matrix -RuntimeSeeds 0,1,2,3,4

# 交通矩阵
.\scripts\run_evaluation.ps1 -Mode traffic -Profile matrix `
  -RuntimeSeeds 0,1,2 -TrafficDensities 0.05,0.10
```

traffic matrix 默认使用两个 Joblib `loky` 进程；可用 `-ExecutionMode serial` 进行串行对照。CUDA 运行需显式传入 `-Extra cuda -Accelerator cuda`，CPU 对照使用 `-Extra cpu -Accelerator cpu`。多 GPU 分配不属于该入口。

推理使用单设备 Lightning Fabric。默认 `-Accelerator auto -Precision auto`：CUDA 上优先 BF16 mixed precision，不支持 BF16 时使用 FP16；CPU 使用 FP32。严格复现 FP32 数值基线时传入 `-Precision 32-true`。

具体 horizon、warmup、产物字段与数组契约、终止类型、实验元数据和其他执行契约以 [system-contract.md](docs/agents/system-contract.md) 与 [docs/experiments/README.md](docs/experiments/README.md) 为准。

## PPO smoke training

小规模训练 profile 使用 CUDA BF16、无交通 `S`/`SC` 和 `2 x 16 x 4` 参数：

```powershell
.venv\Scripts\python.exe scripts\train.py runtime.seed=0 training.replay_id=0
```

`metadrive_builtin_v1` 只验证闭环 PPO 数据流，不是 PlannerRFT parity 或能耗 reward。RL 训练产物与 evaluation 产物使用各自的数据边界和写出路径。训练状态 checkpoint 可用于从已完成 update 恢复；规模化和 learned-policy evaluation 不属于该入口。

### 评测代码接口

`scripts/evaluate.py` 是 Hydra 配置入口，先调用
`eco_planner.evaluation.parse_evaluation_config`，得到不可变的、字段冻结的
`EvaluationJobConfig`，再传给 `run_evaluation(config, output_dir)`。核心评测代码不接收
`DictConfig`，配置字段缺失、类型错误或互相矛盾时由 Pydantic 在入口一次性拒绝。

环境每次执行轨迹后返回的动态 `info`，必须立即通过
`TrajectoryExecutionRecord.from_info(info)` 转成已校验的执行记录；evaluation 层不直接读取
MetaDrive 字典字段。

评测 JSON 产物使用严格 Pydantic 模型；NPZ trace 使用 `validate_trace_arrays` 定义的显式数组契约。JSON 加载时检查当前模型要求的字段和类型，NPZ 加载时检查完整字段集合、数组类型、shape、dtype、有限性以及已实现的跨字段/跨数组不变量。产物契约本身不使用 `schema_version` 进行格式选择。

## 文档导航

| 文件                                                            | 内容              |
| ------------------------------------------------------------- | --------------- |
| [CONTEXT.md](CONTEXT.md)                                      | 领域语言和规范术语       |
| [system-contract.md](docs/agents/system-contract.md)          | 当前已实现系统的数据与执行契约 |
| [docs/adr/](docs/adr/)                                        | 重要设计与实验方法决定     |
| [GitHub Issues](https://github.com/xcz0/Eco-AutoDrive/issues) | active work     |
| [research/README.md](docs/research/README.md)                 | 未决假设、候选方法和开放问题  |
| [docs/experiments/README.md](docs/experiments/README.md)      | 实际评测配置、结果和产物索引  |
| [AGENTS.md](AGENTS.md)                                        | 编码智能体工作规则与验证要求  |

## 主要参考

* [Diffusion Planner](https://github.com/ZhengYinan-AIR/Diffusion-Planner)：基础框架与初始权重
* [MetaDrive](https://github.com/metadriverse/metadrive)：闭环仿真环境
