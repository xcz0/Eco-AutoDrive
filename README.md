# Eco-AutoDrive

Eco-AutoDrive 研究如何在 MetaDrive 闭环中利用预训练 Diffusion Planner，并探索 guidance、强化学习和道路预瞄信息是否能够改善能耗表现。

## 项目进度与边界

当前包含：

- 冻结官方 EMA Diffusion Planner 的 MetaDrive 闭环评测；
- 无交通和有交通 observation / execution 路径；
- DPM/DDIM sampler 与 reference-centered orthogonal guidance；
- Exploration Policy、policy-guided rollout、GAE/PPO update 与 closed-loop smoke training；
- evaluation / RL artifact 与实验记录体系。

道路预瞄如何表示和注入、是否最终采用 PPO、最终 energy-oriented objective 以及精细车辆能耗模型仍属于研究问题，见 [docs/research/README.md](docs/research/README.md)。当前轨迹采用运动学方式执行，不表示 steering/throttle/brake 层面的动力学可执行性。

## 环境准备

项目在 Windows PowerShell 下使用 Python 3.10、uv 和本地 editable MetaDrive。官方 Diffusion Planner 权重放在 `checkpoints/DP-Origin/`，MetaDrive 源码位于 `third_party/metadrive/`。

```powershell
just setup
just check
```

`just --list` 查看全部开发、评测、训练、benchmark 和实验入口。

## 常用工作流

快速闭环评测：

```powershell
# 无交通 smoke
just evaluation run --config-name jobs/evaluation/no_traffic_smoke

# 有交通 smoke
just evaluation run --config-name jobs/evaluation/traffic_smoke

# 无交通 full evaluation
just evaluation run

# 有交通 full evaluation
just evaluation run --config-name jobs/evaluation/traffic
```

矩阵评测：

```powershell
just evaluation run --config-name jobs/evaluation/no_traffic_matrix --multirun
just evaluation run --config-name jobs/evaluation/traffic_matrix --multirun
```

固定 reference guidance smoke：

```powershell
just evaluation run --config-name jobs/evaluation/no_traffic_smoke components/sampler=ddim5 `
    components/guidance=orthogonal_reference `
    guidance.lateral_scale=1 guidance.longitudinal_scale=0
```

PPO closed-loop smoke training：

```powershell
just training run runtime.seed=0 training.replay_id=0
```

PPO 默认记录到本地 MLflow（`outputs/mlflow/`）。在仓库根目录启动 UI：

```powershell
just mlflow ui --backend-store-uri sqlite:///outputs/mlflow/mlflow.db --host 127.0.0.1 --port 5000
```

打开 `http://127.0.0.1:5000`，选择 `eco-autodrive-ppo` experiment，通过 `config.runtime.seed`、`reward_profile` 及 comparison 入口写入的 `arm` / `protocol` 筛选 Run，并在 Compare 中对比 `ppo/*`、`reward/*`、`behavior/*` 和 `energy/*` 曲线。其他实验可使用 `+tracking.tags.study=...` 添加标识；`tracking.run_name=...` 指定显示名，`tracking.checkpoint_interval=5` 将 update checkpoint 上传间隔改为 5。

`tracking.enabled=false` 显式关闭跟踪。连接远程服务时同时设置 `tracking.tracking_uri=https://... tracking.artifact_location=null`，让服务拥有 artifact 存储。从 `training.resume_checkpoint_path=...` 恢复会继续原 Run，目标 `training.update_count` 仍是累计 update 数；同一 Run 的实验参数必须保持一致。详细恢复及指标口径见[训练跟踪契约](docs/agents/contracts/training.md#训练实验跟踪)。

可复用性能诊断与固定能耗矩阵：

```powershell
just benchmark run
just benchmark run --config-name jobs/benchmark/throughput_traffic
just benchmark run --config-name jobs/benchmark/rollout
just exp guidance sweep run --output-dir outputs/energy_matrix/manual-run
```

PlannerRFT reward sanity：

```powershell
just validation reward run --output-dir outputs/reward_sanity/manual-run
```

该命令只计算配置中声明的固定合成 reward case，不运行 PPO。

实验使用薄入口 `just exp ...`，对应 `python -m scripts.experiments`。工作流配置按
comparison、reward、credit、guidance、training 五类职责组织；`--config` 显式选择配置，
`just exp <工作流> <动作> --help` 查看参数。

| 工作流 | 动作 |
| --- | --- |
| compare | train、eval、analyze |
| reward | collect、run、analyze |
| credit | run、analyze |
| guidance authority / sweep | run、analyze |
| training | grid、diagnose、eval、analyze |

先采集一次，再从相同源 batch 独立执行 reward 和 credit；校准尺度与 energy-band 阈值
每次由该 batch 及配置推导，不需要前序诊断目录：

```powershell
just exp reward collect --output-dir outputs/fixed-batch
just exp reward run --source-dir outputs/fixed-batch --output-dir outputs/reward
just exp credit run --config configs/experiments/credit/objectives.yaml --source-dir outputs/fixed-batch --output-dir outputs/credit
```

`reward/collection.yaml` 指定 job 和 overrides；reward run 比较组件、权重和 energy
representation，不执行 actor backward。credit 命名配置为 `sensitivity.yaml`、`objectives.yaml`、
`ablation.yaml`、`energy-band.yaml`，显式指定 reward、advantage 和 credit 对照轴及本实验阈值。

comparison 默认 frozen/A1/A2 三臂，也可通过 `comparison/calibrated.yaml` 选择 calibrated
R0/stress 双臂。每次 train 指定 seed，eval 指定已训练 checkpoint（frozen arm 不带 checkpoint）：

```powershell
just exp compare eval --arm a0 --output-dir outputs/a0
just exp compare train --arm a1 --training-seed 0 --output-dir outputs/a1
just exp compare eval --arm a1 --checkpoint final --checkpoint-path outputs/a1/policy-final.pt --output-dir outputs/a1-evaluation
just exp compare train --config configs/experiments/comparison/calibrated.yaml --arm rstress --training-seed 0 --output-dir outputs/stress
just exp guidance authority run --output-dir outputs/authority
just exp guidance sweep run --output-dir outputs/energy-matrix
just exp training grid --output-dir outputs/optimizer-grid
```

training grid 使用显式 optimizer 笛卡尔积与预算，报告通过项或无候选；没有阶段晋升或回报排名。
`training diagnose --config <文件>` 测量已有训练结果；配置包含 `training_summaries` 路径列表、
`training_seeds` 列表、`mc_draws` 和 `mc_seed`。`training eval --config <文件>` 配置包含
`protocol`、`policy_action_seeds` 及 `records`；每条记录显式提供 `arm`、`training_summary`、
`checkpoint_label`、`checkpoint_path` 和可空的 `deterministic_evaluation_dir`。输入路径均相对
该 YAML 解析，复用评测必须匹配 checkpoint 与随机条件。

软件验证与后端比较继续使用 `just validation reward` 和 `just benchmark execution`。
旧实验命令、Python import、stability 搜索与独立 reproducibility 工作流已移除；历史记录保留
原命令、路径和结果，不提供迁移层。

机器资源通过版本化 profile 选择，例如 `components/resources=rtx_a4000`；它只改变 worker、slot 和线程预算。CLI 与 study bootstrap 会按需读取仓库根目录的可选 `.env`，并以 `MACHINE_NAME` 自动选择同名的 `configs/components/resources/<机器名>.yaml`。进程中已有的 `MACHINE_NAME` 优先于 `.env`，显式 Hydra `components/resources=...` override 又优先于两者；可用值见该目录，`.env.example` 给出格式。

未配置机器 profile 时，semantic job 仍可 compose 和 validate；真正需要资源预算的训练、评测或 benchmark 执行会明确失败，不会静默采用默认 worker 数。sampler、precision、随机性、时间尺度、并行和 artifact 的精确语义以 [system-contract.md](docs/agents/system-contract.md) 和实际 resolved config 为准。

## 结果与实验记录

运行产物默认写入 `outputs/`。

实验运行/汇总入口默认生成 Markdown 报告与 SVG/PNG 图，`--no-figures` 可关闭出图。
`reward collect` 只保存采集产物，不生成诊断或图表。
已有产物可通过统一入口重算描述统计并重绘，不重新运行环境、GAE 或训练：

```powershell
just exp reward analyze --source-dir outputs/reward --output-dir outputs/reward-report
just exp credit analyze --source-dir outputs/credit --output-dir outputs/credit-report
just exp training analyze --source-dir outputs/optimizer-grid --output-dir outputs/grid-report
just exp compare analyze --source-dir outputs/my-protocol --config outputs/my-protocol/comparison.yaml --output-dir outputs/protocol-report
```

源目录与离线输出目录必须独立，不能相同或互相嵌套。实验类型、输入文件和比较配置见
[离线分析与报告契约](docs/agents/contracts/experiments.md#实验离线分析与报告)。

## 文档导航

| 文件 | 职责 |
| --- | --- |
| [AGENTS.md](AGENTS.md) | 编码智能体的事实路由、执行边界、科研实现原则、验证和 Issue 工作流 |
| [docs/agents/domain.md](docs/agents/domain.md) | 容易导致实现或实验解释错误的领域语义 gotchas |
| [CONTEXT.md](CONTEXT.md) | 稳定领域术语的规范定义 |
| [docs/agents/system-contract.md](docs/agents/system-contract.md) | 当前已实现系统的数据与执行契约 |
| [docs/adr/](docs/adr/) | 已接受的重要设计选择及理由 |
| [GitHub Issues](https://github.com/xcz0/Eco-AutoDrive/issues) | 已接受、可执行且需跨会话跟踪的工作及其验收标准；不是当前实现事实 |
| [docs/research/README.md](docs/research/README.md) | 尚未接受或尚未确定的假设、候选方法和开放问题 |
| [docs/experiments/README.md](docs/experiments/README.md) | 已运行实验的 provenance、结果和结论边界 |

## 主要参考

- [Diffusion Planner](https://github.com/ZhengYinan-AIR/Diffusion-Planner)：基础框架与初始权重
- [MetaDrive](https://github.com/metadriverse/metadrive)：闭环仿真环境

