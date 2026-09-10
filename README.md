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

打开 `http://127.0.0.1:5000`，选择 `eco-autodrive-ppo` experiment，通过 `config.runtime.seed`、`reward_profile` 及 scalar-reward 入口写入的 `arm` / `protocol` 筛选 Run，并在 Compare 中对比 `ppo/*`、`reward/*`、`behavior/*` 和 `energy/*` 曲线。其他实验可使用 `+tracking.tags.study=...` 添加标识；`tracking.run_name=...` 指定显示名，`tracking.checkpoint_interval=5` 将 update checkpoint 上传间隔改为 5。

`tracking.enabled=false` 显式关闭跟踪。连接远程服务时同时设置 `tracking.tracking_uri=https://... tracking.artifact_location=null`，让服务拥有 artifact 存储。从 `training.resume_checkpoint_path=...` 恢复会继续原 Run，目标 `training.update_count` 仍是累计 update 数；同一 Run 的实验参数必须保持一致。详细恢复及指标口径见[训练跟踪契约](docs/agents/system-contract.md#训练实验跟踪)。

可复用性能诊断与固定能耗矩阵：

```powershell
just benchmark run
just benchmark run --config-name jobs/benchmark/throughput_traffic
just benchmark run --config-name jobs/benchmark/rollout
just experiment energy-sweep run --output-dir outputs/energy_matrix/manual-run
```

PlannerRFT reward sanity：

```powershell
just experiment reward-sanity run --output-dir outputs/reward_sanity/manual-run
```

该命令只计算配置中声明的固定合成 reward case，不运行 PPO。

实验统一使用 `just experiment <experiment> <action>`，对应
`python -m scripts.experiments`。各实验的配置位于 `configs/experiments/<experiment>/`，
可用 `--config` 指定；`just experiment <experiment> <action> --help` 查看参数。

固定批次采集与诊断分开运行。先采集一次，再显式复用同一批次：

```powershell
just experiment fixed-batch collect --output-dir outputs/fixed-batch
just experiment lambda-identifiability run --source-dir outputs/fixed-batch --output-dir outputs/lambda
just experiment reward-calibration run --source-dir outputs/fixed-batch --reference-dir outputs/lambda --output-dir outputs/calibration
just experiment objective-decomposition run --source-dir outputs/fixed-batch --output-dir outputs/decomposition
just experiment critic-gae-ablation run --source-dir outputs/fixed-batch --reference-dir outputs/decomposition --output-dir outputs/ablation
```

采集配置拥有 protocol、training seed 和 overrides；诊断配置拥有诊断轴、校准目标和阈值。
分解/消融配置中的 expected calibration 是显式的来源校验值，应与所研究批次对应。
参考目录必须来自同一批次、相同初始策略与样本顺序。

其他实验动作：`scalar-reward` 支持 `evaluate-a0`、`train`、`evaluate-policy`；
`ppo-stability` 支持 `stage-a`、`stage-b`、`stage-c`、`diagnose`、`summarize`；
`ppo-reproducibility report` 使用 `--source-dir` 与 `--output-dir`；
`execution-backend report` 使用 `--serial-dir`、`--job-level-dir`、`--vector-dir`、
对应的 `--*-wall-s` 和 `--output-dir`。原独立实验入口已移除，历史记录保留原命令，
当前实现不提供历史产物兼容或迁移。

机器资源通过版本化 profile 选择，例如 `components/resources=rtx_a4000`；它只改变 worker、slot 和线程预算。CLI 与 study bootstrap 会按需读取仓库根目录的可选 `.env`，并以 `MACHINE_NAME` 自动选择同名的 `configs/components/resources/<机器名>.yaml`。进程中已有的 `MACHINE_NAME` 优先于 `.env`，显式 Hydra `components/resources=...` override 又优先于两者；可用值见该目录，`.env.example` 给出格式。

未配置机器 profile 时，semantic job 仍可 compose 和 validate；真正需要资源预算的训练、评测或 benchmark 执行会明确失败，不会静默采用默认 worker 数。sampler、precision、随机性、时间尺度、并行和 artifact 的精确语义以 [system-contract.md](docs/agents/system-contract.md) 和实际 resolved config 为准。

## 结果与实验记录

运行产物默认写入 `outputs/`。

实验运行/汇总入口默认生成 Markdown 报告与 SVG/PNG 图，`--no-figures` 可关闭出图。
`fixed-batch collect` 只保存采集产物，不生成诊断或图表。
已有产物可通过统一入口重算描述统计并重绘，不重新运行环境、GAE 或训练：

```powershell
just experiment lambda-identifiability analyze --source-dir outputs/lambda --output-dir outputs/lambda-report
just experiment reward-calibration analyze --source-dir outputs/calibration --output-dir outputs/calibration-report
just experiment ppo-stability analyze --source-dir outputs/my-study --output-dir outputs/study-report
just experiment scalar-reward analyze --source-dir outputs/my-protocol --config outputs/my-protocol/comparison.yaml --output-dir outputs/protocol-report
```

源目录与离线输出目录必须独立，不能相同或互相嵌套。实验类型、输入文件和比较配置见
[离线分析与报告契约](docs/agents/system-contract.md#实验离线分析与报告)。

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
