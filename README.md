# Eco-AutoDrive

Eco-AutoDrive 研究如何在 MetaDrive 闭环中利用预训练 Diffusion Planner，并探索 guidance、强化学习和是否能够改善能耗表现。

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

测试以回归、集成和功能验证为主，目录职责与运行入口见 [tests/README.md](tests/README.md)。

## 开发与验证

已准备好的智能体环境直接使用 `.venv` 和 `justfile`，不把 `just setup` 或环境探测作为常规步骤。
按改动风险选择必要验证，入口包括 `just test-target <path-or-node>`、`just test`、`just test-sim`、`just test-gpu`、`just lint`、`just format` 和 `just typecheck`；
命令定义以 `justfile` 为准，测试需按沙箱要求申请沙箱外执行。文档修改不运行代码测试。

公共接口提供类型标注，仅为非显然逻辑写简短 docstring；遵循 Ruff 100 字符行宽及 `E/F/I/UP`。
路径使用 `pathlib.Path`，不硬编码平台绝对路径。依赖变更同时更新 `pyproject.toml` 与 `uv.lock`。
提交主题简短、使用祈使式，一次提交只完成一个逻辑变更。

GitHub 操作优先使用环境提供的集成。仅在无可用集成而回退到 Windows `gh` CLI 时，认证相关请求按沙箱要求在沙箱外执行；沙箱内 HTTP 401 可能是 credential manager 不可见。
仅在沙箱外 `gh auth status --hostname github.com` 也失败时才要求重新认证，不输出 token，不使用 `--show-token`。GitHub 写入授权与 Issue 更新边界见 [AGENTS](AGENTS.md)。

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

`tracking.enabled=false` 显式关闭跟踪。连接远程服务时同时设置 `tracking.tracking_uri=https://... tracking.artifact_location=null`，让服务拥有 artifact 存储。从 `training.resume_checkpoint_path=...` 恢复会继续原 Run，目标 `training.update_count` 仍是累计 update 数；同一 Run 的实验参数必须保持一致。续写规则见 [tracking 身份与持久化](docs/contracts/artifacts.md#tracking-身份与续写)，精确恢复保证见 [training contract](docs/contracts/training.md#精确续训范围)，指标口径见 [training protocol](docs/research/protocols/training.md)。

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

实验使用薄入口 `just exp ...`，对应 `python -m scripts.experiments`。工作流配置按 comparison、reward、credit、guidance、training 五类职责组织；`--config` 显式选择配置，`just exp <工作流> <动作> --help` 查看参数。

| 工作流 | 动作 |
| --- | --- |
| compare | train、eval、analyze |
| reward | collect、run、analyze |
| credit | run、analyze |
| guidance authority / sweep | run、analyze |
| training | grid、diagnose、eval、analyze |

研究设计见 [planning/evaluation](docs/research/protocols/planning-and-evaluation.md)、[training](docs/research/protocols/training.md) 与 [diagnostic studies](docs/research/protocols/diagnostic-studies.md)；输入与产物保证见 [artifacts contract](docs/contracts/artifacts.md)。常用入口示例：

```powershell
just exp reward collect --output-dir outputs/fixed-batch
just exp reward run --source-dir outputs/fixed-batch --output-dir outputs/reward
just exp credit run --config configs/experiments/credit/objectives.yaml --source-dir outputs/fixed-batch --output-dir outputs/credit
```

```powershell
just exp compare eval --arm a0 --output-dir outputs/a0
just exp compare train --arm a1 --training-seed 0 --output-dir outputs/a1
just exp compare eval --arm a1 --checkpoint final --checkpoint-path outputs/a1/policy-final.pt --output-dir outputs/a1-evaluation
just exp compare train --config configs/experiments/comparison/calibrated.yaml --arm rstress --training-seed 0 --output-dir outputs/stress
just exp guidance authority run --output-dir outputs/authority
just exp guidance sweep run --output-dir outputs/energy-matrix
just exp training grid --output-dir outputs/optimizer-grid
```

软件验证与后端比较继续使用 `just validation reward` 和 `just benchmark execution`。

机器资源通过版本化 profile 选择，例如 `components/resources=rtx_a4000`；它只改变 worker、slot 和线程预算。CLI 与 study bootstrap 会按需读取仓库根目录的可选 `.env`，并以 `MACHINE_NAME` 自动选择同名的 `configs/components/resources/<机器名>.yaml`。进程中已有的 `MACHINE_NAME` 优先于 `.env`，显式 Hydra `components/resources=...` override 又优先于两者；可用值见该目录，`.env.example` 给出格式。

未配置机器 profile 时，semantic job 仍可 compose 和 validate；真正需要资源预算的训练、评测或 benchmark 执行会明确失败，不会静默采用默认 worker 数。配置、资源和执行边界见 [execution contract](docs/contracts/execution.md)；采样与时间尺度见 [planning/evaluation protocol](docs/research/protocols/planning-and-evaluation.md)。实际运行参数从该次 resolved config 查询。

## 结果与实验记录

运行产物默认写入 `outputs/`。

实验运行/汇总入口默认生成 Markdown 报告与 SVG/PNG 图，`--no-figures` 可关闭出图。`reward collect` 只保存采集产物，不生成诊断或图表。已有产物可通过统一入口重算描述统计并重绘，不重新运行环境、GAE 或训练：

```powershell
just exp reward analyze --source-dir outputs/reward --output-dir outputs/reward-report
just exp credit analyze --source-dir outputs/credit --output-dir outputs/credit-report
just exp training analyze --source-dir outputs/optimizer-grid --output-dir outputs/grid-report
just exp compare analyze --source-dir outputs/my-protocol --config outputs/my-protocol/comparison.yaml --output-dir outputs/protocol-report
```

源目录与离线输出目录必须独立，不能相同或互相嵌套。输入与只读分析边界见 [artifacts contract](docs/contracts/artifacts.md)；比较设计见 [planning/evaluation protocol](docs/research/protocols/planning-and-evaluation.md#matched-comparison-与随机条件)。

## 文档导航

| 文件 | 职责 |
| --- | --- |
| [AGENTS.md](AGENTS.md) | 按知识类型读取、冲突处理、执行边界、长期原则与写回规则 |
| [Semantics](docs/research/semantics.md) | 概念与解释边界 |
| [Planning/evaluation](docs/research/protocols/planning-and-evaluation.md)、[Training](docs/research/protocols/training.md)、[Diagnostic studies](docs/research/protocols/diagnostic-studies.md) | 研究方法与比较设计 |
| [Data/model](docs/contracts/data-and-model.md)、[Execution](docs/contracts/execution.md)、[Training](docs/contracts/training.md)、[Artifacts](docs/contracts/artifacts.md) | 实现研究方法必须保证的软件语义 |
| [docs/adr/](docs/adr/) | 长期设计理由与历史决定，现行要求引用 Protocol/Contract |
| [GitHub Issues](https://github.com/xcz0/Eco-AutoDrive/issues) | 已接受、可执行且需跨会话跟踪的工作及其验收标准；不是当前实现事实 |
| [docs/research/README.md](docs/research/README.md) | 研究问题与文档导航，包括 Hypothesis 和非规范性参考资料 |
| [Findings](docs/research/findings.md) | 当前结论、适用条件与 supporting evidence |
| [docs/experiments/README.md](docs/experiments/README.md) | 真实运行的登记、provenance、检索与 record 模板 |
| [.agents/skills/](.agents/skills/) | 智能体重复工作流程 |

## 主要参考

- [Diffusion Planner](https://github.com/ZhengYinan-AIR/Diffusion-Planner)：基础框架与初始权重
- [MetaDrive](https://github.com/metadriverse/metadrive)：闭环仿真环境
