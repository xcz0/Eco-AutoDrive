# Eco-AutoDrive

Eco-AutoDrive 研究如何在 MetaDrive 闭环中利用预训练 Diffusion Planner，并探索 guidance、强化学习和道路预瞄信息是否能够改善能耗表现。

这是个人科研代码库。仓库中的实现用于建立可验证的研究链路；**已有能力不等于最终研究方法或研究结论**。

## 当前能力与边界

当前仓库已经包含：

- 冻结官方 EMA Diffusion Planner 的 MetaDrive 闭环评测；
- 无交通和有交通 observation / execution 路径；
- DPM/DDIM sampler 与 reference-centered orthogonal guidance；
- Exploration Policy、policy-guided rollout、GAE/PPO update 与 closed-loop smoke training；
- evaluation / RL artifact 与实验记录体系。

道路预瞄如何表示和注入、是否最终采用 PPO、最终 energy-oriented objective 以及精细车辆能耗模型仍属于研究问题，见 [docs/research/README.md](docs/research/README.md)。当前轨迹采用运动学方式执行，不表示 steering/throttle/brake 层面的动力学可执行性。

## 环境准备

项目使用 Python 3.10、uv 和本地 editable MetaDrive。官方 Diffusion Planner 权重放在 `checkpoints/DP-Origin/`，MetaDrive 源码位于 `third_party/metadrive/`。

```powershell
just setup
just check
```

`just --list` 查看全部开发、评测、训练和分析入口。

编码智能体在预先准备好的 `.venv` 中工作，不应重复执行环境初始化；其读取、修改和验证规则见 [AGENTS.md](AGENTS.md)。

## 常用工作流

快速闭环评测：

```powershell
# 无交通 smoke
just eval-smoke

# 有交通 smoke
just eval-traffic-smoke

# 无交通 full evaluation
just eval

# 有交通 full evaluation
just eval-traffic
```

矩阵评测：

```powershell
just eval-no-traffic-matrix
just eval-matrix
```

固定 reference guidance smoke：

```powershell
just eval-guidance guidance.lateral_scale=1 guidance.longitudinal_scale=0
```

PPO closed-loop smoke training：

```powershell
just train-smoke 0 0
```

这些入口均由 `justfile` 调用 Hydra 配置，可在命令末尾追加 override。机器资源通过版本化 profile 选择，例如 `resources=rtx_a4000`；它只改变 worker、slot 和线程预算，不改变 PPO、reward、sampler 或 guidance。仓库本身不读取 `.env`；若在命令包装层使用它，其中只能保存本机的 profile 选择（例如 `ECO_RESOURCE_PROFILE=rtx_a4000`），不能保存未版本化的实验配置。sampler、precision、随机性、时间尺度、并行和 artifact 的精确语义以 [system-contract.md](docs/agents/system-contract.md) 和实际 resolved config 为准，不在 README 重复维护。

## 结果与实验记录

运行产物默认写入被 Git 忽略的 `outputs/`。用于研究结论的运行应在 `docs/experiments/records/` 中登记，并更新 [实验索引](docs/experiments/README.md)。记录实际代码状态、resolved config、Hydra overrides、随机种子、环境、结果和产物位置；未运行的计划不要写成实验结果。

## 文档导航

| 文件 | 职责 |
| --- | --- |
| [AGENTS.md](AGENTS.md) | 编码智能体的事实路由、实现原则、验证和写回规则 |
| [docs/agents/domain.md](docs/agents/domain.md) | 容易导致实现或实验解释错误的领域语义 gotchas |
| [CONTEXT.md](CONTEXT.md) | 稳定领域术语的规范定义 |
| [docs/agents/system-contract.md](docs/agents/system-contract.md) | 当前已实现系统的数据与执行契约 |
| [docs/adr/](docs/adr/) | 已接受的重要设计选择及理由 |
| [GitHub Issues](https://github.com/xcz0/Eco-AutoDrive/issues) | active implementation work 与持久化验收标准 |
| [docs/research/README.md](docs/research/README.md) | 未决假设、候选方法和开放问题 |
| [docs/experiments/README.md](docs/experiments/README.md) | 已运行实验的 provenance、结果和结论边界 |

## 主要参考

- [Diffusion Planner](https://github.com/ZhengYinan-AIR/Diffusion-Planner)：基础框架与初始权重
- [MetaDrive](https://github.com/metadriverse/metadrive)：闭环仿真环境
