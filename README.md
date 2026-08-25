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

项目使用 Python 3.10、uv 和本地 editable MetaDrive。官方 Diffusion Planner 权重放在 `checkpoints/DP-Origin/`，MetaDrive 源码位于 `third_party/metadrive/`。

```powershell
just setup
just check
```

`just --list` 查看全部开发、评测、训练和分析入口。

## 常用工作流

快速闭环评测：

```powershell
# 无交通 smoke
just evaluate no_traffic_smoke

# 有交通 smoke
just evaluate traffic_smoke

# 无交通 full evaluation
just evaluate no_traffic

# 有交通 full evaluation
just evaluate traffic
```

矩阵评测：

```powershell
just evaluate-matrix no_traffic
just evaluate-matrix traffic
```

固定 reference guidance smoke：

```powershell
just evaluate no_traffic_smoke planner/sampler=ddim5 `
    planner/guidance=orthogonal_reference `
    guidance.lateral_scale=1 guidance.longitudinal_scale=0
```

PPO closed-loop smoke training：

```powershell
just train ppo_smoke 0 0
```

可复用性能诊断与固定能耗矩阵：

```powershell
just benchmark throughput
just benchmark throughput_traffic
just benchmark rollout
just energy outputs/energy_matrix/manual-run
```

机器资源通过版本化 profile 选择，例如 `resources=rtx_a4000`；它只改变 worker、slot 和线程预算。仓库本身不读取 `.env`；若在命令包装层使用它，其中只能保存本机的 profile 选择（例如 `ECO_RESOURCE_PROFILE=rtx_a4000`）。sampler、precision、随机性、时间尺度、并行和 artifact 的精确语义以 [system-contract.md](docs/agents/system-contract.md) 和实际 resolved config 为准。

## 结果与实验记录

运行产物默认写入 `outputs/`。

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
