# PlannerRFT PPO-only 复现研究

## 文档状态

本文描述一个**论文级候选复现**，用于指导后续拆分 GitHub Issues；它不是当前系统契约、已接受
设计、已实现功能或实验结果。当前系统仍是冻结官方 EMA、10-step DPM-Solver++、每个规划周期
执行前 0.5 s 轨迹的 MetaDrive 运动学闭环，详见
[`system-contract.md`](../../agents/system-contract.md)。

## 目标边界

本专题只复现 PlannerRFT 的 Exploration Optimization：

1. 保留可独立运行的官方 Diffusion Planner baseline；
2. 增加可切换的 5-step DDIM；
3. 用冻结的 reference planner 产生参考轨迹，并施加横向/纵向 guidance；
4. 用 Exploration Policy 输出两个 guidance scale 的概率分布和状态价值；
5. 在 MetaDrive 闭环中收集 rollout，以 GAE 和 PPO 只更新 Exploration Policy 与 value head；
6. 通过无 guidance、随机/固定 guidance 和 learned guidance 的配对消融判断 PPO 是否有用。

不包含 GRPO，不直接用 PPO 更新 DiT，也不声称训练后可以移除 Exploration Policy。PPO-only
阶段若有效，其含义是“策略学会如何引导冻结 planner”，不是“Diffusion Planner 已被强化学习
微调”。

## 候选数据流

```text
raw observation
      |
      +--> frozen scene encoder -----------------------+
      |                                                |
      +--> frozen reference planner --> x_ref ---------+--> Exploration Policy
      |                                                       |
      |                                               Beta distributions + value
      |                                                       |
      +--> frozen planner DiT <--- DDIM + orthogonal guidance <+
                              |
                       ego trajectory [80, 4]
                              |
                 MetaDrive kinematic execution
                              |
                    reward, termination, next state
                              |
                         GAE + PPO update
```

训练期间拟冻结 scene encoder、reference planner 和 planner DiT；只训练 Exploration Policy 的
actor/value 参数。若未来加入 GRPO，必须另立专题和设计决定。

## 当前仓库适配结论

| 区域 | 当前事实 | 对复现的含义 |
| --- | --- | --- |
| 模型 | 已移植 checkpoint-compatible encoder、DiT 和 10-step DPM-Solver++ | 新 sampler 必须是显式可选边界，不能覆盖 baseline 语义 |
| 编码 | `DiffusionPlanner.encode()` 只返回 scene token `encoding` | policy 输入的 mask、reference token 和 route 条件需要显式定义，不能假定现有 API 已提供 |
| 扩散 | VP-SDE、`x_start` prediction、baseline 初始噪声 `0.5 * N(0,I)` 已形成契约；论文描述 DDIM 从标准高斯开始 | DDIM 必须复用 normalization 和当前状态约束；初始噪声尺度/timestep 先经 parity gate，不能静默混同 |
| guidance | 已有可选的固定 reference-centered orthogonal guidance；默认仍关闭 | 阶段 2 的 neutral、离散几何与注入系数是 ADR 0013 的项目复现决定，不是作者实现 parity；Exploration Policy 仍未实现 |
| 环境动作 | `TrajectoryMetaDriveEnv.step()` 固定执行 5 个 0.1 s 点 | 与论文“执行第一个动作后重规划”的口径存在待决差异 |
| reward | 环境返回 MetaDrive reward 及 5 个 substep reward；当前系统不定义最终优化奖励 | 不能把现有 `total_reward` 或 `configs/reward/energy.yaml` 直接称为 PlannerRFT reward |
| RL | `src/eco_planner/rl/` 只有包声明，没有 rollout/GAE/PPO 实现 | `pyproject.toml` 和旧配置中的 DPPO 命名是遗留意图，不是可用能力 |
| 运行时 | 每个 evaluation job 是单进程、单设备 Fabric；traffic matrix 可做隔离作业并行 | 论文规模的并行环境仍需要独立 training orchestrator 和相应 ADR |

## 文档导航

- [分阶段实现与验收](implementation-plan.md)：代码边界、测试、产物和进入下一阶段的门槛。
- [契约与待决问题](design-gates.md)：必须先回答的 MDP、概率、奖励、运行时和实验口径问题。
- [一手资料核查](../plannerrft-ppo-primary-sources.md)：论文/补充材料/官方代码支持的事实与未公开细节。

## 研究门槛

必须按顺序通过以下门槛，不能用后续训练曲线代替前序正确性：

```text
baseline 可复现
  -> DDIM 数学与 sampler 对照通过
  -> 固定 guidance 的方向和量纲通过
  -> policy distribution/log-prob/value 单元测试通过
  -> rollout 时间轴、done 与 reward 对齐通过
  -> GAE/PPO 数值测试通过
  -> 小规模 closed-loop 学习信号成立
  -> 配对消融优于随机/固定 guidance
  -> 才讨论论文规模训练
```

缺少必需 checkpoint、配置、shape、单位、seed 或奖励定义时应立即失败。不得通过轨迹平滑、裁剪、
中心线投影、回退控制器、坏回合过滤或零轨迹掩盖失败。

## 与现有 ADR 的关系

- ADR 0001 要求保留官方 baseline。DDIM、reference planner 和 PPO 必须作为显式新路径加入；
  baseline checkpoint、normalization、模型层级和 DPM 入口保持可独立验证。
- ADR 0002 要求本仓库拥有 MetaDrive 边界。不得从 `ref/` 导入 PlannerRFT 或上游
  Diffusion Planner 运行时代码。
- ADR 0003 当前接受 0.5 s 运动学 receding-horizon execution。若为论文口径改成 0.1 s，必须
  先新增或修订 ADR，并同步 system contract。
- ADR 0006 要求保存原始 planner 失败。guidance 不是修复失败的回退器；所有候选轨迹、被选
  动作和失败回合必须可追溯。
- ADR 0007/0008 要求在稳定场景上比较并按终止类型分层。PPO 消融必须遵循配对 seed 和完整
  回合保留规则。

论文训练使用 nuMax、nuPlan/PDM-Closed LQR tracker、运动学自行车模型和 log-replay traffic；
本项目使用 MetaDrive 运动学轨迹点执行与不同场景分布。论文分数、40M environment steps 和
8xH100 资源条件只能作为来源事实，不能成为本项目默认验收阈值。

## 转为 active work

当且仅当某阶段的 design gate 已关闭，才为该阶段建立 GitHub Issue。Issue 至少应包含：目标、
非目标、依赖的决定、受影响文件、显式配置字段、测试矩阵、实验产物和退出条件。实现结果写入
`docs/experiments/`，成为当前不变量的内容写入 `system-contract.md`，需要长期保留理由的
选择写入 `docs/adr/`。
