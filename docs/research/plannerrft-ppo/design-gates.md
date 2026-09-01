# PlannerRFT PPO-only Design Gates

## 目的

本文件只维护 **PPO-only 工具本身**仍可能影响研究结论或后续架构选择的开放问题。

总体 reward / information 研究问题已经迁移到：

- [`reward-objective.md`](../reward-objective.md)
- [`information-representation.md`](../information-representation.md)
- [`ablation-plan.md`](../ablation-plan.md)

当前实现行为以 [`system-contract.md`](../../agents/system-contract.md) 为准；可执行工作由 GitHub Issues 跟踪。

## 当前开放 Gates

### G-P1：稳定 PPO 配置能否迁移到新的 objective 与 observation？

**状态：开放。**

E-028 已经在 `metadrive_builtin_v1`、no-traffic S/SC、当前 observation 与训练规模下找到多 seed 长程稳定候选。

尚未证明该稳定性可以直接外推到：

- `plannerrft_energy_v1` 或其他 scalar energy objective；
- richer road / traffic information；
- trainable encoder；
- 更长训练；
- traffic scenarios。

因此后续 reward / information 实验应把“稳定配置迁移”视为 mechanical gate，而不是默认成立。

关闭条件不是重新搜索全参数空间，而是在目标实验条件下证明训练无明显数值或行为崩溃，并通过 matched evaluation 的基本 retention 检查。

---

### G-P2：长程 credit assignment 是否真正提供额外价值？

**状态：开放。**

当前 PPO 使用 closed-loop return 与 GAE，但“使用 PPO”本身不能证明 long-horizon feedback 是有效因素。

需要通过 controlled ablation 判断：

- short / local feedback；
- longer discounted return / GAE；

是否在相同 reward、observation 与训练条件下产生可重复差异。

若长 horizon 没有独立收益，项目应避免仅凭算法形式声称“长程反馈带来能耗优化”。

该 gate 属于 [`reward-objective.md`](../reward-objective.md) 的核心实验问题，但会影响 PPO 的配置与解释，因此在此保留。

---

### G-P3：guidance action 是否具有可解释的闭环局部增益？

**状态：开放。**

Exploration Policy 输出 lateral / longitudinal guidance。需要确保 policy distribution 的变化与闭环行为变化之间存在可解释关系。

主要观察：

```text
policy distribution change
        ↓
guidance action change
        ↓
trajectory / speed behavior change
        ↓
energy / safety / progress change
```

如果很小的 action distribution 漂移会导致大幅闭环失败，则 reward / information 实验可能被 action sensitivity 混淆。

该问题只需作为诊断 gate，不预设需要修改 guidance range 或 action definition。

---

### G-P4：multi-head critic 是否需要改变当前 actor/value sharing？

**状态：暂缓，等待 scalar reward study。**

当前 Exploration Policy 使用 actor/value 相关的共享 representation。未来若 [`reward-objective.md`](../reward-objective.md) 支持 multi-head critic，需要重新决定：

- 多个 value heads 是否共享 trunk；
- actor 与 critic 是否继续共享 scene representation；
- 不同 objective 的 value target scale 是否需要独立处理；
- gradient competition 是否影响 policy update。

在 scalar reward 尚未证明存在不可接受目标冲突前，不提前关闭该 gate，也不启动结构重构。

---

### G-P5：encoder fine-tuning 后 PPO stability 如何保持？

**状态：暂缓，等待 information study。**

当前 information ablation 的第一阶段使用 frozen pretrained encoder。

如果 [`information-representation.md`](../information-representation.md) 证明某些信息有稳定增量价值，并进一步进入 encoder fine-tuning，则需要决定：

- 哪些参数参与 RL update；
- encoder 与 actor/critic 的优化耦合是否导致 instability；
- 是否需要不同 learning rate 或 staged unfreezing；
- frozen reference planner 与 trainable RL representation 如何保持实验可解释性。

这些问题只在 representation adaptation 被实验支持后进入 active work。

---

### G-P6：是否需要 scale-out PPO runtime？

**状态：按需开放。**

scale-out 不再是研究路线中的独立阶段。

只有当 reward / information 实验已经显示值得扩大样本量的信号，并且 simulator throughput 成为主要阻碍时，才考虑：

- larger rollout batches；
- vectorized / multi-process rollout；
- longer training；
- additional training orchestration。

是否 scale-out 应由研究预算和统计需求驱动，而不是为了接近 PlannerRFT 原论文规模。

## 已关闭的历史 Gates

以下问题已经完成决策或实现验证，不再作为 active research work 展开。

| Gate | 状态 | 决策来源 |
| --- | --- | --- |
| G-01 PPO MDP step | Closed | ADR 0017；Issue #18 |
| G-02 reference trajectory generation | Closed | ADR 0013；Issue #6 |
| G-03 orthogonal guidance / neutral action | Closed | ADR 0013；Issue #6 |
| G-04 Exploration Policy observation baseline | Closed for current baseline | ADR 0016；Issue #16 |
| G-05 Beta action mapping / probability accounting | Closed | ADR 0016；Issue #16 |
| G-06 candidate count `K` | Closed for current PPO-only design | ADR 0016；Issue #16 |
| G-08 terminal / truncation bootstrap | Closed | ADR 0017、0018；Issues #18、#19 |
| G-11 DDIM initial distribution / timestep schedule | Closed | ADR 0011；Issue #4 |
| G-12 guidance gradient space / target channels | Closed | ADR 0013；Issue #6 |

旧 G-07（energy objective）已经被提升为总体研究方向，不再作为 PPO-specific gate；其内容见 [`reward-objective.md`](../reward-objective.md)。

旧 G-10（PlannerRFT reproduction 与 Eco-AutoDrive extension 区分）现在作为文档规范处理：外部来源事实见 primary-source notes，项目研究结论必须使用 Eco-AutoDrive 自身命名与实验边界。

## 研究不变量

- 当前实现行为以 `system-contract.md` 为准；
- 外部方法事实以 primary-source notes 为准；
- 已接受的长期决定以 ADR 为准；
- map、diffusion noise、policy action 等随机来源必须能够独立解释；
- 原始规划结果、实际执行结果和 termination information 不因为结果不理想而过滤；
- 运动学闭环结论不直接外推为低层车辆动力学结论；
- MetaDrive proxy energy 与更精细能耗模型使用不同名称和结论边界；
- reward、information、representation 与 PPO optimizer 尽量通过分阶段消融解耦。

## 从 Gate 到实现

```text
research question
      ↓
design gate
      ↓
ADR（若为长期技术决定）
      ↓
GitHub Issue
      ↓
implementation
      ↓
experiment evidence
```

本文件不维护具体配置、实验预算或 Issue 完成进度。
