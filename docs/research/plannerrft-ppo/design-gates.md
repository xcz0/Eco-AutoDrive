# PlannerRFT PPO-only Design Gates

## 目的

本文件只维护仍会影响研究结论、实验设计或后续架构选择的开放问题。

已经关闭的 gate 不再复制当前实现细节。其长期决定由 ADR 保存，当前实现行为由
[`system-contract.md`](../../agents/system-contract.md) 保存。

当某个开放问题已经转化为明确的可执行工作时，应建立 GitHub Issue；Issue 而不是本文件负责跟踪实现进度。

## 当前开放 Gates

### G-07：energy-oriented optimization objective 应如何定义？

**状态：部分关闭，energy extension 仍开放。**

已经完成：

- MetaDrive reward 已足以支持 closed-loop PPO smoke training；
- smoke objective 的定位和使用理由已经形成项目决定；
- PPO 数据流和更新器不再依赖“最终 reward 必须先设计完成”才能存在。

仍未解决：

- 最终能耗优化是否直接使用 reward-based RL；
- energy term 使用哪一种指标；
- energy、安全、有效进度、旅行时间和舒适性如何组合；
- 如何避免通过停车、低速或任务失败获得表面上的低能耗；
- 不同 termination type 是否应该进入同一 reward 比较；
- 是否需要 PlannerRFT-style parity reward 作为中间实验，还是直接研究 Eco-AutoDrive objective。

相关来源：

- ADR 0019
- [`system-contract.md`](../../agents/system-contract.md)
- GitHub Issue #3
- [`primary-source notes`](../plannerrft-ppo-primary-sources.md)

关闭该 gate 需要的是研究 objective 和比较口径的明确决定，而不是重新证明 PPO 基础设施存在。

---

### G-09：是否需要 scale-out PPO training runtime？

**状态：开放。**

当前已有 smoke training 足以验证训练链路，但尚未回答是否需要更大规模的 rollout/runtime。

需要先回答：

- 小规模配对实验能否观察到稳定的 learned-guidance signal；
- 研究问题需要多少场景和随机 seed 才能形成有意义的统计比较；
- simulator wall-clock 是否已经成为主要实验瓶颈；
- scale-out 的复杂性是否会影响随机性、重放和实验解释。

候选方向包括：

- 保持当前小规模/串行训练；
- 同步向量环境；
- 多进程 rollout collection；
- 其他独立训练 orchestration。

只有当实验规模本身成为研究阻碍时，才需要关闭该 gate 并建立相应实现 Issue。

当前 evaluation runtime 的系统行为见
[`system-contract.md`](../../agents/system-contract.md)。

---

### G-10：PlannerRFT reproduction 与 Eco-AutoDrive extension 如何分开？

**状态：开放。**

PlannerRFT 的原始研究目标与 Eco-AutoDrive 的道路预瞄和能耗目标不同。

需要明确区分至少三类结论：

1. **source fact**  
   PlannerRFT 论文或官方代码实际公开的内容。

2. **project reproduction choice**  
   为了在 Eco-AutoDrive 中实现 PPO-only 路线而作出的项目级决定。

3. **Eco-AutoDrive extension**  
   道路预瞄、能耗 reward、MetaDrive 场景和其他本项目新增内容。

任何实验报告都不应把第 2、3 类内容描述为 PlannerRFT 作者事实。

需要进一步决定：

- 是否仍需要专门的 PlannerRFT parity experiment；
- 如果需要，parity 的目的和停止条件是什么；
- energy-oriented experiment 是否应与 parity experiment 使用完全独立的命名和指标。

外部来源事实统一见
[`primary-source notes`](../plannerrft-ppo-primary-sources.md)。

---

## 已关闭 Gates

以下问题已经完成决策或实现验证，不再作为 active research work 展开。

| Gate | 状态 | 决策来源 |
| --- | --- | --- |
| G-01 PPO MDP step | Closed | ADR 0017；Issue #18 |
| G-02 reference trajectory generation | Closed | ADR 0013；Issue #6 |
| G-03 orthogonal guidance definition / neutral action | Closed | ADR 0013；Issue #6 |
| G-04 Exploration Policy observation | Closed | ADR 0016；Issue #16 |
| G-05 Beta action mapping / probability accounting | Closed | ADR 0016；Issue #16 |
| G-06 candidate count `K` | Closed for current PPO-only design | ADR 0016；Issue #16 |
| G-08 terminal / truncation bootstrap | Closed | ADR 0017、0018；Issues #18、#19 |
| G-11 DDIM initial distribution / timestep schedule | Closed | ADR 0011；Issue #4 |
| G-12 guidance gradient space / target channels | Closed | ADR 0013；Issue #6 |

若未来研究改变这些前提，例如：

- 从单候选变为多候选；
- 改变 PPO action definition；
- 改变 rollout MDP 时间尺度；
- 改变 guidance 作用对象；

应重新打开相应 gate 或建立新的 gate，而不是静默修改已有结论。

## 研究不变量

研究实验仍应保持以下基本边界：

- 当前实现行为以 `system-contract.md` 为准；
- 外部方法事实以 primary-source notes 为准；
- 已接受的长期决定以 ADR 为准；
- map、diffusion noise 和 policy action 等不同随机来源必须能够独立解释；
- 原始规划结果、实际执行结果和 termination information 不应因为结果不理想而被过滤；
- 运动学闭环结论不直接外推为低层车辆动力学结论；
- MetaDrive proxy energy 与 FASTSim 等精细能耗模型必须使用不同名称和单位；
- PlannerRFT reproduction choice 不得描述为作者未公开的实现事实。

## 从 Gate 到实现

Gate 的作用是回答“需要决定什么”，不是维护实施计划。

当 gate 关闭并产生明确工作后：

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
      ↓
system contract（若形成当前系统事实）
````

本文件不维护配置目录建议、阶段依赖链或已经完成工作的进度。
