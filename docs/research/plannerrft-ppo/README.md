# PlannerRFT PPO-only 研究

## 研究问题

本专题研究：

> PlannerRFT 的 Exploration Optimization 思路是否可以作为 Eco-AutoDrive 中优化长程驾驶表现和能耗的一种有效方法？

这里关注的是研究问题，而不是当前实现说明。

仓库已经具备 PlannerRFT PPO-only 路线所需的主要基础设施，包括 policy-guided planning、closed-loop rollout、GAE/PPO update 和 MetaDrive smoke training。其当前行为和数据契约统一记录在
[`system-contract.md`](../../agents/system-contract.md)。

基础设施已经实现并不意味着：

- PPO 已被选为最终能耗优化方法；
- 当前 reward 是最终 energy reward；
- 当前实现与 PlannerRFT 论文训练环境完全 parity；
- learned guidance 已被证明优于非 RL 方法。

这些仍属于研究问题。

## 研究范围

本专题只讨论 PlannerRFT 的 PPO / Exploration Optimization 路线。

概念上：

```text
frozen Diffusion Planner
        |
        +--> reference trajectory
        |
        +--> Exploration Policy
                  |
             guidance action
                  |
        guided trajectory generation
                  |
             MetaDrive rollout
                  |
                 reward
                  |
              PPO update
````

研究重点是让 Exploration Policy 学习如何改变冻结规划器的 guidance，而不是直接使用 PPO 更新 Diffusion Planner 的 DiT 参数。

GRPO 不属于本专题。

## PlannerRFT 与 Eco-AutoDrive 的概念映射

| PlannerRFT 概念             | Eco-AutoDrive 中的对应              |
| ------------------------- | ------------------------------- |
| pretrained planner        | 预训练 Diffusion Planner           |
| reference planning result | 冻结 planner 产生的参考轨迹              |
| Exploration Policy        | 根据场景和参考轨迹产生 guidance action 的策略 |
| exploration action        | 横向/纵向 guidance                  |
| closed-loop simulator     | MetaDrive                       |
| environment reward        | MetaDrive 中定义的实验 objective      |
| PPO optimization          | 更新 Exploration Policy           |
| long-horizon evaluation   | MetaDrive 多规划周期闭环运行             |

这一映射只说明 PlannerRFT 方法如何被解释到本项目中。

PlannerRFT 使用的训练环境、车辆模型、交通来源、奖励组成和实验规模与 Eco-AutoDrive 不相同，因此论文结果不能直接作为本项目的数值验收阈值。

## 与能耗研究的关系

PlannerRFT PPO-only 是候选优化方法之一。

最终研究问题不是：

> “如何继续完成 PPO 实现？”

而是：

> “在已经具备 PPO 实验基础设施的前提下，PPO 是否比固定 guidance、无 guidance 或非 RL 方法更适合利用长程信息改善能耗？”

因此后续研究需要首先区分：

1. PlannerRFT 方法解释或 parity 问题；
2. Eco-AutoDrive 自身的 energy-oriented extension；
3. PPO 与其他候选优化方法之间的方法选择。

## 文档导航

* [一手资料核查](../plannerrft-ppo-primary-sources.md)
  PlannerRFT 论文、补充材料和官方代码能够支持的事实，以及未公开的实现细节。

* [Design gates](design-gates.md)
  尚未解决的研究决定，以及已经关闭 gate 的简要索引。

* [Remaining-work plan](implementation-plan.md)
  在现有 PPO 基础设施之上的剩余研究和实验工作。

* [System contract](../../agents/system-contract.md)
  当前已经实现的 planner、guidance、rollout、reward transport、GAE 和 PPO 行为。

* [ADRs](../../adr/)
  已经接受的项目级技术决定及其理由。

* [Experiment records](../../experiments/)
  已完成运行的实验配置、结果和 provenance。

## 相关 Issues

当前可执行工作的状态以 GitHub Issues 为准，不在本文复制。

与本专题直接相关的历史实现 Issues 包括：

* #4 — 5-step DDIM sampler
* #6 — reference planner 与 orthogonal guidance
* #16 — Exploration Policy 与 value head
* #18 — 10 Hz closed-loop rollout
* #19 — TorchRL GAE/PPO updater
* #20 — closed-loop PPO smoke training

这些工作已经完成，不再作为本专题的待实现阶段。

当前研究文档整理由 #32 跟踪。

与能耗研究基础直接相关的 active work 包括：

* #1 — 建立稳定的短程或中程能耗场景矩阵
* #3 — 在稳定场景基线上比较现有能耗指标

新的 PPO 消融、energy reward、scale-out training 或 preview-conditioned optimization 工作，应在研究决定明确后建立独立 Issue，而不是直接加入本文件作为承诺任务。
