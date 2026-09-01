# PlannerRFT PPO-only Method Notes

## 定位

本目录保留 PlannerRFT Exploration Optimization / PPO-only 路线的研究说明。

它现在是 Eco-AutoDrive 的**长程闭环优化工具之一**，而不是项目总体研究路线。

项目总体研究问题已经重新组织为：

- [Reward and objective study](../reward-objective.md)：如何设计长程 optimization objective；
- [Information and representation study](../information-representation.md)：策略需要哪些环境信息；
- [Ablation plan](../ablation-plan.md)：如何分离 objective、information、representation 的主效应与交互效应。

PlannerRFT PPO-only 在其中主要承担：

> 利用 closed-loop rollout + GAE/PPO，把多个仿真时刻后的反馈传播回当前 guidance decision。

因此后续研究关注的不是“是否完成 PPO 复现”，而是这个优化工具能否帮助回答 objective 与 information 两个核心问题。

## 方法范围

当前 PPO-only 研究只优化 Exploration Policy，不直接使用 PPO 更新 Diffusion Planner 的 DiT。

概念上：

```text
frozen Diffusion Planner
        |
        +--> reference trajectory
        |
        +--> pretrained scene representation
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
              GAE / PPO
```

GRPO 和直接 DiT fine-tuning 不属于当前 PPO-only baseline。

## PlannerRFT 与 Eco-AutoDrive 的概念映射

| PlannerRFT 概念 | Eco-AutoDrive 当前对应 |
| --- | --- |
| pretrained planner | 预训练 Diffusion Planner |
| reference planning result | frozen planner reference trajectory |
| Exploration Policy | 根据场景与 reference 产生 guidance action |
| exploration action | lateral / longitudinal guidance |
| closed-loop simulator | MetaDrive |
| environment reward | 当前实验 objective |
| PPO optimization | 更新 Exploration Policy |
| long-horizon feedback | closed-loop return / GAE |

这只是方法映射，不表示项目与 PlannerRFT 的 simulator、reward、数据规模或训练配置 parity。

外部论文和官方代码事实统一见 [`plannerrft-ppo-primary-sources.md`](../plannerrft-ppo-primary-sources.md)。

## 当前证据

仓库已经具备：

- reference-centered orthogonal guidance；
- Exploration Policy + value head；
- closed-loop rollout；
- GAE / PPO update；
- reward transport 与 training artifact；
- PPO 稳定性搜索与多 seed 长程验证工作流。

E-028 在固定 `metadrive_builtin_v1`、no-traffic S/SC 条件下确认了一个 3 seeds × 100 updates 的稳定候选配置，并观察到较多 epochs 与较高 learning rate 可能推动 Beta guidance distribution 向边界塌缩。

这说明当前 PPO-only infrastructure 已经可以作为研究工具，但不支持：

- PPO 能够改善能耗；
- 当前 `plannerrft_energy_v1` 优于 builtin reward；
- 当前配置可以直接迁移到有交通、更长训练或新的 observation；
- learned guidance 一定优于 fixed guidance；
- PPO 是最终能耗优化算法。

E-026 中曾出现的大量 out-of-road 退化后来被 E-028 定位为 MetaDrive reset 语义 bug，因此不再作为 PPO update 强度过大的证据。

## 在新研究路线中的作用

### 对 Reward study

PPO-only baseline 用来研究：

- scalar reward structure 是否产生不同 long-horizon behavior；
- energy normalization / weight 如何影响 trade-off；
- 更长 credit assignment 是否比短程反馈提供额外价值。

PPO hyperparameters 在这类实验中原则上应作为控制变量，而不是和 reward 同时搜索。

### 对 Information study

Exploration Policy 当前可以借用预训练 scene representation，研究：

- ego/reference 是否已经足够；
- local scene 是否有增量价值；
- long-range road/navigation information 是否有增量价值；
- dynamic traffic information 是否有增量价值。

只有信息价值被确认后，才需要进一步研究 encoder fine-tuning。

### 对后续 multi-objective extension

如果 scalar reward 出现持续目标冲突，可以在现有 PPO baseline 上扩展 multi-head critic 或 constrained objective。

这属于 [`reward-objective.md`](../reward-objective.md) 的第二阶段，不需要在当前 PPO-only method notes 中预设结构。

## PPO-specific open questions

当前真正与 PPO 工具本身相关、仍可能影响研究解释的问题主要是：

- 稳定候选配置迁移到 energy reward / richer observation 后是否仍稳定；
- 不同 credit horizon 下，GAE 是否提供可测的 long-horizon benefit；
- guidance action distribution 的变化是否与闭环行为变化一致；
- 当前 shared actor/value representation 在 multi-head critic 扩展时是否仍合适；
- 当 encoder 允许 fine-tune 时，PPO stability 是否发生新的变化。

这些问题只在对应主研究实验需要时再转为 active Issue。

## 文档导航

- [Primary-source notes](../plannerrft-ppo-primary-sources.md)
- [PPO-specific design gates](design-gates.md)
- [PPO support / remaining work](implementation-plan.md)
- [System contract](../../agents/system-contract.md)
- [Experiment records](../../experiments/)

## 历史实现索引

与 PPO-only infrastructure 直接相关的历史 Issues 包括：

- #4 — 5-step DDIM sampler
- #6 — reference planner 与 orthogonal guidance
- #16 — Exploration Policy 与 value head
- #18 — closed-loop rollout
- #19 — GAE/PPO updater
- #20 — closed-loop PPO smoke training
- #59 — energy-oriented scalar reward 与 matched reward A/B
- #76 — PPO stability search

这些是已完成或已存在的工程/实验基础，不再构成总体研究阶段。
