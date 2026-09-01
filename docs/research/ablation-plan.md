# Ablation Plan

## 目的

本文件组织研究级消融逻辑，不维护具体运行参数和 Issue 状态。

核心目标是把“长程反馈优化能耗”拆成可解释的因果比较，优先回答：

1. 哪种 optimization objective 能产生可接受的 energy trade-off；
2. 哪些 environment information 提供独立增量价值；
3. representation adaptation 和 multi-head critic 是否在前两者已经有效的基础上继续提供收益。

## 基本原则

### 一次只回答一个主要问题

Reward study 中固定 observation；information study 中固定 reward；representation study 中固定 information 和 objective。

避免同时改变 reward、encoder、observation 和 PPO hyperparameters 后只比较最终分数。

### 使用 matched evaluation

主要比较应尽量保持：

- scenario / map seed 一致；
- planner noise / policy action seed 可解释；
- evaluation horizon 与 termination semantics 一致；
- 初始 checkpoint 一致；
- reward-independent metrics 一致。

训练 reward 只用于优化和诊断，不作为方法优劣的唯一证据。

### 先做主效应，再做交互效应

不直接做完整笛卡尔积。

先分别筛选：

- 代表性的 reward formulation；
- 代表性的 information set；
- 是否值得 fine-tune encoder；
- 是否值得扩展 multi-head critic。

最后只对少量代表配置做交叉实验。

## Stage 1 — Reward ablation

固定：

```text
information set
encoder state
PPO baseline
training/evaluation scenarios
```

改变：

```text
reward structure
energy normalization
energy weight
credit horizon
```

推荐研究角色：

| Role | Purpose |
| --- | --- |
| R0 | 不含 energy term 的 baseline objective |
| R1 | 当前 scalar energy objective |
| R2 | alternative energy normalization / composition |
| R3 | energy trade-off weight variants |
| R4 | short vs longer credit assignment |

具体公式不在本文件固定，以 [`reward-objective.md`](reward-objective.md) 为研究定义，以后续 Issue 为执行配置。

Stage 1 的主要输出不是“best reward”，而是：

- 是否存在稳定的 learned behavioral effect；
- energy 改善是否伴随不可接受的 speed/progress/safety 退化；
- 哪些 reward 结构能够形成可解释的 Pareto / trade-off 区域；
- 长程 credit assignment 是否比短程反馈提供独立价值。

若所有 scalar reward 都只能通过明显降速、少走或失败降低 energy，则优先修正 objective，而不是进入更复杂 observation 或 encoder fine-tuning。

## Stage 2 — Information ablation

选择 Stage 1 中一个稳定且有代表性的 scalar objective，固定 reward 与 optimizer。

改变 policy 可观察的信息：

| Role | Information |
| --- | --- |
| I0 | ego state + reference trajectory |
| I1 | I0 + local scene |
| I2 | I1 + long-range road/navigation information |
| I3 | representative dynamic traffic information |

详细定义见 [`information-representation.md`](information-representation.md)。

Stage 2 的主要输出是每类信息的增量贡献，而不是 full-observation 模型的绝对分数。

至少需要回答：

- local scene 相比 ego/reference 是否有增益；
- long-range road/navigation 是否有独立增益；
- dynamic traffic information 是否有独立增益；
- 某类信息的增益是否只在特定场景类型出现；
- 增益是否表现为真正的提前速度/行为调整，而不是终止或随机性差异。

## Stage 3 — Representation ablation

只对 Stage 2 中已经证明有价值的信息做 representation study。

核心比较：

```text
same information
same reward
same PPO

frozen pretrained encoder
vs
partially fine-tuned encoder
```

必要时再增加 joint fine-tuning，但不作为默认实验。

Stage 3 需要回答：

- fine-tuning 是否提供独立 energy / behavior gain；
- gain 是否来自 information exploitation，而不是破坏原 planner representation；
- safety / progress / speed 是否保持；
- 增益是否足够大到值得承担更高训练复杂度和稳定性风险。

## Stage 4 — Objective extension

如果 Stage 1 显示 scalar reward 存在明显目标冲突或高度 weight-sensitive，再进入多头 critic / constrained objective。

核心比较可压缩为：

```text
single scalar critic
vs
multi-head value estimation
```

只选择 Stage 2 已筛出的少量 information set，不重新做所有 information 组合。

这一阶段关注的是：

- multi-head critic 是否改善 energy 与其他目标的 trade-off；
- 是否降低 reward-weight sensitivity；
- 是否改善长程 credit assignment 的可解释性或稳定性。

## Stage 5 — Joint factorial ablation

最终只选择少量代表因素做交叉验证，例如：

```text
objective:
  baseline / best scalar / optional multi-head

information:
  minimal / local / local + useful long-range information

representation:
  frozen / tuned (only if Stage 3 supports it)
```

不要求完整三维笛卡尔积。优先选择能够回答交互问题的最小矩阵。

关键交互问题包括：

1. **Information × Objective**  
   新信息是否只有在 energy-oriented long-horizon objective 下才有价值？

2. **Representation × Information**  
   fine-tuning 是否只对某类长程信息有帮助？

3. **Objective × Representation**  
   更复杂 objective 是否只有在 trainable representation 下才产生增益？

4. **Long-horizon feedback × information**  
   长程信息是否需要长程 credit assignment 才能转化为行为收益？

## 统一指标

主表至少包含：

| Category | Metrics |
| --- | --- |
| Energy | total proxy energy、energy intensity / normalized variants |
| Task | route progress、completion、distance |
| Efficiency | mean speed、travel time / travel efficiency |
| Safety | collision、out-of-road、wrong-direction、必要时 TTC |
| Comfort | 当 reward 或行为变化涉及舒适性时报告 |

不同 energy normalization 可作为 reward 变量，但最终 evaluation 应保留至少一个统一 reward-independent energy metric，避免“优化哪个指标就用哪个指标证明自己更好”。

## 结果解释

### 可以支持 energy improvement 的基本形式

```text
energy metric improves
AND
safety does not materially regress
AND
progress / completion remains comparable
AND
speed / travel efficiency does not collapse
AND
result is reproducible across matched seeds/scenarios
```

### 常见伪改善

以下情况应单独标记，不计为 energy optimization 成功：

- 提前 termination 导致 total energy 更低；
- distance / progress 显著减少；
- 长时间停车；
- mean speed 显著下降；
- 只在单一 seed 或极少场景出现；
- training reward 上升但独立 evaluation 无改善。

## 证据推进顺序

```text
mechanical validity
      ↓
optimization stability
      ↓
behavioral effect
      ↓
energy effect
      ↓
information contribution
      ↓
representation / objective extension contribution
      ↓
energy-model robustness
```

后续阶段不能替代前一阶段。例如，full model 在更精细 energy model 上表现较好，也不能补足缺失的 information ablation；同样，训练非常稳定也不能证明 reward 有效。

## 与工程工作的边界

本文件不固定：

- PPO update 数；
- batch / minibatch；
- exact reward weights；
- preview distance；
- encoder layer freeze policy；
- seed 数量与显著性检验方法。

这些应在具体研究假设转化为 active work 后，由 GitHub Issue 和 experiment record 维护。
