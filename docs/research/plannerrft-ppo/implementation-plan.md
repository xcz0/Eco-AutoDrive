# PlannerRFT PPO-only Remaining-Work Plan

## 目的

PlannerRFT PPO-only 的基础实现阶段已经完成。

本文件不再描述 Stage 0–6 的实现步骤，而只记录：

- 已完成工作的历史索引；
- 在现有基础设施之上仍可能需要完成的研究工作；
- 哪些候选工作只有在研究决定成立后才值得转成 GitHub Issue。

当前实现契约见
[`system-contract.md`](../../agents/system-contract.md)。

实际开发状态以 GitHub Issues 为准。

## 已完成的基础工作

此前的阶段路线可以压缩为以下历史索引：

| 历史阶段 | 结果 | 主要 Issue |
| --- | --- | --- |
| baseline / prerequisite validation | 已完成作为后续工作的基础 | experiment records |
| DDIM-5 | 已实现并验证 | #4 |
| reference planner + orthogonal guidance | 已实现并验证 | #6 |
| Exploration Policy + value head | 已实现 | #16 |
| 10 Hz closed-loop PPO rollout | 已实现 | #18 |
| GAE + PPO updater | 已实现 | #19 |
| closed-loop PPO smoke training | 已实现并验收 | #20 |

这些阶段不再是 remaining work。

其详细实现行为不在本文件重述。

长期技术决定见 [`docs/adr/`](../../adr/)，运行证据见
[`docs/experiments/`](../../experiments/)。

## 当前状态

目前已经可以进行：

```text
frozen planner
    ↓
Exploration Policy
    ↓
guided planning
    ↓
MetaDrive closed-loop rollout
    ↓
reward
    ↓
GAE / PPO update
````

因此下一阶段研究的重点不再是“把 PPO 跑起来”，而是判断：

> PPO 是否值得作为 Eco-AutoDrive 的能耗优化方法继续投入。

这仍是开放研究问题。

## Remaining Work 1：建立可比较的能耗评价基础

在比较 learned guidance 之前，需要有稳定且可解释的能耗研究基线。

当前相关 active Issues：

* #1 — 建立稳定的短程或中程能耗场景矩阵
* #3 — 在稳定场景基线上比较现有能耗指标

这里需要解决的是评价问题，而不是 PPO infrastructure 问题：

* 什么场景具有代表性；
* 什么能耗指标具有可比性；
* termination type 如何分层；
* 实际执行距离、速度和能耗如何共同解释；
* proxy energy 是否足以支撑方法筛选。

这些工作完成前，可以继续进行 PPO 工具链测试，但不应把 smoke objective 的变化解释为最终 energy improvement。

## Remaining Work 2：决定 PPO 的研究 objective

对应 [G-07](design-gates.md)。

需要明确区分：

```text
smoke-training objective
        ≠
PlannerRFT parity reward
        ≠
Eco-AutoDrive energy objective
```

后续研究需要决定：

* 是否需要 PlannerRFT reward parity；
* 是否直接研究 energy-oriented reward；
* energy signal 是 reward 的组成部分、主要目标还是独立 evaluation metric；
* 如何防止停车、低速或任务失败造成错误优化方向；
* 如何报告安全、有效进度、旅行时间、舒适性和能耗之间的 trade-off。

只有在研究定义明确后，具体 reward implementation 才应成为新的 GitHub Issue。

## Remaining Work 3：验证 learned guidance 是否提供实际价值

如果继续研究 PPO，应使用配对实验回答 learned Exploration Policy 是否优于更简单的方法。

基本比较对象可包括：

```text
unguided planner
fixed guidance
random / observation-independent guidance
learned Exploration Policy
```

研究重点不是单独观察 PPO loss，而是判断：

* learned action 是否真正随 observation 改变；
* learned guidance 是否在相同场景和随机条件下改善目标指标；
* 改善是否来自有效驾驶行为，而不是停车或失败；
* 结果是否跨 seed 保持；
* 不同 termination type 下结论是否一致；
* PPO 是否比固定或非 RL 方法提供足够增益。

具体实验矩阵、指标和统计方法应在建立对应 GitHub Issue 时确定。

如果 learned guidance 不能稳定优于简单对照，则没有必要因为 PPO infrastructure 已经存在而继续扩大训练规模。

## Remaining Work 4：决定是否需要规模化训练

对应 [G-09](design-gates.md)。

规模化不是当前默认下一步。

只有当小规模实验表明：

1. learned guidance 存在值得继续研究的信号；
2. 当前结果主要受 rollout 数量或训练规模限制；
3. 更大的样本量能够回答明确的研究问题；

才需要研究：

* vectorized environment；
* multi-process rollout；
* larger PPO batches；
* 更长训练；
* 独立 training orchestration。

如果当前小规模运行足以回答方法选择问题，则不需要为了接近论文资源规模而进行 scale-out。

## Remaining Work 5：道路预瞄与 PPO 的关系

PlannerRFT PPO-only infrastructure 本身不等于道路预瞄方法。

如果项目后续加入新的 preview information，需要单独回答：

```text
preview information
       ↓
representation / encoder
       ↓
planner or policy conditioning
       ↓
behavior change
       ↓
energy effect
```

需要区分至少两种问题：

1. planner 能否利用新的 preview information；
2. PPO 是否有助于学习如何利用这些信息。

因此不应默认把“加入 preview”与“使用 PPO”绑定成同一个实现步骤。

可能先通过监督式、固定 guidance 或推理期方法验证 preview signal 是否有价值，再决定是否使用现有 PPO infrastructure。

具体 preview 表示和注入方式属于独立研究问题，不在本文提前固定。

## Remaining Work 6：决定 PlannerRFT parity 是否仍有研究价值

对应 [G-10](design-gates.md)。

如果研究目标主要转向 Eco-AutoDrive 的能耗与道路预瞄，完整复现 PlannerRFT 的训练规模或 reward 未必是必要条件。

需要明确：

* 哪些 PlannerRFT 事实只用于方法理解；
* 哪些部分需要 parity experiment；
* 哪些部分只是项目复现决定；
* 哪些实验已经属于 Eco-AutoDrive extension。

如果未来仍进行 parity experiment，应与 energy-oriented experiment 使用不同名称、配置和结论范围。

## 建立新工作的规则

本文件中的 remaining work 都是候选研究工作，不等于 active implementation task。

工作流为：

```text
open research question
        ↓
design gate / research decision
        ↓
ADR（需要长期保留的技术决定）
        ↓
GitHub Issue
        ↓
implementation
        ↓
experiment record
```

GitHub Issue 应负责保存：

* 明确目标；
* 非目标；
* 受影响代码；
* 配置；
* 测试；
* 实验矩阵；
* acceptance criteria。

本文件不重复这些实施细节，也不维护 Issue 的完成进度。

## 当前最重要的研究判断

PlannerRFT PPO-only 路线当前需要回答的核心问题可以缩写为：

```text
PPO infrastructure exists
          ↓
energy evaluation becomes reliable
          ↓
define a meaningful optimization objective
          ↓
paired learned-vs-simple-guidance experiment
          ↓
Does PPO add value?
       /        \
     yes         no
      |           |
consider scale   keep simpler
or preview RL    optimization route
```

因此，后续是否继续扩展 PPO 应由实验结果驱动，而不是由旧 Stage 路线图驱动。
