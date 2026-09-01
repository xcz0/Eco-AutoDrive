# PlannerRFT PPO-only Support Plan

## 目的

PlannerRFT PPO-only 的基础实现阶段已经完成。

本文件不再维护总体研究路线，也不重复 DDIM、reference guidance、Exploration Policy、rollout、GAE/PPO 等已完成实现。

总体研究计划见：

- [`../README.md`](../README.md)
- [`../reward-objective.md`](../reward-objective.md)
- [`../information-representation.md`](../information-representation.md)
- [`../ablation-plan.md`](../ablation-plan.md)

本文件只说明：当这些研究需要 PPO-only 工具继续演化时，哪些支持工作可能值得转化为 GitHub Issue。

## 已完成的基础工作

| 能力 | 状态 | 主要来源 |
| --- | --- | --- |
| 5-step DDIM baseline | 已实现并验证 | #4 / experiments |
| reference planner + orthogonal guidance | 已实现并验证 | #6 |
| Exploration Policy + value head | 已实现 | #16 |
| closed-loop rollout | 已实现 | #18 |
| GAE + PPO updater | 已实现 | #19 |
| PPO smoke training | 已实现 | #20 |
| energy-oriented scalar reward transport | 已实现并完成 matched short A/B | #59 / E-024~E-026 |
| PPO stability search workflow | 已实现并形成受限稳定候选 | #76 / E-028 |

这些能力是后续研究的实验工具，不再构成 remaining implementation stages。

## 当前 PPO baseline 的角色

```text
frozen planner + pretrained scene representation
                    ↓
             Exploration Policy
                    ↓
             guidance action
                    ↓
          closed-loop MetaDrive
                    ↓
            scalar reward
                    ↓
               GAE / PPO
```

当前默认研究策略是：

1. 尽量固定 PPO optimizer；
2. 先研究 reward；
3. 再研究 information；
4. 最后才考虑 critic / encoder 的结构扩展。

因此 PPO 代码只有在成为研究瓶颈时才继续扩展。

## Support Work A — Reward study support

对应 [`../reward-objective.md`](../reward-objective.md)。

可能需要的支持工作包括：

- 将不同 scalar reward formulation 保持为明确、可审计的 profile；
- 统一记录 objective components、raw metrics 和 final scalar reward；
- 支持 credit-horizon / return ablation 所需的配置；
- 保证 training reward 与 reward-independent evaluation metric 可区分；
- 保持 matched seed / initial-policy / rollout pairing。

只有当具体 reward hypothesis 确定后，才建立实现 Issue；本文件不提前固定 reward 公式或权重。

## Support Work B — Information ablation support

对应 [`../information-representation.md`](../information-representation.md)。

第一阶段目标是允许 policy 使用不同层级的 observation，同时保持 pretrained encoder frozen。

可能需要的支持工作包括：

- 对 information group 做显式 mask / include-exclude 控制；
- 保持 observation shape / artifact 可审计；
- 能够区分 ego/reference、local scene、long-range road/navigation、traffic information；
- 支持 matched perturbation / range ablation；
- 记录 policy action 对信息变化的响应。

这里优先支持实验控制，而不是提前设计复杂的新 encoder。

## Support Work C — Encoder extraction and fine-tuning

只有在 information ablation 已经确认某类信息有稳定增量价值后才进入。

可能的演化路径：

```text
borrowed pretrained encoder, frozen
          ↓
explicit RL perception module
          ↓
partial fine-tuning
          ↓
optional joint optimization
```

此阶段可能需要：

- 清晰区分 frozen reference planner 与 trainable RL representation；
- 参数组和 checkpoint contract；
- encoder-specific optimizer / learning-rate control；
- representation drift / gradient diagnostics；
- frozen-vs-tuned matched evaluation。

是否需要这些能力由 Stage 2/3 实验结果决定。

## Support Work D — Multi-head critic

只有 scalar reward study 显示明显 objective conflict 或高度 weight sensitivity 后再进入。

可能需要支持：

- objective-specific return / value target；
- multiple value heads；
- per-head diagnostics；
- policy advantage aggregation；
- 必要时的 constraint / Lagrangian bookkeeping。

在真正进入实现前，需要先通过 [`design-gates.md`](design-gates.md) 的 G-P4 明确 actor/value sharing 与优化语义。

## Support Work E — Stability re-validation

任何改变以下条件的实验，都不能无条件继承 E-028 的稳定性结论：

- reward scale / reward structure；
- observation richness；
- encoder trainability；
- critic structure；
- traffic complexity；
- training horizon。

因此新研究实验启动前，应有最小规模的 mechanical/stability gate。

该 gate 的目的不是重复完整 Optuna search，而是确认：

- diagnostics finite；
- guidance distribution 不发生快速边界塌缩；
- closed-loop behavior 无明显 catastrophic failure；
- final checkpoint 在固定 evaluation 上没有基础能力崩溃。

只有稳定性确实重新成为主要问题时，才需要新的 hyperparameter search。

## Support Work F — Scale-out runtime

scale-out 不是默认目标。

只有在研究实验已经出现可靠 signal，且样本量或 simulator throughput 明确限制结论时，再考虑：

- larger rollout batch；
- vectorized / multi-process collection；
- longer training；
- distributed orchestration。

运行时优化应服务于明确实验问题，不作为独立研究贡献。

## 新工作的建立规则

```text
research hypothesis
      ↓
minimal ablation design
      ↓
PPO support gap identified
      ↓
design gate / ADR if needed
      ↓
GitHub Issue
      ↓
implementation
      ↓
experiment record
```

GitHub Issue 应负责具体：

- 目标与非目标；
- 受影响代码；
- 配置；
- 测试；
- 实验矩阵；
- acceptance criteria。

本文件只维护“哪些 PPO 支持能力可能需要”，不维护 active task 进度。

## 当前优先级

在新的总体研究路线下，PPO-only 支持工作的优先级为：

```text
1. support scalar reward experiments
2. support controlled information ablations
3. re-validate stability under changed conditions
4. only then consider encoder fine-tuning or multi-head critic
5. scale out only when experiments require it
```

因此近期工作重点是让 PPO 成为稳定、可控、可消融的研究工具，而不是继续追求 PlannerRFT implementation parity。
