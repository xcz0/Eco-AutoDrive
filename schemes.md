# 规划器实现提取

来源于 Issue #96 中的 Task A。重构核心目标是把“单次 planner 决策”从 RL rollout 和 evaluation 两套 runtime 中抽出来，让 planning 成为模型决策语义的唯一 owner。

## 1. 当前代码的实际问题

当前 learned-guidance 的决策链大致是：

```text
rl.rollout.FabricRolloutRuntime._decide_batch()
    │
    ├─ Fabric.to_device(observation)
    ├─ sample_batched_standard_normal(...)
    │
    ├─ PretrainedDiffusionPlanner.prepare_policy_guidance(...)
    │     ├─ normalize observation
    │     ├─ encode_policy_features()
    │     ├─ 构造 initial DDIM state
    │     ├─ 准备 guidance randomness
    │     └─ reference DDIM pass
    │
    ├─ 手工 PlannerPolicyContext → ExplorationPolicyContext
    ├─ policy_context_tensordict(...)
    ├─ ExplorationPolicy.forward_tensordict(...)
    ├─ sample / mean policy action
    │
    ├─ PretrainedDiffusionPlanner.complete_policy_guidance(...)
    │     └─ guided DDIM pass
    │
    ├─ build_training_decision(...)       ← RL 数据构造
    ├─ HostTransfer.defer(...)             ← RL audit 字段
    └─ execution_trajectories(...)
```

核心问题集中在 `src/eco_planner/rl/rollout/runtime.py`。

这个文件现在同时拥有四种本来应该分开的责任：

| 当前责任                      | 应归属                                   |
| ----------------------------- | ---------------------------------------- |
| planner + policy 单次组合决策 | `planning.inference`                     |
| policy 输入转换               | `planning.policy.inputs`                 |
| policy action sampling 语义   | `planning.policy` / `planning.inference` |
| PPO TensorDict、rollout audit | `rl.rollout`                             |

而 evaluation 的 learned-policy 路径又通过 `PolicyCheckpointEvaluationAgent(runtime: FabricRolloutRuntime)` 直接复用整个 RL runtime。`evaluation/engine.py` 还直接导入 `rl.optimization.load_exploration_policy_checkpoint` 和 `rl.rollout.create_fabric_rollout_runtime`。这正是 Issue 所说的反向依赖。

同时，base/fixed-guidance 又有另一套 `evaluation/inference/FabricInferenceRuntime`。所以目前实际上存在两套模型执行 runtime：

```text
base / fixed:
evaluation.inference.FabricInferenceRuntime

learned:
rl.rollout.FabricRolloutRuntime
```

Task A 的终态应该把这两个收敛为 planning 所拥有的正式决策能力，但不要求把 training/evaluation 的 episode loop 合并。

---

## 2. 最关键的设计判断：不要重写 prepare/complete

当前 `PretrainedDiffusionPlanner.prepare_policy_guidance()` / `complete_policy_guidance()` 已经实现了 Issue A 最重要的一条科学语义：

> reference 和 guided prediction 在同一次 decision 内共享 encoding、initial noise、DDIM transition randomness。

`PreparedPolicyGuidance` 当前保存：

```text
initial
denoiser
constrain
transition_generator
guidance_randomness
reference_prediction
current_states
policy_context
```

这里前 7 项的基本思路是对的。

真正不合适的是最后的 `policy_context`：diffusion 层正在直接制造 policy schema。

所以这一阶段不应该把 prepare/complete 合并掉，而应该将其改造成：

```text
diffusion.prepare_prediction(...)
    → PreparedPrediction(
          neutral_representations,
          reference_prediction,
          shared DDIM state/randomness,
          ...
      )

policy.inputs.build(...)
    → PolicyInputs

policy(...)
    → PolicyOutput

planning.inference
    → sample/mean action
    → diffusion.complete_guidance(prepared, action)
    → DecisionResult
```

这正好对应 Issue 给出的目标链：

```text
observation
→ diffusion encoding + reference preparation
→ policy input selection / conversion
→ trainable policy encoding + actor/critic
→ guidance action
→ guided sampling
→ decision result
```

---

## 3. 建议的最终 planning 边界

建议 Task A 完成后先形成下面的接口。类名可以再调整，边界本身更重要。

```text
planning/
├── diffusion/
│   ├── checkpoint.py
│   ├── config.py
│   ├── guidance.py
│   ├── network.py
│   ├── planner.py
│   └── sampling.py
│
├── policy/
│   ├── inputs.py
│   ├── config.py
│   ├── model.py
│   ├── distribution.py
│   └── checkpoint.py
│
├── inference.py
├── result.py
└── __init__.py
```

`diffusion` 不应该再 import `planning.policy`。依赖必须是：

```text
planning.inference
    ├── planning.diffusion
    └── planning.policy

planning.policy.inputs
    ↓
读取 diffusion 暴露的中性 representation 类型

planning.diffusion
    ✗ 不知道 policy 的 feature selection
    ✗ 不知道 PPO
    ✗ 不知道 evaluation trace
```

这里可以让 `policy.inputs` import diffusion 的 representation 类型；反过来绝不能成立。

---

## 4. 具体实现步骤

1. **先建立 characterization tests，不先移动代码。** 固定一份 observation、planner checkpoint、policy state、显式 diffusion generator 和 policy generator，记录当前 learned-guidance 单次 decision 的 `reference_prediction`、policy alpha/beta/value、base action、guidance action、joint log-prob、final prediction，以及 generator 调用前后的 RNG state。再为 deterministic mean 建测试，确认 mean 模式不消费 policy RNG。Task A 后续每一步都用这些测试判断是否偷偷改变科学语义。

2. **机械迁移 `models → planning.diffusion`，但暂时不改模型行为。** 当前 `models/checkpoint.py`、`config.py`、`guidance.py`、`network.py`、`planner.py`、`sampling.py` 整体迁入 `planning/diffusion/`。这一 commit 只处理 import path 和测试路径，不顺带重写 sampler/guidance。这样能把“目录迁移 bug”和“决策语义 bug”分开。

3. **迁移 `rl.policy → planning.policy`。** `config.py`、`distribution.py`、`model.py` 搬到 `planning/policy/`。当前 `ExplorationPolicyConfig` 暂时可以保持原数学结构，不在 Task A 就完成 Issue B 的 encoder/fusion/head 大拆分。`evaluation/config.py`、PPO、probe 等全部改为从 `planning.policy` 引用，使 policy 不再由 RL 包拥有。

4. **立即拆 policy checkpoint ownership。** 当前 `rl/optimization/checkpoint.py` 中的 `save_exploration_policy_checkpoint()`、`load_exploration_policy_checkpoint()` 和 policy-only report 应迁到 `planning/policy/checkpoint.py`；training checkpoint 的 optimizer/scheduler/Fabric/loop RNG 部分继续留在 RL。`policy_state_hash` 也应放到 policy-owned 模块或轻量 serialization helper。否则即使 evaluation 不再使用 rollout runtime，它仍然需要 import RL 才能加载 policy，不满足 Task A 的目标。

5. **把 `PlannerPolicyContext` 从 diffusion 中删除。** 将目前 `encode_policy_features()` 暴露的数据改成中性类型，例如 `DiffusionRepresentations`。它可以继续包含现有 scene/navigation token 和 masks，但命名和 API 不再表示“这是给 ExplorationPolicy 用的”。同时将 `PreparedPolicyGuidance` 改成 `PreparedPrediction` 一类的 diffusion-owned 类型，里面保留 reference prediction 和完整共享 sampling state。关键约束是：reference pass 仍然执行，不能因为某个新 policy ablation 不读 reference 就跳过 reference sampling。

6. **新增最小版 `planning.policy.inputs`。** Task A 阶段先做当前行为的 1:1 adapter，而不是立刻实现任意 feature ablation。比如 `build_policy_inputs(representations, reference_prediction)` 负责生成当前五项输入：scene tokens/mask、navigation tokens/mask、reference trajectory。这样当前 `_decide_batch()` 中手写的 `PlannerPolicyContext → ExplorationPolicyContext` 映射消失。以后 Issue B 做 representation 消融，只改这个 owner 和对应配置，不需要再碰 collector/evaluation。

7. **抽取 learned-guidance 的核心 `_decide_batch()` 到 `planning.inference`。** 从 `FabricRolloutRuntime._decide_batch()` 移走：observation H2D、diffusion noise generation、`prepare_prediction()`、policy forward、policy sample/mean、`complete_guidance()`。planning 层输出 `DecisionResult`，但不要调用 `build_training_decision()`，也不要生成 evaluation trace TensorDict。建议 `DecisionResult` 至少表达 final trajectory、reference trajectory、policy inputs、policy output、sampled policy action以及 guidance diagnostics。RL 所需的 old log-prob/value 应作为 policy decision 的正常输出存在，而不是叫 PPO 字段。

8. **不要为了统一而把三种方法塞成一个巨型 `if mode` runner。** base、fixed guidance、learned guidance 应保留明确路径，但都属于 `planning.inference`。现有 `evaluation/inference/FabricInferenceRuntime` 中的 base/fixed planner model execution 应迁到 planning；learned 路径从 rollout runtime 迁入同一领域。可以共享 H2D、noise sampling、shape validation、host execution conversion 等 helper，但 base 不应为了“代码统一”强行经过 learned-policy prepare/complete 流程，否则极容易改变 DDIM RNG consumption。

9. **RL 改成 DecisionResult → training adapter。** 当前 `build_training_decision()` 保留在 `rl.rollout.contracts`，但签名从接收 `ExplorationPolicyContext + action + logprob + value` 改成接收 planning result / policy decision，内部构造 PPO TensorDict。`HostTransfer.defer()` 中 RL 独有的 scene tokens、old logprob、state value、RNG audit 等字段也继续留在 RL adapter。也就是说 planning 不知道 `PPO_BATCH_KEYS`、`TRAINING_KEYS`、`RolloutEpisode`。

10. **evaluation 改成 DecisionResult → trace adapter。** `PolicyCheckpointEvaluationAgent.runtime` 不再是 `FabricRolloutRuntime`，而是 planning-owned policy-guidance inference。`evaluation/engine.py` 改为从 `planning.policy.checkpoint` 加载 frozen policy。evaluation 自己决定哪些 decision 字段写 trace；planning 不返回 evaluation TensorDict。做完这一步后，可以加一个明确的 dependency test：`eco_planner.evaluation` 不允许 import `eco_planner.rl`。

11. **处理 bootstrap，但不要把它误并入共享 decision。** 当前 `FabricRolloutRuntime.bootstrap_value_batch()` 会生成 diffusion noise、执行 reference preparation，然后只算 critic value。这个流程是 RL credit/bootstrap 的消费者，不是正式 action decision。Task A 可以让它复用 `planning.diffusion.prepare_prediction()` 和 `planning.policy.inputs.build_policy_inputs()`，但“什么时候 bootstrap、使用哪个 generator、是否保存/恢复 generator state”仍然归 RL。不要为了消除几十行代码，把 bootstrap 塞进 evaluation/shared runner。

12. **最后删除旧 runtime ownership。** 当 training、evaluation、experiments 和 benchmark 都已迁移后，再删除 `evaluation/inference/runtime.py` 中的模型执行职责以及 `rl/rollout/runtime.py` 中的决策职责。RL 可以保留很薄的 collection runtime/facade，但它只负责 training-specific generator lifecycle、profiling/audit adapter 和 bootstrap，不再拥有“planner 如何做一次 learned-guidance decision”。

---

## 5. `DecisionResult` 建议不要做成 TensorDict

这是这次重构里比较重要的一点。

当前 TensorDict 同时承担 policy interface、PPO batch 和 artifact storage，很容易重新产生边界泄漏。planning 的公共结果更适合 typed dataclass，例如概念上：

```python
@dataclass(frozen=True)
class PolicyDecision:
    inputs: PolicyInputs
    output: GuidancePolicyOutput
    action: GuidanceAction


@dataclass(frozen=True)
class DecisionResult:
    prediction: torch.Tensor
    reference_prediction: torch.Tensor | None
    policy: PolicyDecision | None
    guidance_diagnostics: GuidanceDiagnostics | None
```

其中 `GuidanceAction` 可以保留：

```text
base_action
guidance_action
joint_log_prob
```

这样 RL 可以构造：

```text
PolicyInputs
+ guidance_action
+ joint_log_prob
+ state_value
→ training TensorDict
```

evaluation 则构造：

```text
prediction
reference_prediction
guidance_action
guidance diagnostics
→ trace
```

两边都不要求 planning 知道消费者 schema。

---

## 6. `policy.inputs` 的最小正确实现

Task A 和 Task B 有一点交叉，但建议 A 只完成“ownership”，不要提前完成全部 representation research framework。

当前：

```python
PlannerPolicyContext(
    scene_tokens=features["scene_tokens"],
    scene_padding_mask=...,
    navigation_tokens=...,
    navigation_padding_mask=...,
    reference_trajectory=reference_prediction[:, 0],
)
```

应变成类似：

```python
representations = prepared.representations

policy_inputs = build_policy_inputs(
    representations=representations,
    reference_prediction=prepared.reference_prediction,
    config=policy_input_config,
)
```

A 阶段的 config 甚至可以只有一个固定 profile，语义完全等价于当前输入。

这样 B 阶段加入：

```text
reference masked
navigation removed
different reference encoder
actor/critic distinct inputs
```

时，只需要扩展 `planning.policy.inputs` 和 policy config，而无需修改 diffusion 或 inference 的 reference generation。

尤其注意：

> “policy 不读取 reference” ≠ “reference pass 不执行”。

因为 reference 同时是 guidance 本身的物理基准，而且共享 reference/guided DDIM randomness 是当前算法契约。

---

## 7. RNG 是此次重构最容易出错的地方

当前 learned 路径的顺序实际是：

```text
capture diffusion RNG state
capture policy RNG state
↓
draw initial diffusion noise
↓
prepare_policy_guidance
    ├─ prepare DDIM transition randomness
    └─ reference DDIM sampling
↓
policy forward
↓
policy action sample   # mean 模式无消费
↓
guided DDIM sampling
```

这个顺序应该视为 `planning.inference` 的 observable semantics。

不能做的事情包括：

```text
先 sample policy action 再跑 reference
给 reference/guided 分别重新准备 transition randomness
mean policy 仍调用 sampler 后再取 mean
为了抽象方便重新生成 initial noise
base/fixed/learned 强行走同一随机调用序列
```

这些都会让“纯架构重构”变成实验语义变化。

特别是当前 learned completion 使用：

```python
with torch.enable_grad():
    planner.complete_policy_guidance(...)
```

这是因为 guidance gradient 本身需要 autograd，哪怕 planner 权重是 frozen。迁移时不能顺手改成 `torch.inference_mode()`。

---

## 8. 文件级迁移建议

| 当前文件                                            | Task A 后                                                                 |
| --------------------------------------------------- | ------------------------------------------------------------------------- |
| `models/*`                                          | `planning/diffusion/*`                                                    |
| `rl/policy/config.py`                               | `planning/policy/config.py`                                               |
| `rl/policy/model.py`                                | `planning/policy/model.py`，之后 B 再拆 encoder/fusion                    |
| `rl/policy/distribution.py`                         | `planning/policy/distribution.py`                                         |
| policy-only checkpoint functions                    | `planning/policy/checkpoint.py`                                           |
| `rl/rollout/runtime.py::_decide_batch`              | `planning/inference.py`                                                   |
| `rl/rollout/runtime.py::bootstrap_value_batch`      | 保留 RL orchestration，但调用 planning primitives                         |
| `rl/rollout/contracts.py::build_training_decision`  | 保留 RL                                                                   |
| `evaluation/inference/runtime.py` model execution   | `planning/inference.py`                                                   |
| `evaluation/inference/decision.py` trace/audit 映射 | evaluation 保留；通用 planning result validation 可下沉                   |
| `evaluation/inference/agent.py`                     | 保留薄 adapter，依赖 planning                                             |
| `evaluation/engine.py` policy checkpoint loading    | 改依赖 `planning.policy`                                                  |
| `benchmarking/rollout.py`                           | 调用正式 planning decision 或正式 RL collection，不实现 planner execution |

---

## 9. 建议拆成 5 个可 review 的 PR/commit slice

| Slice | 范围                                                        | 关键验收                                                  |
| ----- | ----------------------------------------------------------- | --------------------------------------------------------- |
| A1    | characterization tests + `models → planning.diffusion`      | planner 数值/RNG 完全不变                                 |
| A2    | `rl.policy → planning.policy` + policy checkpoint ownership | evaluation config/checkpoint 不再依赖 RL policy           |
| A3    | neutral representations + `policy.inputs`                   | 当前五项输入逐值一致                                      |
| A4    | shared `planning.inference` + `DecisionResult`              | rollout/evaluation 对同 observation/RNG 得到相同 decision |
| A5    | rewire RL/evaluation/benchmark，删除旧执行 runtime          | evaluation 无 `eco_planner.rl` 运行时依赖                 |

这样比一个大 PR 同时移动目录、重写 policy、改 evaluation 和改 RNG 安全得多。

---

## 10. Task A 完成时最值得设置的验收测试

我建议重点不是测试“新路径能跑”，而是设置跨消费者 contract tests。

| 测试                          | 应验证                                                                 |
| ----------------------------- | ---------------------------------------------------------------------- |
| learned decision parity       | ref trajectory、action、log-prob、value、final trajectory 与重构前一致 |
| RNG parity                    | diffusion/policy generator before/after state 一致                     |
| deterministic mean            | policy RNG state 完全不变                                              |
| training/eval shared decision | 同 observation/checkpoint/RNG 下两者 action/trajectory 一致            |
| neutral representation        | diffusion output 不包含 `PolicyContext` 类型                           |
| policy input adapter          | 当前 input tensors 与旧手工 mapping 逐值一致                           |
| reference preservation        | 即使 policy input profile 屏蔽 reference，reference pass 仍执行        |
| dependency test               | `evaluation` 不 import `rl.rollout`、`rl.optimization`、`rl.policy`    |
| result ownership              | planning 不 import `RolloutEpisode` / PPO batch / evaluation artifacts |
| frozen planner                | decision + PPO update 后 planner hash 和 `.grad` 状态满足原契约        |

最后一个架构判断是：**Task A 不应该顺手完成 Task B。** A 需要建立 `planning.policy.inputs` 这个 owner，但当前输入可以先完全等价；encoder/fusion/actor-critic 的正式拆分放到 B。否则一次改动同时改变 package ownership、feature semantics、network graph 和 checkpoint schema，出现数值差异后很难定位。

如果按这个方案推进，Task A 的真正完成标志不是“出现了 `planning/` 目录”，而是这条依赖链成立：

```text
RL rollout ──────┐
evaluation ──────┼──> planning.inference
benchmark ───────┘         │
                            ├── planning.diffusion
                            └── planning.policy

planning ──X──> rl
planning ──X──> evaluation
evaluation ──X──> rl.rollout / PPO
diffusion ──X──> policy schema
```

这才会真正解决 Issue #96 中 Task A 所针对的耦合。
