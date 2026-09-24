---
kind: contract
status: proposed
scope: [policy-abi, rollout, rng, bootstrap, resume]
read_when:
  - changing policy probability accounting or rollout boundaries
  - diagnosing PPO RNG checkpoint or resume behavior
---

# Training contract

这是 [Issue #102](https://github.com/xcz0/Eco-AutoDrive/issues/102) 的 proposed 草案，旧入口未切换。
MUST／MUST NOT 表示拟保留的软件保证。优化／reward 方法由
[training protocol](../research/protocols/training.md) 拥有；执行／配置边界见 [execution](execution.md)，
持久化载体及 tracking 身份见 [artifacts](artifacts.md)。

## Policy context 与输出

输入 MUST 为冻结 scene tokens `[B,N,H]` 与 bool padding mask、冻结 navigation token `[B,1,H]`
与 bool validity/padding mask，以及 ego-local physical reference `[B,80,4]`。
Reference 使用 data/model ABI 的时间、米和 heading 编码。Features 同 batch、dtype、device 且有限，
每个 item 至少有一个有效 context token。完整数值／token 有效性校验在 CPU collection 边界或
显式 debug 时执行，受控热路径保留结构检查，不重复同层校验。

Context MUST detach；planner eval/frozen、参数不被更新或获得 `.grad`。
Policy 输出严格正的 `alpha,beta [B,2]` 和 `state_value [B,1]`；typed collection value 可为 `[B]`。
Rollout、current-policy recomputation、bootstrap 和 probe 使用一致的 key/shape 含义。
Compile 是显式配置选择，不改变 sampler、guidance、随机流、planner hash 或 state-dict identity；
所需 backend/graph capture/执行失败必须报错，不静默切回 eager。

## 动作与概率记账

Base action `u` MUST 严格位于 `(0,1)^2`，保存供 replay；guidance `g=2u−1` MUST 严格位于
`(-1,1)^2`，顺序为 lateral、longitudinal。边界值、非有限参数和错误 shape/dtype/device 立即失败，
不得 clamp。固定人工干预的闭区间接口仅见 [data/model](data-and-model.md#固定干预动作)。

两个独立 Beta 维度的记账为：

```text
log_prob_u = Σi log Beta(ui; alpha_i,beta_i)
log_prob_g = log_prob_u − 2 log(2)
entropy_g  = Σi H(Beta_i) + 2 log(2)
```

PPO MUST 保存并使用 transformed old joint log-prob，与 current `log_prob_g` 同概率空间。
PPO batch 中每 transition 一个 scalar，形状 `[T]`，不得保留会造成广播的尾部 singleton。
采集 audit 中 `[B,1]` 的 log-prob/value 需在该边界正确投影。
Diffusion transition probability 不参与 PPO ratio；K=1 不引入候选选择概率。
Mean action 不消费 policy RNG；sample/rsample 只消费调用者给定的独立 policy generator，不改变全局 RNG。

## 逻辑采集与 episode

逻辑 scenario 数与 transitions_per_environment 决定 PPO batch；物理 worker 数只决定并发容量。
MUST 按确定性顺序复用物理 slots，保持每个逻辑 slot 的场景、generator、seed 与 episode boundary。
每 update 从全部逻辑 scenarios 收集完整配额；不能因机器容量改变样本集合或其拼接顺序。
拼接顺序为 scenario、episode、transition。Training pool 可复用，但换 episode 只 reset 对应环境。

一个 transition 的 context/action/reference/noise、实际 prefix、reward、flags、next state 必须对齐。
正式 cadence 引用 planning/evaluation protocol；terminal/truncation 时记录短前缀。
Root training fields 保存 context、guidance、old log-prob 和 current value；`next` 保存 scalar reward、
done、terminated、truncated 与 next value。Base action、RNG 等 replay 事实在 audit 中保留。
内部 next value 来自后一 decision；尾部来自显式 bootstrap。原始 episode training/audit 不被 GAE 改写。
训练采集内部错误直接终止，不把 partial trajectory 伪装成正式 batch；诊断部分证据另有契约。

## Episode 与 bootstrap

Bootstrap mask `b = not terminated`。真实环境 `done = terminated or truncated`；
GAE 的 recursion boundary 还包括 collector rollout-limit tail，故 episode 最后一项始终停止递归。
Collector tail 不得被写成虚假的环境 terminal；原始 termination/truncation 与 tail kind 分开保留。

| 尾部情况 | 原始 terminated / truncated | Tail value | GAE 能否跨尾继续 |
| --- | --- | --- | --- |
| Terminal | true / false | 0 | 否 |
| 同时 terminal 与 truncation | true / true，两个 flags 都保留 | 0，tail kind 为 terminated | 否 |
| 纯 truncation | false / true | 最终状态 critic value | 否 |
| Rollout-limit | false / false | 最终状态 critic value | 否 |

Pure truncation 与 rollout-limit 可 bootstrap，但 advantage 不跨 episode/collector boundary 泄漏。
Tail critic 使用对应最终状态与冻结 context；终止优先级不能由后一个 bool 覆盖。
Per-transition gamma 与 GAE 公式见 training protocol，本篇不另定义时间折扣。

## Reward 与 audit 对齐

Collector 使用同一 execution facts 求每个实际子步 reward，再按 training protocol 归约。
同一个 `reward_total` MUST 写入 PPO `next.reward` 与 audit，PPO 不消费 worker 零 reward 占位。
离线 reweight/rescore 逐子步复用同一 reward 数学与归约，不从 transition 的 min gate 和 component sum
反推 scalar reward，也不维护第二套公式。

Audit MUST 保存逐子步 component scores、gate、progress/comfort 运动量、proxy step mL、distance、
distance-valid 与 `reward_substep_count`。数组容量为 canonical prefix，terminal 后无效 suffix
可按 schema 补零／False，但 count 明确有效前缀；这些 padding 不是已执行的零 reward 样本。
统计／重评分只读取有效子步。Transition domain aggregation 按 training protocol，行为 bool 从
domain facts 消费，不从状态／crash flags 再派生一次。

Fixed-batch reward/credit MUST 保持 sample identity、scenario 顺序与初始 policy 对齐。
每 transition 的 substep_count 与各 arm component sums/min gate 长度等于 sample 数；per-substep
原始事实归源 audit，不能以汇总数组替代缺失 audit。完整 schema 与 reader validation 归 artifacts。

## PPO batch 与随机流

GAE 完成后，update batch 只保留 context、guidance action、old transformed log-prob、advantage
和 value target；collection value、reward 与 boundary 不混入 loss 输入。
Batch 在 policy device 上做 GAE/normalization，training fields 保持 detached 直至 update。
Actor 与 critic 使用同一 policy 参数所有者和同次 context，不维护独立网络副本。

| 随机命名空间 | 生命周期与保证 |
| --- | --- |
| Map | 由显式 scenario 指定，不借用 action/noise 流。 |
| Training diffusion / action | 固定 SeedSequence namespace 从 training seed 派生；每逻辑 slot 各一条持久流，跨 episodes 与 updates 延续。 |
| Minibatch | 独立 seed 初始化，位于 policy/storage device，跨 updates 延续，不消费全局 RNG。 |
| CPU/CUDA global RNG | 独立保存／恢复，不以重设 training seed 替代实际状态。 |

同设备、同配置下顺序可复现；不承诺 CPU/CUDA 或不同精度排列／结果逐位相同。
每 update 仅拥有当前 batch 的无放回 sampler；正常结束或 KL 早停后丢弃未消费 permutation，
下一 update 从新 batch 采样，不重置 generator。Scheduler 与 optimizer step count 按实际更新，
累计 early-stop 次数保留。Ratio/probe/可选 gradient diagnostics 不能消费 action/minibatch 流或
改变最终 total backward；不同 evaluated-minibatch／optimizer-step 数量不能混为一个计数。

## 精确续训范围

精确恢复 MUST 限定在相同实验与执行配置的完整 update 边界；不承诺 mid-update simulator 恢复。
下一次 collect 重新 reset scenarios，但逐逻辑 slot diffusion/action RNG 必须从保存状态继续，
不能从 seed 重启。训练前后冻结 planner 参数 hash 必须相同。

Training checkpoint 必须恢复 policy、optimizer、scheduler、optimizer-step count、累计 KL early-stop、
minibatch RNG、CPU/CUDA RNG、completed-update loop state，以及按逻辑 scenario 顺序保存的
diffusion/action generator uint8 states。MUST 校验状态数量、类型及内嵌 seed 与逻辑 slot 对应关系，
并在下一 collect 前恢复。缺少任一两组 slot RNG state 或旧 minibatch_sampler_state 明确拒绝，
不提供旧 sampler state 迁移。

Policy-only export 只表示可加载策略参数，不能用它宣称精确续训。
同 tracking Run 允许改变 resources 等 invocation 参数是索引写入规则，
不扩大本节的精确恢复范围，见 [tracking 持久化](artifacts.md#tracking-身份与续写)。

## 来源、冲突与代码导航

来源：[旧 training contract](../agents/contracts/training.md)、[旧 experiments](../agents/contracts/experiments.md)；
接受依据：[ADR 0016](../adr/0016-add-forward-only-exploration-policy.md)、
[0018](../adr/0018-use-torchrl-for-gae-and-ppo-math.md)、[0039](../adr/0039-unify-closed-loop-cadence.md)。
ADR 0018 已将原闭区间表述标明为与 ADR 0016 冲突的历史文字问题，并引用本篇动作定义。
草案保留 ADR 0016 的严格开区间及端点失败决定；这不是根据当前实现新增或放宽规范。

| 任务 | 实现定位 | 相关测试（未运行） |
| --- | --- | --- |
| 动作空间／context | [distribution](../../src/eco_planner/planning/policy/distribution.py)、[inputs](../../src/eco_planner/planning/policy/inputs.py) | [policy inputs](../../tests/planning/test_policy_inputs.py)、[PPO](../../tests/training/test_ppo.py) |
| RNG / resume | [seeds](../../src/eco_planner/rl/rollout/seeds.py)、[training state](../../src/eco_planner/rl/training_state.py)、[checkpoint](../../src/eco_planner/rl/optimization/checkpoint.py) | [PPO](../../tests/training/test_ppo.py)、[rollout](../../tests/training/test_rollout.py) |
| Episode / bootstrap | [rollout contracts](../../src/eco_planner/rl/rollout/contracts.py)、[collector](../../src/eco_planner/rl/rollout/collector.py) | [rollout](../../tests/training/test_rollout.py)、[execution consistency](../../tests/simulation/test_execution_consistency.py) |
| Reward parity / fixed batch | [aggregation](../../src/eco_planner/reward/aggregation.py)、[RL reward](../../src/eco_planner/rl/reward.py) | [reward](../../tests/training/test_reward.py)、[fixed batch](../../tests/training/test_fixed_batch.py)、[credit](../../tests/training/test_credit_assignment.py) |
| Tracking 恢复身份 | [tracking](../../src/eco_planner/rl/tracking.py) | [tracking](../../tests/training/test_tracking.py) |
