---
kind: protocol
status: active
scope: [fixed-batch, attribution, intervention, measurement]
read_when:
  - designing fixed-batch credit or guidance diagnostics
  - interpreting causal interventions or performance measurements
---

# Diagnostic studies protocol

MUST／MUST NOT 表示强制要求。本篇记录已声明诊断设计，不规定每次任务运行全部实验，
也不把有实现／有配置视为已运行或已通过。正式 baseline 见
[planning/evaluation](planning-and-evaluation.md)，reward/GAE 定义见 [training](training.md)。
研究阈值、干预轴和对比方向 MUST 预先显式配置，不能看结果后改 gate 或挑 seed。

## 固定批次与校准

采集 MUST 使用 initial policy、update 0、无 optimizer/scheduler step，保存完整 batch 与初始策略。
所有 reward/credit 条件共享 transition 顺序、动作、log-prob、value/next-value 与 boundary。
Reward 重算只变 score／权重，不恢复 actor 或 backward；credit 从同源 batch 独立重建条件，
不依赖前序诊断输出目录。数值归约引用 training protocol，audit 对齐归 contracts。

Progress 校准用完整 batch 正向 delta 的中位数除以 target score。Comfort 仅对存在零分的子项用
`max(original_limit, P50(abs(metric))/(2-target_score))`，保留分段线性和四项取最小值。
启动 transition 不删除；未失活子项、Energy/TTC/Speed/Safety 不变。
校准 Comfort 只表示此运动学分布的相对平顺性，不重定义物理限值。

Energy 表示有两种不同设计：

- Batch-derived band：由本 batch 的显式分位数推导阈值，不核验它等于历史冻结值。
- Frozen band：直接使用显式冻结阈值，不从当前 batch 重新估计；只切换 energy 表示，
  Progress/Comfort 校准仍由源 batch 的 calibrated profile 承担。

两者互斥；`calibration`、`energy_band`、`frozen_energy_band` 是必填可空字段，不隐式选择设计。
MUST 保存实际配置和尺度、原始有符号运动量、评分绝对值、原 limit 超限率、零／满分率、
包含并列的最小项归因，以及 scenario/cycle 数组。Energy-only 是逐子步 `gate × energy score`
再聚合的诊断 endpoint，不据此新增训练 profile。

## Credit attribution

配置 MUST 显式列出 reward arms、advantage form 与 credit form：

| 轴 | 定义 |
| --- | --- |
| `standard_gae` | 保留原 current/next critic value，复用训练 GAE。 |
| `reward_only_gae` | Current/next value（含 tail）同时置零，保留 gamma、lambda 与 boundary。 |
| `discounted_return` | 逐 episode 反向递推，gamma 折扣、lambda=1，无 critic/bootstrap；不等同 reward-only GAE 的 gamma×lambda 衰减。 |
| `raw` / `center` / `z` | 原样／只减 full-batch mean／复用训练 sample-std normalization。 |

MUST 仅对 actor `loss_objective` 做 backward，不混入 critic/entropy 梯度，不 clipping，
optimizer/scheduler steps 均为零。梯度按 actor head、shared trunk、lateral/longitudinal head 行分组。
统计用 float64；GAE/backward 保持训练 dtype/device。零初始化 actor head 阻断 initial trunk actor
梯度，不能把零 norm cosine 当同方向；零分母 ratio 为 null。
固定批次归因不证明训练后行为变化；gate 只适用于声明的实验条件。

分布统计中 advantage std 为 `ddof=1`；value target 使用持久化 `value_target_ddof`
（已有 ablation 配置为 1，其他命名配置为 0），其余分布为 `ddof=0`。
Pearson/Spearman 使用含 ties 的既定统计定义，常量或不足样本为 undefined。
报告保留 cosine、`1−cosine`、norm ratio 原值和 undefined，不裁剪以美化结果。

## 已退役的 guidance 诊断

Authority、execution horizon、replanning deferral、longitudinal/lateral decomposition 与
frozen-policy execution bridge 已退役，其专用实现、配置、CLI 和报告重算入口不再维护。
历史方法与实现见[退役前版本](https://github.com/xcz0/Eco-AutoDrive/tree/7a3ca7366e9f699552ce73e2f4c305a862dbb754)；
对应实验记录与原始证据继续保留，历史结论的适用条件仍以具体 record 为准。
本次退役不改变正式训练、评测 cadence 或保留的 energy sweep。

## Training grid、测量与 critic attribution

Grid 显式声明 learning rate × epochs × gradient norm、update budget 与诊断阈值。
先检查更新量、KL、ratio、probe/Beta 与行为，再对通过项做 matched initial/final held-out 测量。
通过候选依次按最低 learning rate、epochs、gradient norm 选择，不按训练 return 排名。
无候选为 selected_config=null；异常保留 partial summary／原始证据并传播，不把未完成网格称成功。

Post-update 测量显式声明 training runs/seeds、MC draws/seed，区分训练 loss 中的 pre-update
approximate KL 与同 batch 的 post-update seeded-MC KL，保留参数 delta、ratio 与 probe 变化。
Post-update 同时报告 analytic Beta `KL(old || new)` 与 seeded-MC/k3，不能把单 minibatch
训练前的 `kl_approx≈0` 当成没有更新。MC log ratio 在 base coordinates 以 float64 求值，
利用 affine Jacobian 抵消，避免 guidance 往返转换把近边界 draw 舍入为端点；不 clip draws，
真实非有限结果仍保留为失败。Measurement-only gradient norms 与 learning-rate exposure
的持久化/核对保证见 [artifacts contract](../../contracts/artifacts.md#matched-inputs-与离线读取)。
Mean/sample 评测固定 checkpoint hash/label、场景、horizon、sampler 和 diffusion seed，另声明
policy action seeds；所有 seed 和缺项保留。可复用 mean 结果必须满足上述条件，不按目录猜测。

Training critic attribution 是无仿真／无 planner 的离线 backward-only 设计。
对每个 update 用保存 rollout 与对应 pre-update policy（update 0 用 initial，其后用前一 update）
重建 standard GAE，与源 summary 的 raw advantage mean/std 在显式容差内核对。
只做 actor full-batch backward，无 clipping／optimizer step，checkpoint hash 前后不变。
比较 credit/advantage forms 与预登记 actor/lateral/longitudinal cosine、sign-flip materiality 阈值。
All-checkpoint 判据失败为 critic_material_candidate，通过为 critic_not_material_to_actor_direction；
“candidate”不是主因结论。Within-arm 对照与 cross-arm unmatched 描述量分开，后者不作因果。

## Cadence 与 PPO control 归因

#105 的 Phase A/B/C 已完成并退出当前维护范围，cadence attribution、counterfactual
attribution、control re-freeze 的 run/analyze 入口及专属配置已移除。
历史设计、参数和裁定见 [E-050](../../experiments/records/e-050-issue105-phase-a-cadence-attribution.md)、
[E-051](../../experiments/records/e-051-issue105-phase-b-counterfactuals.md)、
[E-052](../../experiments/records/e-052-issue105-phase-c-canonical-ppo-refreeze.md)。
当时的实现和配置保存在 Git revision `5445f54569bea807b902172f4b7e46441baf9bc3` 的
`src/eco_planner/experiments/training/` 与 `configs/experiments/training/`。
这些历史研究规则不再约束当前训练入口；现行优化与 cadence 定义归
[training protocol](training.md) 与 [execution contract](../../contracts/execution.md)。

## Benchmark 的测量语义

规模、warmup、正式样本与 repeats MUST 显式配置，保留原始样本与中位／极值。
Serial/vector 比较使用各自真实 collector、同一 workload，并核验声明 topology 与运行 metadata。
Rollout 显式声明 PPO epochs/minibatch size，scheduler horizon 为
`update_count × epochs × (batch_size/minibatch_size)` 个 optimizer steps。

Host call wall、accelerator phase 及跨 stream 时间不是互斥分解，不能直接相加。
CPU collate 不计入 planner wall；同步 execution copy、异步 audit copy、resolve 剩余 host wait 分开。
Decision 与 bootstrap batch 分开统计；collection residual 只表示未归类 host wall，不能叫
planner/environment overhead。Worker busy 为 environment+observation，
`transport_sync=max(0,batch_wall−max(worker_busy))`，另记 busy imbalance；不推断独立 IPC send/receive。
测量结果不直接成为 runtime 默认值；profiling 的无干扰保证见 execution contract。

## 来源、待确认项与导航

迁移来源：[旧 experiments contract](https://github.com/xcz0/Eco-AutoDrive/blob/eaafe5bf911390ef3263bea808aadc40215068a6/docs/agents/contracts/experiments.md)、
[旧 runtime 计时](https://github.com/xcz0/Eco-AutoDrive/blob/eaafe5bf911390ef3263bea808aadc40215068a6/docs/agents/contracts/runtime.md)；机制边界依据
[ADR 0038](../../adr/0038-consolidate-scientific-workflows.md)，正式／诊断 cadence 边界依据
[ADR 0039](../../adr/0039-unify-closed-loop-cadence.md)。具体实验的阈值与判据由对应显式配置持久化，
不将某次 gate 升格为通用科学真值。

本篇仅区分 frozen band 的设计，不据旧计划重做机制，也不把配置存在解释为实验通过。
Transfer 验收归 [Issue #83](https://github.com/xcz0/Eco-AutoDrive/issues/83)；
本分支已登记的 E-048～E-060 证据与各自条件见
[Findings](../findings.md#canonical-cadence-transfer-的证据边界)。#105 设计的迁移来源为
[本分支旧 experiments contract](https://github.com/xcz0/Eco-AutoDrive/blob/f3c3d23/docs/agents/contracts/experiments.md)，
接受与运行依据为 E-050/E-051/E-052；本篇没有重新运行或重裁这些实验。

| 任务 | 实现／配置定位 | 相关测试（未运行） |
| --- | --- | --- |
| 固定批次／credit | [fixed batch](../../../src/eco_planner/rl/rollout/fixed_batch.py)、[credit](../../../src/eco_planner/experiments/credit/)、[configs](../../../configs/experiments/credit/) | [fixed batch](../../../tests/training/test_fixed_batch.py)、[credit](../../../tests/training/test_credit_assignment.py) |
| Offline attribution / reporting | [credit math](../../../src/eco_planner/rl/optimization/credit.py)、[experiments](../../../src/eco_planner/experiments/) | [credit](../../../tests/training/test_credit_assignment.py)、[reports](../../../tests/analysis/test_reports.py) |
| Benchmark | [benchmarks](../../../benchmarks/)、[configs](../../../configs/components/benchmark/) | [rollout benchmark](../../../tests/benchmarking/test_rollout.py) |
