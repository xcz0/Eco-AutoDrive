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

## 人工 guidance 的共同控制条件

Matched group MUST 固定 scenario、reset seed、slot、batch shape 与 diffusion noise seed。
各 arm 独立 reset，核对首 observation、simulator state 和逐周期 initial noise；DDIM stochasticity=0。
固定 action 是 `(lateral,longitudinal)`，允许 ±1，直接干预 guidance，不经过 Beta 概率空间。

Terminal 后不得继续 step 或补造执行子步；为了匹配可保持 inference batch/noise draw 次数，
但终止后的 planner audit 不能进入执行指标。异常保存当前 arm 已有部分证据并传播。
安全、完整性、执行窗口与 planner-response 的差异 MUST 单列；gate 不通过不等于软件执行失败。
Proxy intensity 使用实际窗口 fuel/distance，零距离 undefined；零噪声不能使零效应通过。

### Control authority

该设计显式以 `execution_steps=1` 进行 0.1 s 诊断，首步 immediate 与默认 20 子步／2 s 窗口
分开解释，env horizon 大于窗口。普通 evaluation 不继承该 override。
测量实际运动、终止、reference/guided prediction、guidance diagnostics 与 noise。
对场景内 arm repeat 均值计算 Spearman、±1 endpoint 差及重复噪声；阈值和同方向场景数由配置给出。
常量相关性 undefined，安全／完整性失败与时间窗口方向冲突分别报告。

### Execution horizon

固定同一物理窗口（既有设计 2 s），仅改变每次重规划前执行的 waypoint 数 k。
MUST 要求 k 整除 total_window_steps 且小于 prediction horizon；replan 数为 total_window_steps/k。
首次完整预测在 0.1/0.2/0.5/1/2/4/8 s 记录前向位移并跨 k 核对匹配。

按 k 分组统计 speed、endpoint speed、distance、progress、energy intensity、tracking error 的
逐场景 Spearman、endpoint 差与噪声；planner response 为 first-plan matched 位移差。
预声明判据区分 receding-horizon mismatch 强证据、符号未变、混合／不确定与 safety/proxy failure。
此干预不选择或重定义正式 cadence。

### Replanning deferral

固定 `execution_steps=1`、默认 2 s 窗口与 5 个配置 longitudinal arms，每周期重新规划；
只加载冻结 planner，记录每周期全部未来 waypoints。
以 `Δ=D(g=+1)−D(g=−1)` 先对 noise repeats 取中位，再逐 cycle 记录首点、完整预测时域效应、
首个正 response 的 zero-crossing waypoint，以及跨 cycle 稳定性。

多数 cycle 首点 ≤0、full-horizon >0、正 response 在执行 prefix 外且 crossing 稳定，是场景级
repeated-deferral 判据；场景多数满足则判 repeated_deferral。首点多数为正为 deferral_absent，
否则 mixed/inconclusive；安全、完整性或 proxy 失败覆盖为 safety_or_proxy_failure。
它定位预测效应是否反复落在未执行部分，不证明某个训练 reward 有效。

### Longitudinal / lateral decomposition

按 training seed 配置 `r0/lon/lat/joint` 四个常量动作。Lon 仅改纵向，lat 仅改横向，
`joint = r0 + (lon−r0) + (lat−r0)`，值域遵守固定动作契约。
既有设计的常量来自 E-041 frozen final-policy Beta mean；只加载 planner，诊断 prefix 为 1，
episode 可变长。组内 arm 顺序固定，以 scenario/seed/noise 配对。

相对 r0 的 effect 记为 `E_arm`，interaction 为 `E_joint−E_lon−E_lat`。
方向按多数场景门槛，归因按预声明 dominance_share 与 expected_joint_direction 区分
longitudinal-dominated、lateral-dominated、nonlinear-interaction、not-reproduced-state-dependent。
Seed 不一致时 overall 为 mixed-across-seeds。常量干预不是完整 state-dependent policy 的替代证据。

### Frozen-policy execution bridge

冻结 learned policy checkpoints，以 deterministic mean action 替代常量；保持 matched reset、
固定 batch 与逐周期 noise，核对 policy hash 未变，不训练、不修改 planner/reward/PPO。
既有设计取 E-040 r0/rstress、training seeds {0,1}，明确分两部分：

- Part A：按 prefix k（既有 1/2/5，须整除 evaluated horizon）闭环比较 `Rstress−R0`，
  first-plan response 跨 prefix 必须匹配。保留执行／安全／长度／误差与逐周期 action。
  两个 seed 均出现 k=1 negative、k=5 positive 才符合 crossover；其他结果按原配置记 amplify
  或 no-material。k=5 是历史 held-out 的方向复现对照，不宣称不同 collector 的逐值复现。
- Part B：在同一 held-out context，用两个 policy 对同 observation、planner、noise 各求一次
  response，只替换 guidance。记录 0.1/0.2/0.5/1/2/8 s 的 forward/lateral effect、Δg 和
  zero-crossing。多数 context Δg_lon>0、0.1 s 中位≤0、0.2/0.5 s 中位>0 是该设计的判据；
  若 0.1 s 已为正则为 local-response-differs，不能将 ±1 全幅 sweep 等同 learned operating point。

最终 verdict 按持久化设计的四个预声明结论选择；本篇不重裁任何历史 gate。

## Training grid、测量与 critic attribution

Grid 显式声明 learning rate × epochs × gradient norm、update budget 与诊断阈值。
先检查更新量、KL、ratio、probe/Beta 与行为，再对通过项做 matched initial/final held-out 测量。
通过候选依次按最低 learning rate、epochs、gradient norm 选择，不按训练 return 排名。
无候选为 selected_config=null；异常保留 partial summary／原始证据并传播，不把未完成网格称成功。

Post-update 测量显式声明 training runs/seeds、MC draws/seed，区分训练 loss 中的 pre-update
approximate KL 与同 batch 的 post-update seeded-MC KL，保留参数 delta、ratio 与 probe 变化。
Mean/sample 评测固定 checkpoint hash/label、场景、horizon、sampler 和 diffusion seed，另声明
policy action seeds；所有 seed 和缺项保留。可复用 mean 结果必须满足上述条件，不按目录猜测。

Training critic attribution 是无仿真／无 planner 的离线 backward-only 设计。
对每个 update 用保存 rollout 与对应 pre-update policy（update 0 用 initial，其后用前一 update）
重建 standard GAE，与源 summary 的 raw advantage mean/std 在显式容差内核对。
只做 actor full-batch backward，无 clipping／optimizer step，checkpoint hash 前后不变。
比较 credit/advantage forms 与预登记 actor/lateral/longitudinal cosine、sign-flip materiality 阈值。
All-checkpoint 判据失败为 critic_material_candidate，通过为 critic_not_material_to_actor_direction；
“candidate”不是主因结论。Within-arm 对照与 cross-arm unmatched 描述量分开，后者不作因果。

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
旧计划与新增评论的状态差异见 [Findings 的核验边界](../findings.md#canonical-cadence-transfer-的证据边界)。

| 任务 | 实现／配置定位 | 相关测试（未运行） |
| --- | --- | --- |
| 固定批次／credit | [fixed batch](../../../src/eco_planner/rl/rollout/fixed_batch.py)、[credit](../../../src/eco_planner/experiments/credit/)、[configs](../../../configs/experiments/credit/) | [fixed batch](../../../tests/training/test_fixed_batch.py)、[credit](../../../tests/training/test_credit_assignment.py) |
| Authority / horizon | [intervention](../../../src/eco_planner/evaluation/intervention.py)、[guidance configs](../../../configs/experiments/guidance/) | [authority](../../../tests/training/test_guidance_control_authority.py)、[horizon](../../../tests/training/test_guidance_horizon.py) |
| Deferral / decomposition | [guidance studies](../../../src/eco_planner/experiments/guidance/) | [deferral](../../../tests/training/test_guidance_deferral.py)、[decomposition](../../../tests/training/test_guidance_decomposition.py) |
| Frozen-policy bridge | [policy intervention](../../../src/eco_planner/evaluation/policy_intervention.py) | [bridge](../../../tests/training/test_guidance_execution_bridge.py) |
| Offline attribution / reporting | [credit math](../../../src/eco_planner/rl/optimization/credit.py)、[analysis](../../../src/eco_planner/analysis/) | [credit](../../../tests/training/test_credit_assignment.py)、[reports](../../../tests/analysis/test_reports.py) |
| Benchmark | [benchmarking](../../../src/eco_planner/benchmarking/)、[configs](../../../configs/components/benchmark/) | [rollout benchmark](../../../tests/benchmarking/test_rollout.py) |
