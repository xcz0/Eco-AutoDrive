---
kind: protocol
status: active
scope: [policy, reward, gae, ppo]
read_when:
  - changing policy optimization action sampling or reward
  - reviewing training and evaluation research conditions
---

# Training protocol

MUST／MUST NOT 表示强制要求。本篇拥有优化方法、reward 和统计定义；
动作 ABI、RNG、bootstrap 和 resume 保证归 [training contract](../../contracts/training.md)，
正式 cadence 与评测设计归 [planning/evaluation](planning-and-evaluation.md)。

## 优化对象与动作方法

MUST 只优化 Exploration Policy 的 actor head、value head 和共享 trunk；官方 EMA planner
保持冻结、eval mode，planner context 进入 policy 前 detach。策略消费 scene/navigation context
与 physical reference，具体 ABI 见 training contract。网络规模、dropout、concentration 等研究参数
必须显式配置，不用隐式默认补足。

每个 planning cycle 是单候选 `K=1`。横向和纵向分别使用独立 Beta 分布，concentration 为
`softplus(raw) + minimum_concentration`。Actor head 零权重、对称 bias 初始化，使 `alpha=beta`，
初始 guidance mean 为零而方差非零。训练使用显式 policy RNG 的 reparameterized sample；
deterministic evaluation 使用 Beta mean，随机策略评测使用 sample，不公开 mode。
动作变换、有效域及 Jacobian 由 training contract 唯一定义，不采用 DDIM transition probability。

Collection 与 optimization MUST 保持策略网络的确定性模式，避免 dropout 等网络随机性污染
old/new probability ratio；优化阶段仍记录梯度。训练、固定批次与评测共享冻结 planner／guidance
定义，正式训练与评测共享 cadence；episode 编排、RNG 生命周期和产物边界不因 cadence 相同而合并。

## Reward 定义

PlannerRFT-style MetaDrive reward 全部是 smoke-only adaptation，不等于 nuPlan scorer parity、
真实车辆舒适性或已验证节能目标。每个实际执行子步先计算客观事实，再求 reward。
设 `T,P,C,S,E` 分别为 TTC、Progress、Comfort、Speed、Energy score，`G` 为 safety gate：

```text
B0 = (5T + 5P + 2C + 4S) / 16
Bλ = (5T + 5P + 2C + 4S + λE) / (16 + λ), λ > 0
rj = Gj × Bj
```

以上为已声明 profile 家族的权重关系；阈值、权重和尺度由 resolved reward profile 显式提供。
`G` 是 collision（profile 选择的 crash 类型）、drivable-area（当前 out_of_road）与 domain
wrong-direction 三项 score 的乘积。Wrong-direction 的几何判定引用 planning/evaluation protocol，
不在 reward 中另设同名阈值。

| Component | 研究含义 |
| --- | --- |
| TTC | Ego-forward corridor 中 constant-velocity closing estimate；threshold/margin 显式配置。 |
| Progress | 当前 route/reference lane 的非负纵向 step delta，按 profile 尺度归一化。 |
| Comfort | 实际执行 acceleration、jerk 与 yaw-rate 的评分；四项取最小值，保留 profile 的分段线性规则。 |
| Speed | 相对当前 lane speed limit 的评分。 |
| Energy | 只使用 execution fuel proxy，native simulator energy 仅供审计。 |

Reward 子步距离小于 `minimum_step_distance_m` 时，denominator-valid=false、mL/km=0、E=0。
这是评分编码，不把 episode 零距离的 undefined 改成零。有效子步强度记为 `e`：

```text
reference_exponential: E = exp(-e / reference_ml_per_km)
calibrated_band:       E = clip((band_zero_score_ml_per_km - e)
                             / (band_zero_score_ml_per_km - band_full_score_ml_per_km), 0, 1)
```

这里的饱和是声明的 reward 定义，不是对模型失败的 repair。

| Profile 家族 | Objective 与解释 |
| --- | --- |
| `plannerrft_energy_v1` | Rλ；energy 权重大于零，分母含该权重。 |
| `plannerrft_no_energy_v1` | R0；仍审计 energy score，但不将其放入 objective。 |
| `plannerrft_no_energy_calibrated_v1` | 使用已冻结 Progress/Comfort 校准的 R0。 |
| `plannerrft_energy_band_lam{1,2,4,8,64}_v1` | 与 calibrated R0 共享校准分量，energy 使用冻结 band；λ=64 是 stress arm，λ=1/2/4/8 是已声明 λ 家族。 |

λ=0 MUST 用 no-energy profile 表达；band profile 名称与 energy 权重必须一致。Calibrated R0 与
band arms 的 objective 差异只在 energy 项、分母及 profile identity，不另改共享分量。
E-034 校准和 E-038 band 的实际冻结数值由
[reward configs](../../../configs/components/reward/) 拥有；诊断中从 batch 重新校准是另一种设计，
见 [diagnostic studies](diagnostic-studies.md#固定批次与校准)。本篇不声称这些历史表示已通过新 cadence transfer。

## 从子步到 transition

正式 execution prefix 使用 planning/evaluation protocol。对实际执行的 `n` 个子步 MUST 使用：

```text
reward_total      = sum_j(Gj × Bj)
reward_base_total = sum_j(Bj)
component_i       = sum_j(component_i,j)
reward_safety_gate = min_j(Gj)
```

PPO 只消费 `reward_total`。一般情况下它不等于 `reward_base_total × reward_safety_gate`。
在线与离线 reweight/rescore MUST 先逐子步重建 objective，再使用同一归约；不能在聚合后对非线性
分量一次缩放。有效前缀及 audit 编码由 training/artifacts contract 保证。

Diagnostics 归约保持下列含义：

| Quantity | 归约 |
| --- | --- |
| Route progress delta、distance、native step mL、execution proxy step mL | Sum |
| Native episode mL | 最后子步值，不能 sum 累计量 |
| Proxy mL/km | `1000 × sum(fuel mL) / sum(distance m)`，不是强度的算术平均；无有效分母保留相应 validity |
| 其他 intensive diagnostics | Mean |
| `has_ttc_candidate` / `energy_distance_valid` | Any / all |
| Collision、drivable、wrong-direction score | Min |
| Transition domain distance | Sum |
| Transition domain speed、position error、heading error | Mean |
| Transition domain stopped、collision、wrong_direction | Any |
| Route completion delta | 覆盖完整实际 prefix |

## GAE 与 PPO

GAE 每个 transition 递推一次；记 `b_t` 为 bootstrap mask，`c_t` 为允许继续递归的 mask：

```text
δt = rt + γ × bt × V(next_t) - V(st)
At = δt + γ × gae_lambda × ct × A(t+1)
value_target_t = At + V(st)
```

Masks、tail value 与原始 terminal/truncated flag 的精确关系由 training contract 拥有。
`gamma` MUST 保持 per-transition，当前配置 0.99；在 canonical cadence 下有效物理折扣视界
约 50 s，旧 0.1 s transition 下约 10 s。依据 ADR 0039，不作 rebase 补偿。

Advantage MUST 仅在完整 PPO batch 上按 sample standard deviation（`ddof=1`）标准化一次；
样本少于两个、零方差或非有限统计立即失败，不以 epsilon/clamp 隐藏退化 batch。
PPO 使用 clipped policy objective、unclipped L2 value objective 与 entropy term，共同更新 policy。
Ratio 为 `exp(new_log_prob_g - old_log_prob_g)`；optimizer 为 Adam，scheduler 为 cosine，参数显式配置。
GAE/PPO 数学复用成熟库，不能另养一套同名公式。

每个完整 epoch 对本 update batch 无放回遍历，不跨 update 复用旧 batch。
`target_kl` 为显式 nullable 配置；非 null 时每个 minibatch forward 后、backward 前，若 approximate
KL `> 1.5 × target_kl`，触发 minibatch 不更新并结束本 update 剩余 epochs/minibatches。
Scheduler 只按实际 optimizer step 前进；evaluated minibatches 与 optimizer steps 分开统计。
Null 关闭该早停，不删除汇总 KL 诊断。PPO ratio、KL 与 policy probe 的诊断不能改变 action RNG。

## 训练统计的分母

Reward sum 是 transition reward 总和，mean 除以 sample_count；普通 rollout mean/sum/max 按
transition 加权，不能称为 episode 平均。Collision/out-of-road transition fraction 不是 episode
failure rate。Energy 强度为总 proxy mL／总距离 km；零距离不发送该可选指标，不补零。
Beta/action 分 lateral/longitudinal 维度报告。

Loss/KL/entropy 对 evaluated minibatches 求均值，包含触发 KL early-stop 的 minibatch；
pre-clip gradient max 只在实际 optimizer steps 上统计，无 step 时为零。
固定 probe 的 before/after 对比不另生成每 update 随机样本。聚合状态每 update 重建，NaN 报错。
持久化与 tracking step 身份由 [artifacts](../../contracts/artifacts.md) 规定。

## 来源与待确认边界

接受依据：[ADR 0016](../../adr/0016-add-forward-only-exploration-policy.md)、
[0018](../../adr/0018-use-torchrl-for-gae-and-ppo-math.md)、
[0024](../../adr/0024-add-plannerrft-energy-reward.md)、
[0039](../../adr/0039-unify-closed-loop-cadence.md)；详细迁移来源是
[旧 training contract](https://github.com/xcz0/Eco-AutoDrive/blob/eaafe5bf911390ef3263bea808aadc40215068a6/docs/agents/contracts/training.md) 与 [旧 experiments](https://github.com/xcz0/Eco-AutoDrive/blob/eaafe5bf911390ef3263bea808aadc40215068a6/docs/agents/contracts/experiments.md)。

ADR 0018 已显式记录原闭区间表述与 ADR 0016 的冲突；有效动作域只由 training contract
定义，本次收口不放宽端点。ADR 0024 已将环境持有 reward 标为历史位置；后续
collector-side ownership 的迁移依据见 execution contract 的来源说明。历史 builtin/smoke reward
不据 ADR 自动恢复为当前可用 profile。未公开的方法选择不宣称论文 parity。

## 代码与测试导航

| 修改面 | 实现／配置 | 相关测试（定位，未运行） |
| --- | --- | --- |
| Beta 策略 | [policy](../../../src/eco_planner/planning/policy/)、[policy config](../../../configs/components/policy.yaml) | [policy inputs](../../../tests/planning/test_policy_inputs.py) |
| Reward 与校准 | [reward](../../../src/eco_planner/reward/)、[离线 reward 适配](../../../src/eco_planner/rl/reward.py) | [reward](../../../tests/training/test_reward.py)、[calibration](../../../tests/training/test_reward_calibration.py) |
| GAE/PPO 与 credit | [optimization](../../../src/eco_planner/rl/optimization/) | [PPO](../../../tests/training/test_ppo.py)、[credit](../../../tests/training/test_credit_assignment.py) |
| 固定批次与 rollout | [rollout](../../../src/eco_planner/rl/rollout/)、[training jobs](../../../configs/jobs/training/) | [fixed batch](../../../tests/training/test_fixed_batch.py)、[rollout](../../../tests/training/test_rollout.py) |
