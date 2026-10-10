# E-051 Issue #105 Phase B — canonical cadence PPO causal counterfactuals

**日期 / 类型 / 目的**：2026-09-28 / 正式 matched 训练因果反事实诊断 / Issue #105 Phase B：在 E-050 Gate A
裁定（critic/shared-trunk/global-clipping coupling 为 primary，reward/value scale 与 physical-time credit 为
upstream）之上，用一组单机制 override 诊断臂分离：哪个机制改变 actor effective update、哪些只是
critic/value-scale 后果、哪些臂不进入正式 control。不进入 optimizer 搜索、不重新冻结 control。

**代码**：Git commit `b788477`（branch `issue83-engery-select`，`git status --short=[]` clean run）；上游
`3eaa87c`（E-050 Phase A）。本次落地：`ppo.diagnostic_reward_divisor`（默认 1.0，仅 GAE 边界缩放 PPO 训练
信号，审计 reward 不变）、`training counterfactual-attribution run/analyze` 工作流与 matched 归因、共享
measurement 抽到 `experiments/training/attribution.py`（Phase A runner 改为委托，语义不变）。

**环境**：Windows-10-10.0.26200 / Python 3.10.20 / torch 2.12.1+cu126 / CUDA:0 RTX A4000 / Lightning 2.6.6 /
MetaDrive 0.4.3；bf16-mixed。

**协议**：protocol `experiments/comparison/calibrated.yaml` arm `r0`（`plannerrft_no_energy_calibrated_v1`），
base job `jobs/training/ppo_conservative`（lr=1.5e-4 / epochs=1 / batch=minibatch=128 / target_kl=0.006 /
mgn=0.5 / scheduler=50），canonical cadence 0.1 s × 5 = 0.5 s pinned，16 场景（S/SC × seeds 0–7）× 8
transitions = 128/update，50 updates，training/runtime seed 0、replay 0、horizon=80、DDIM5，
`gradient_diagnostics=true`。所有 6 臂 matched，唯一差异是各自声明的 override。

**命令**：

~~~powershell
just exp training counterfactual-attribution run `
  --output-dir outputs/studies/scalar-reward/e-051-issue105-phase-b-counterfactuals
~~~

配置清单：`configs/experiments/training/counterfactual-attribution.yaml`；实现位于
`src/eco_planner/experiments/training/counterfactual_attribution/`，共享 measurement 位于
`src/eco_planner/experiments/training/attribution.py`。

**Arms（全部 diagnostic-only，均不进入正式 contract）**：

| label | mechanism | override | 含义 |
| --- | --- | --- | --- |
| `k5_baseline` | canonical_k5_reference | — | canonical k=5 参考臂 |
| `k1_reference` | cadence_reference | `training.diagnostic_execution_steps=1` | E-050 matched k=1 causal reference（actor recovery 目标，非 contract） |
| `reward_scale_normalized` | reward_value_scale | `ppo.diagnostic_reward_divisor=5.0` | 仅把 GAE 训练 reward ÷5，k=5 cadence 与审计 reward 不变 |
| `temporal_credit_time_consistent` | physical_time_credit | `ppo.gamma=0.9509900499`, `ppo.gae_lambda=0.7737809375` | 保持旧 0.1 s 物理衰减时间常数（γ⁵/λ⁵） |
| `value_loss_scaled` | critic_coupling | `ppo.value_coefficient=0.1` | value-loss coefficient ÷5 |
| `global_clip_compensated` | critic_coupling | `ppo.max_gradient_norm=2.5` | 提高 global clip 阈值以恢复 k=1 effective clip coeff |

**结果（per-update median，matched）**：

| metric | k5_baseline | k1_reference | reward_scale | temporal_credit | value_loss_scaled | global_clip |
| --- | --- | --- | --- | --- | --- | --- |
| analytic Beta KL | 6.2837e-7 | 1.4319e-6 | 8.5202e-7 | 1.0198e-6 | 8.7141e-7 | 9.6036e-7 |
| seeded-MC KL | 3.2575e-7 | 1.8124e-6 | 7.1055e-7 | 7.2990e-7 | 7.8882e-7 | 6.4581e-7 |
| k3 KL | 6.0397e-7 | 1.3481e-6 | 8.1604e-7 | 1.1536e-6 | 8.3512e-7 | 9.2477e-7 |
| policy ratio change | 1.0971e-3 | 1.6431e-3 | 1.2741e-3 | 1.5114e-3 | 1.2902e-3 | 1.3572e-3 |
| probe guidance RMS | 0.022046 | 0.033842 | 0.025221 | 0.027638 | 0.025560 | 0.027721 |
| actor_head param Δ | 0.017714 | 0.028490 | 0.022221 | 0.022067 | 0.022356 | 0.022340 |
| value_head param Δ | 0.050872 | 0.050853 | 0.050872 | 0.050882 | 0.050872 | 0.050800 |
| shared_trunk param Δ | 1.7936 | 1.7974 | 1.7936 | 1.7931 | 1.7936 | 2.1361 |
| actor_head_policy grad | 0.36358 | 0.36486 | 0.34800 | 0.34984 | 0.36416 | 0.39636 |
| shared_trunk_policy grad | 0.0058931 | 0.0096412 | 0.0074445 | 0.0070619 | 0.0070311 | 0.0074717 |
| value_head_critic grad | 78.766 | 16.085 | 15.014 | 50.527 | 15.752 | 85.041 |
| shared_trunk_critic grad | 100.29 | 20.509 | 19.116 | 64.314 | 20.057 | 106.84 |
| total pre-clip norm | 127.53 | 26.077 | 24.320 | 81.796 | 25.515 | 136.56 |
| effective clip coeff | 0.0039253 | 0.019199 | 0.020584 | 0.0061196 | 0.019620 | 0.018329 |
| value target mean | 17.271 | 5.1576 | 4.9555 | 11.852 | 17.270 | 17.411 |
| reward total | 498.76 | 100.88 | 498.74 | 498.85 | 498.76 | 498.76 |
| raw advantage std | 7.0045 | 1.3922 | 1.3508 | 3.0639 | 7.0045 | 6.9990 |

**Ratios vs `k5_baseline`（median）**：

| metric | k1_reference | reward_scale | temporal_credit | value_loss_scaled | global_clip |
| --- | --- | --- | --- | --- | --- |
| analytic Beta KL | 2.279 | 1.356 | 1.623 | 1.387 | 1.528 |
| seeded-MC KL | 5.564 | 2.181 | 2.241 | 2.422 | 1.983 |
| policy ratio change | 1.498 | 1.161 | 1.378 | 1.176 | 1.237 |
| probe guidance RMS | 1.535 | 1.144 | 1.254 | 1.159 | 1.257 |
| actor_head param Δ | 1.608 | 1.254 | 1.246 | 1.262 | 1.261 |
| actor_head_policy grad | 1.004 | 0.957 | 0.962 | 1.002 | 1.090 |
| value_head_critic grad | 0.204 | 0.191 | 0.641 | 0.200 | 1.080 |
| shared_trunk_critic grad | 0.204 | 0.191 | 0.641 | 0.200 | 1.065 |
| total pre-clip norm | 0.204 | 0.191 | 0.641 | 0.200 | 1.071 |
| effective clip coeff | 4.891 | 5.244 | 1.559 | 4.998 | 4.669 |
| value target mean | 0.299 | 0.287 | 0.686 | 1.000 | 1.008 |
| reward total | 0.202 | 1.000 | 1.000 | 1.000 | 1.000 |
| raw advantage std | 0.199 | 0.193 | 0.437 | 1.000 | 0.999 |

**Gate（复用冻结 Gate F 阈值，c1 floor=1e-6 / c7 held-out floor=0.1%）**：`k1_reference` 7/7 通过且 c7 超噪声；
`k5_baseline`、`reward_scale_normalized`、`value_loss_scaled`、`global_clip_compensated` 失败 c1 + c7；
`temporal_credit_time_consistent` 仅失败 c1。所有 k=5 cadence 反事实臂 median seeded-MC KL 均 <1e-6。

**held-out（initial→final relative change，S/SC seeds 16–23、horizon 300、runtime seed 760025、ddim5）**：
`k1_reference` 五项指标全部超 0.1% 噪声界（speed +0.341%、distance +0.143%、route +0.151%、energy_total
+0.254%、energy/km +0.134%）。`temporal_credit_time_consistent` 仅 `energy_total_ml` 超界（+0.115%）。
其余臂（含 `k5_baseline`）全部 <0.1%。held-out 只用于归因，不改写历史 Gate T2。

## Gate B 证据与裁定

- **global clipping 是直接 proximal 瓶颈**。`global_clip_compensated`（mgn 0.5→2.5）在 reward/value/advantage
  scale 几乎不变（reward 1.000×、value_target 1.008×、raw_adv_std 0.999×、critic grad 1.065–1.080×）下，把
  effective clip coeff 由 0.00393 恢复到 0.01833（4.67×，≈k=1 的 0.01920），actor_head policy grad 量级不变
  （1.090×），actor_head param Δ 1.261×、analytic KL 1.528×。即 global clip 阈值本身是 actor 有效步长被压缩
  的直接原因。
- **reward/value scale 经 value loss 驱动 pre-clip norm**。两条独立路径都把 critic 梯度与 total pre-clip norm
  降到 ≈0.19–0.20×（≈k=1 的 0.204×）并把 effective clip coeff 恢复到 ≈k=1：`value_loss_scaled`
  （value_coefficient 0.5→0.1；value_target/raw_adv 不变 1.000×）与 `reward_scale_normalized`（÷5；
  reward_total 审计不变 1.000×、value_target 0.287×、raw_adv_std 0.193×）。两臂 actor_head policy grad 基本
  不变（1.002×/0.957×）、actor_head param Δ ≈1.25×、analytic KL 1.36–1.39×。故 reward/value scale 是 upstream，
  作用路径为 value-loss magnitude → critic gradient → 共享 pre-clip norm → global clip。
- **physical-time credit 是第二部分 upstream**。`temporal_credit_time_consistent`（γ⁵/λ⁵）把 critic 梯度与
  total pre-clip norm 降到 0.641×（部分；符合 value target 0.686×、更短有效视界），effective clip coeff 1.559×，
  actor_head param Δ 1.246×、analytic KL 1.623×（本组最高，达 k=1 的 71%）。只有该臂 held-out `energy_total_ml`
  越过噪声界。temporal-credit 通过改变 value-target 尺度对 actor 有效更新有独立贡献；其 held-out 行为后果使其
  不能作为 control。
- **没有 Phase B arm 达到 k=1 effective-update region**。所有 k=5 cadence arm 仍 fail c1：analytic KL 最大
  1.02e-6 < k=1 的 1.43e-6，seeded-MC 全部 <1e-6；actor_head param Δ 最大 ≈1.26×（k=1 为 1.61×）。单独任一
  override（value-loss scaling / reward normalization / clip threshold / temporal credit）都不能完全恢复
  E-039 candidate 的 per-update actor effective update。residual gap 表明完全恢复需要组合或多个耦合项，超出
  override-only 诊断范围。
- **哪些臂不进入正式 control**：`reward_scale_normalized`、`temporal_credit_time_consistent`、
  `value_loss_scaled`、`global_clip_compensated` 全部 diagnostic-only，且无一通过 c1；`temporal` 还存在 held-out
  energy 偏移。`k1_reference` 7/7 通过但 k=1 非 canonical contract。**Phase B 不产生新的 canonical control**；
  Phase C（canonical PPO re-freeze）需在 A/B 归因之上另行设计，且不得把本组任一诊断臂直接当作正式配置。

## 实现变更（本次任务落地）

- `ppo.diagnostic_reward_divisor`（`PPOConfig`，默认 1.0，`>0`）：仅缩放喂给 GAE 的训练 reward
  （`_compute_gae` 入口），单 field 同时覆盖 online training 与 offline provenance 重算；持久化 reward 审计、
  component sums 与 `TrainingUpdateSummary.total_reward` 保持 canonical（`reward_scale_normalized` 的
  reward_total ratio=1.000 为此提供交叉验证）。
- `training counterfactual-attribution run/analyze`：strict manifest（reference + 单机制 `arms`，校验 label 唯一、
  reference 无 override、arm 非空 override、无 `runtime.seed`、无重复 override key），matched 训练、共享
  initial-policy 守卫、离线 GAE provenance 核对、held-out、per-arm Gate 与 `gate_b_evidence`（按 mechanism 分组
  的 ratios_vs_reference）。
- 共享 measurement 抽到 `experiments/training/attribution.py`，Phase A cadence runner 改为委托（既有
  `test_cadence_attribution` 不改动仍通过）。

## 验证与产物

- 测试：`just test-target tests/training/test_counterfactual_attribution.py tests/training/test_ppo.py
  tests/training/test_cadence_attribution.py tests/configuration/test_jobs.py`（58 passed；新增 manifest 校验、
  GAE divisor、Gate B 结构、CLI 覆盖）；`just lint`、`just typecheck` 0 errors。
- 只读核验：6 臂 `git_status_short=[]`、`git_head=b788477`、initial policy hash 全部
  `049697739ab5…`（与 E-038/E-040/E-048/E-049/E-050 一致）、resolved `diagnostic_reward_divisor`/gamma/lambda/
  value_coefficient/mgn 逐臂与声明一致、`execution_steps`：k5 各臂=5 canonical、`k1_reference`=1（runtime
  metadata 记录）。
- checkpoint identity：`k5_baseline` final `7a7809264b2f…`（=E-049/E-050 k5）、`k1_reference` final
  `0c0ff16a42b1…`（=E-050 k1）；`k5_baseline` analytic KL 6.2837e-7 / MC 3.2575e-7 与 E-050 逐值复现。
- 正式产物（git ignored）：
  `outputs/studies/scalar-reward/e-051-issue105-phase-b-counterfactuals/`
  - `study_manifest.yaml`：resolved 研究清单。
  - 6 个臂目录：各 `summary.json`（逐 update 测量）、`resolved_config.yaml`、`runtime_metadata.json`、
    `policy-*.pt`、`updates/update-NNN/*.npz`。
  - `heldout/initial|<arm>/`：matched held-out 评测产物。
  - `summary.json`：逐臂 metrics、Gate、held-out 与 `attribution`（含 `ratios_vs_reference`、`gate_b_evidence`）。
  - `analysis.json`、`report.md`、`figures/post-update-kl.*`。

## 结论边界

**支持**：在 canonical k=5、该 initial policy、该 no-traffic 池、calibrated R0 下，E-039 candidate 的 per-update
actor effective update 被压缩的直接 proximal 瓶颈是单一 global clip；其 upstream 为 value-loss magnitude 所
放大的 critic/shared-trunk 梯度，而 value-loss magnitude 又由 per-transition reward/value scale（多 substep
求和）与 physical-time credit 尺度决定。两条独立降 critic-gradient 路径（value-loss scaling、reward ÷5）都恢复
effective clip coeff 到 ≈k=1 并部分恢复 actor 步长，且 actor_head policy grad 本身不变（排除 actor/state/advantage
geometry）；提高 clip 阈值在不动 reward/value scale 的前提下同样部分恢复 actor 步长。

**不支持 / 限制**：

- 单 seed、单 no-traffic 训练池、单 initial policy；不构成跨 seed 总体结论。
- `global_clip_compensated` 同时放大 actor 与 critic 更新，不是纯 clip 隔离；未执行 actor-only/separated
  clipping 与 shared-trunk critic gradient detach 两个架构级 counterfactual（保留为未执行项）。
- `reward_scale_normalized` 只缩放 GAE 训练信号，不等价于修改 reward representation 或正式 reward 语义
  （审计 reward 不变）。
- 没有任何 override-only 诊断臂通过 c1 或达到 k=1 effective-update region；residual gap 未归因。
- held-out 变化方向（temporal 的 energy_total 偏移）不解释为节能或行为改善。
- 不做 optimizer 搜索、不重排 reward/PPO；不进入 Phase C，不宣称正式 control 已 re-freeze。
