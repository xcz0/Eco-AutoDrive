# E-052 Issue #105 Phase C — independent canonical k=5 PPO control re-freeze

[返回实验索引](../README.md) · Issue #105 Phase C · [E-050 Phase A](e-050-issue105-phase-a-cadence-attribution.md) ·
[E-051 Phase B](e-051-issue105-phase-b-counterfactuals.md) · [E-049 Gate T2](e-049-issue83-task-1b-effective-update-transfer.md)

**日期 / 类型 / 目的**：2026-09-28 / 正式 matched 训练 control 重冻结 / Issue #105 Phase C：在 E-050 Gate A
（critic/shared-trunk/global-clipping coupling 为 primary，reward/value scale 与 physical-time credit 为
upstream）与 E-051 Gate B（无任一 override-only 诊断臂达到 k=1 effective-update region）之上，用一个
**结果前冻结**的候选搜索空间与选择规则，在 calibrated R0 + canonical k=5 execution 下重新冻结一个通过
Gate 的 canonical PPO control，供 #106 重跑 Gate T2。不做 λ sweep、不改 reward representation / policy
feature / cadence、不按 training return 选 control。

**代码**：Git commit `06a7d62`（branch `issue83-engery-select`，`git_status_short=[]` clean run；由
`runtime_metadata.json` 逐 run 记录）；冻结搜索空间与选择规则的配置提交 `e6a856b`（Phase C 实现 +
`configs/experiments/training/control-refreeze.yaml`，在任何 Phase C 结果产生前提交）；上游 `f0f3adf`
（Phase B 记录）。本次落地：`training control-refreeze run/analyze` 工作流、严格 manifest
（`ALLOWED_CONTROL_OVERRIDE_KEYS` 守卫仅允许 optimizer control 旋钮）、`first_passing_in_declared_order`
选择规则、逐候选 Gate 与 held-out、analytic/MC/k3 estimator 一致性检查、非有限值守卫，以及候选训练崩溃的
失败记录（不中断整个搜索）。

**环境**：Windows-10-10.0.26200 / Python 3.10.20 / torch 2.12.1+cu126 / CUDA:0 RTX A4000 / Lightning 2.6.6 /
MetaDrive 0.4.3；bf16-mixed。

**协议**：protocol `experiments/comparison/calibrated.yaml` arm `r0`
（`plannerrft_no_energy_calibrated_v1`），base job `jobs/training/ppo_conservative`，canonical cadence
0.1 s × 5 = 0.5 s pinned，16 场景（S/SC × seeds 0–7）× 8 transitions = 128/update，50 updates，
training/runtime seed 0、replay 0、horizon=80、DDIM5、batch=minibatch=128、target_kl=0.006、
`gradient_diagnostics=true`。所有 run matched，唯一差异是各候选声明的 control override。

**命令**：

~~~powershell
just exp training control-refreeze run `
  --output-dir outputs/studies/scalar-reward/e-052-issue105-phase-c-canonical-ppo-refreeze
~~~

配置清单：`configs/experiments/training/control-refreeze.yaml`；实现位于
`src/eco_planner/experiments/training/control_refreeze/`。

## 结果前冻结的搜索空间与选择规则

搜索空间为 7 个命名单一/组合 control 候选（按声明优先级排列），选择规则为
`first_passing_in_declared_order`：按声明顺序取第一个通过 Gate c1–c7（c1 analytic/MC KL ≥1e-6；c2 KL
稳定且无 runaway；c3 ratio 变化 ≥1e-4；c4 probe RMS shift ≥0.01；c5 无 Beta boundary collapse；c6 无
behavioral collapse；c7 held-out 超噪声）的候选。**不按 training return 排名**。候选仅允许覆盖
`ppo.{learning_rate,epochs,minibatch_size,gamma,gae_lambda,value_coefficient,max_gradient_norm,clip_epsilon}`；
reward representation / λ / policy feature / cadence 在 manifest 校验期被拒绝。`canonical_k5` reference
为无 override 的校准基线。

机制依据（来自 E-050/E-051）：k=5 per-transition reward 为 5 substep 求和使 value-loss magnitude 与
critic 梯度放大约 5×，global clipping 把共享 actor-critic 梯度整体压缩（effective clip coeff
0.0192→0.0039），同时 γ/λ 按 transition 生效使物理 credit 视界约扩大 5×。故优先级最高的候选同时修正
value-loss magnitude（`value_coefficient=0.1`）与 physical-time credit（`gamma=0.99^5`、
`gae_lambda=0.95^5`），不触碰学习率；lr-only 候选仅作保守 fallback 排在末尾。

## 候选与结果（per-update median，matched）

| candidate | override | status | analytic KL | seeded-MC KL | k3 KL | ratio change | probe RMS | actor param Δ | eff clip coeff | pre-clip norm | value target | raw adv std | gate |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| `canonical_k5` (ref) | — | completed | 6.284e-7 | 3.258e-7 | 6.040e-7 | 1.097e-3 | 0.02205 | 0.01771 | 0.00393 | 127.53 | 17.271 | 7.0045 | c1+c7 fail |
| `time_consistent_value_scaled` | `value_coefficient=0.1`, `gamma=0.99^5`, `gae_lambda=0.95^5` | completed | **1.333e-6** | **1.044e-6** | **1.496e-6** | 1.719e-3 | 0.03111 | 0.02641 | 0.03057 | 16.373 | 11.852 | 3.0639 | **7/7 pass** |
| `value_loss_scaled` | `value_coefficient=0.1` | completed | 8.714e-7 | 7.888e-7 | 8.351e-7 | 1.290e-3 | 0.02556 | 0.02236 | 0.01962 | 25.515 | 17.270 | 7.0045 | c1+c7 fail |
| `time_consistent` | `gamma=0.99^5`, `gae_lambda=0.95^5` | completed | 1.020e-6 | 7.299e-7 | 1.154e-6 | 1.511e-3 | 0.02764 | 0.02207 | 0.00612 | 81.796 | 11.852 | 3.0639 | c1 fail |
| `time_consistent_value_scaled_lr2` | + `learning_rate=3e-4` | **failed** | — | — | — | — | — | — | — | — | — | — | invalid action |
| `time_consistent_value_scaled_lr3` | + `learning_rate=4.5e-4` | **failed** | — | — | — | — | — | — | — | — | — | — | non-finite log-prob |
| `lr2` | `learning_rate=3e-4` | completed | 1.113e-4 | 1.095e-4 | 9.987e-5 | 1.413e-2 | 0.29642 | 0.03490 | 0.00088 | 568.10 | 26.913 | 6.7125 | 7/7 pass |
| `lr3` | `learning_rate=4.5e-4` | **failed** | — | — | — | — | — | — | — | — | — | — | non-finite log-prob |

**选择结果**：`first_passing_in_declared_order` → **`time_consistent_value_scaled`**。它是声明顺序中第一个
通过全部 Gate 的候选；`lr2` 也 7/7 通过但因声明优先级更低、且 KL/ratio 变化量级大得多（analytic 177×、
ratio 变化 12.9×、probe RMS 13.4×、pre-clip norm 4.45×）未被选中。**未按 training return 选择**。

## Estimator 一致性（candidate 验证要求）

所有 completed 候选的三估计量同量级：`canonical_k5` ratio 1.93、`time_consistent_value_scaled` ratio 1.43、
`value_loss_scaled` 1.11、`time_consistent` 1.58、`lr2` 1.11，均 ≤10× 且有限。selected 的 analytic
1.333e-6 / MC 1.044e-6 / k3 1.496e-6 给出一致数量级判断（effective-update 已回到 1e-6 floor 之上）。
无非有限值、无 NaN/Inf。

## 失败候选（invalid action / Beta boundary collapse）

- `time_consistent_value_scaled_lr2`：训练后 probe 触发
  `ValueError: guidance action must be strictly inside (-1, 1)`（Beta boundary collapse 到采样动作越界）。
- `time_consistent_value_scaled_lr3`、`lr3`：
  `RuntimeError: rollout host tensor 'old_joint_guidance_log_prob' contains non-finite values`（Beta
  参数塌缩导致 log-prob 非有限）。

三者均为 lr≥3e-4 的激进臂，与 E-039 观察到的 lr≥4.5e-4 / epochs≥2 训练内 Beta 边界塌缩方向一致。失败被
完整记录（`summary.json` 的 `failed_candidates` 与逐 run `failure{exception_type,message,traceback}`），
不进入选择，也未被静默跳过为“成功”。

## selected control（frozen canonical k=5 PPO control）

- label：`time_consistent_value_scaled`
- Hydra overrides（相对 canonical R0）：`ppo.value_coefficient=0.1`、
  `ppo.gamma=0.9509900499000001`（=0.99⁵）、`ppo.gae_lambda=0.7737809375`（=0.95⁵）
- 其余 resolved PPO 保持 canonical：`learning_rate=1.5e-4`、`epochs=1`、`max_gradient_norm=0.5`、
  `clip_epsilon=0.2`、`target_kl=0.006`、batch=minibatch=128、`entropy_coefficient=0.01`、
  `value_coefficient=0.1`、`gamma=0.9509900499000001`、`gae_lambda=0.7737809375`，
  `scheduler_total_optimizer_steps=50`；cadence 仍 0.1 s × 5 = 0.5 s。
- resolved config：`.../time_consistent_value_scaled/resolved_config.yaml`（`summary.json` 内
  `selected_config.resolved_ppo` 同步内联）。

selected final policy hash `39aeb2571cad…`（相对初始 `049697739ab5…` 确有更新，非 no-op）；shared
initial policy hash 全部 completed run 一致（`049697739ab5…`，与 E-038/E-040/E-048/E-049/E-050/E-051 一致）；
`canonical_k5` final hash `7a7809264b2f…`、analytic KL 6.2837e-7 / MC 3.2575e-7 逐值复现
E-049/E-050/E-051。

## selected 相对 canonical 的归因量（ratios_vs_reference）

| 量 | selected | ratio |
| --- | ---: | ---: |
| analytic KL | 1.333e-6 | 2.122 |
| seeded-MC KL | 1.044e-6 | 3.204 |
| k3 KL | 1.496e-6 | 2.476 |
| policy ratio change | 1.719e-3 | 1.567 |
| probe guidance RMS | 0.03111 | 1.411 |
| actor_head policy grad | 0.34995 | 0.963 |
| actor_head param Δ | 0.02641 | 1.491 |
| shared_trunk param Δ | 1.79302 | 1.000 |
| value_head param Δ | 0.050882 | 1.000 |
| shared_trunk critic grad | 12.863 | 0.128 |
| value_head critic grad | 10.105 | 0.128 |
| pre-clip gradient norm | 16.373 | 0.128 |
| effective global clip coeff | 0.03057 | 7.789 |
| value target mean | 11.852 | 0.686 |
| raw advantage std | 3.0639 | 0.437 |
| reward total | 498.85 | 1.000 |

即：selected 在不改变 reward representation / actor_head policy grad（0.963×，排除 actor/state geometry
变化）的前提下，把 critic 梯度与 pre-clip norm 降到 0.128×、effective clip coeff 恢复到基本不裁剪
（7.789×），使 actor 有效步长恢复（analytic 2.12×、ratio 1.57×、probe RMS 1.41×、actor param Δ 1.49×），
与 E-050/E-051 归因的两条 upstream 机制（value-loss magnitude + physical-time credit）一致。

## Gate / 健康检查（selected）

| 条件 | 判据 | selected |
| --- | --- | --- |
| c1 median KL ≥ 1e-6 | MC | ✓ 1.044e-6（analytic 1.333e-6、k3 1.496e-6 同侧） |
| c2 KL 稳定无 runaway | within≥90%、tail-10 ≤10×median | ✓ |
| c3 policy ratio 变化 ≥1e-4 | — | ✓ 1.719e-3 |
| c4 probe RMS shift ≥0.01 | — | ✓ 0.03111 |
| c5 无 Beta boundary collapse | min α,β ≥0.1、boundary mass ≤0.2 | ✓ min α/β 1.898/1.985、mass 0.0225 |
| c6 无 behavioral collapse | collision+OOR ≤2、ep-len 保留 ≥50% | ✓ 0/0、8.0→8.0 |
| c7 held-out 超噪声 | ≥1 指标 ≥0.1% | ✓ distance +0.106%、energy_total +0.136% |

held-out（initial→final relative change，S/SC seeds 16–23、horizon 300、runtime seed 760025、ddim5）：
speed +0.0861%、distance +0.1058%、route +0.0863%、energy_total +0.1364%、energy_ml_per_km +0.0287%。
held-out 只用于确认 control 有可测 learned effect，不解释为节能或行为改善，不改写 E-049 历史 Gate T2。

## Phase B diagnostic-only counterfactual 澄清（Issue 完成判据）

E-051 的 `k1_reference`、`reward_scale_normalized`、`temporal_credit_time_consistent`、
`value_loss_scaled`、`global_clip_compensated` **全部是 matched causal diagnostic，未进入正式 contract**。
Phase C 的 selected control **不是** Phase B 任一诊断臂的直接晋升：它来自独立的、结果前冻结的 control
搜索空间；其中 `time_consistent_value_scaled` 虽与 Phase B 的 `temporal_credit_time_consistent` 和
`value_loss_scaled` 共享机制方向，但是新组合候选、以正式 control 目标重新训练并单独验证。Phase B 的
`reward_scale_normalized`（`diagnostic_reward_divisor`）与 `k1_reference`（
`diagnostic_execution_steps=1`）不进入 control。

## 验证与产物

- 测试：`just test-target tests/training/test_control_refreeze.py`（6 passed：manifest control-space 守卫、
  唯一 label、reference 无 override、`first_passing_in_declared_order`、estimator 一致性/有限值、失败候选
  记录并继续、CLI run/analyze 路由）；`just test-workflow training`（248 passed）；`just lint`、
  `just typecheck` 0 errors。
- 只读核验：全部 run `git_status_short=[]`、`git_head=06a7d62`、initial policy hash 唯一
  `049697739ab5…`、`execution_steps=5`（canonical k=5，`canonical_k5` 与候选一致）、resolved
  `value_coefficient/gamma/gae_lambda/learning_rate` 逐候选与声明一致、`canonical_k5` analytic/MC KL 与
  final hash 逐值复现 E-049/E-050/E-051。
- 正式产物（git ignored）：
  `outputs/studies/scalar-reward/e-052-issue105-phase-c-canonical-ppo-refreeze/`
  - `study_manifest.yaml`：resolved 冻结搜索空间 + 选择规则。
  - 8 个 run 目录（reference + 7 candidates）：`summary.json`、`resolved_config.yaml`、
    `runtime_metadata.json`、`policy-*.pt`、`updates/update-NNN/*.npz`；失败的 3 个候选保留 partial 证据。
  - `heldout/initial|*/`：matched held-out 评测产物。
  - `summary.json`：逐 run metrics、Gate、held-out、`estimator_consistency`、`failed_candidates`、
    `selected_config` 与 `attribution`（含 `ratios_vs_reference`）。
  - `analysis.json`、`report.md`、`figures/post-update-kl.*`。

## 结论边界

**支持**：在 canonical k=5、该 initial policy、该 no-traffic 池、calibrated R0 下，Phase C 冻结搜索空间中
**第一个通过 Gate 的 control 是 `time_consistent_value_scaled`**（`value_coefficient=0.1` +
`gamma=0.99^5` + `gae_lambda=0.95^5`，lr 保持 1.5e-4）。它在不改变 actor_head policy grad / reward
representation / cadence 的前提下，把 critic 梯度与 pre-clip norm 降到 0.128×、恢复 effective clip
coeff 到 0.0306（基本不裁剪），使 per-update actor effective update 回到 1e-6 floor 之上（analytic
1.333e-6 / MC 1.044e-6 / k3 1.496e-6，同量级），c1–c7 全通过，无 NaN/Inf/invalid action/Beta boundary
collapse/behavioral collapse，held-out 出现可测变化。这是测试空间内机制可解释、可核验的 canonical
control，供 #106 重跑 Gate T2。

**不支持 / 限制**：

- **c1 余量薄**：selected seeded-MC KL `1.044e-6` 仅高于冻结 floor `1e-6` 约 4.4%，analytic/k3 约 1.3e-6；
  单点值不应读作精确量。因此该 control 在 #106 单次 T2 复跑中对 floor 边界敏感，判定以三估计量同侧为准。
- 单 seed、单 no-traffic 训练池、单 initial policy；不构成跨 seed 总体结论。
- 搜索空间只覆盖出题允许的 optimizer control 旋钮；`time_consistent_value_scaled` 是“同时修 value-loss
  magnitude 与 physical-time credit”的机制候选，不是对两条机制各自独立贡献的分离证明（该分离属 Phase B
  diagnostic）。
- `lr2` 也通过 Gate，但未选中；其 KL/ratio/probe 变化量与 pre-clip norm 远大于 selected，说明“通过
  Gate”不等于唯一，选择由结果前冻结的声明优先级决定。
- lr≥3e-4 的组合/lr-only 臂在训练内 Beta 边界塌缩（invalid action / 非有限 log-prob），已按失败候选记录；
  这符合 E-039 的历史观察，但不能据此一般化到其他 seed/配置。
- held-out 变化方向不解释为节能或行为改善；本实验不重排 reward/PPO 主效应，不进入 λ sweep 或 final
  A0/A1/A2。
- 未在 #106 重跑 Gate T2 前，不宣称 canonical cadence transfer 已通过；本记录只冻结 control 与 provenance。

## Issue #105 Phase C provenance 清单

- commit `06a7d62`、branch `issue83-engery-select`、`git_status_short=[]`（clean run）；冻结搜索空间配置
  提交 `e6a856b`（结果前）；上游 `f0f3adf`；
- 环境：Windows-10-10.0.26200 / Python 3.10.20 / torch 2.12.1+cu126 / CUDA:0 RTX A4000 / Lightning 2.6.6 /
  MetaDrive 0.4.3；
- resolved config：各 run `resolved_config.yaml`；selected 见上；
- checkpoint identity：initial policy `049697739ab5…`（与 E-038/E-040/E-048/E-049/E-050/E-051 一致）、
  `canonical_k5` final `7a7809264b2f…`（=E-049/E-050/E-051）、selected final `39aeb2571cad…`；
- seeds：training/runtime seed 0、replay 0、16 scenarios S/SC seeds 0–7；DDIM5；cadence 0.1 s × 5 = 0.5 s；
- commands / artifact path：见上；
- failure 状态：3 个激进候选训练内 Beta 边界塌缩（完整记录）；reference 与 selected 无异常。
