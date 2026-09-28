# E-053 Issue #106 Task 1：canonical PPO re-entry / Gate T2

[返回实验索引](../README.md) · Issue #106 Task 1 · [E-052 冻结 control](e-052-issue105-phase-c-canonical-ppo-refreeze.md) ·
[E-049 历史 Gate T2](e-049-issue83-task-1b-effective-update-transfer.md) · Issue #105 Phase A/B/C ·
[ADR 0039](../../adr/0039-unify-closed-loop-cadence.md)

**日期 / 类型 / 目的**：2026-09-28 / 正式单臂训练 transfer gate / Issue #106 Task 1：把 #105 handoff
（E-052 冻结的 canonical k=5 PPO control、结果前冻结的搜索空间与选择规则、analytic Beta KL 实现与验证、
E-050/E-051 cadence-PPO 归因结论与 provenance）写回 #106，并以 calibrated R0、canonical k=5、冻结 initial
policy 与 matched protocol **重新执行 effective-update transfer（Gate T2）**。复用 E-049 的单臂
`TrainingGridConfig` + `experiments/training/grid.py::run`（冻结 Gate F 七条判据 = Gate T2），唯一变化是把
E-052 选定的 control override 写入 `base_overrides`；不重新搜索 PPO、不改 reward representation / λ /
policy feature / cadence，不按 training return 选 control。

**裁定：Gate T2 PASSED（7/7）。** median seeded-MC post-update KL `1.0437e-6 ≥ 1e-6`（analytic
`1.3331e-6`、k3 `1.4957e-6` 同量级），c1–c7 全部通过；policy 确有效更新（final hash `39aeb2571cad…`
≠ initial `049697739ab5…`），无 NaN/Inf/invalid action/Beta boundary collapse/behavioral collapse。

## #105 → #106 handoff（写回内容）

### frozen canonical PPO control

label `time_consistent_value_scaled`；相对 canonical R0 的 Hydra overrides（arm `r0` =
`plannerrft_no_energy_calibrated_v1`）：

- `ppo.value_coefficient=0.1`
- `ppo.gamma=0.9509900499000001`（=0.99⁵）
- `ppo.gae_lambda=0.7737809375`（=0.95⁵）

其余保持 canonical：`learning_rate=1.5e-4`、`epochs=1`、`max_gradient_norm=0.5`、`clip_epsilon=0.2`、
`target_kl=0.006`、batch=minibatch=128、`entropy_coefficient=0.01`、`scheduler_total_optimizer_steps=50`；
cadence `0.1 s × 5 = 0.5 s`。resolved config（E-052）：
`outputs/studies/scalar-reward/e-052-issue105-phase-c-canonical-ppo-refreeze/time_consistent_value_scaled/resolved_config.yaml`；
run commit `06a7d62`（branch `issue83-engery-select`，clean）。

### 搜索空间与选择规则（结果前冻结）

`configs/experiments/training/control-refreeze.yaml` 提交于 `e6a856b`（任何 Phase C 结果产生前）：7 个
命名单一/组合 control 候选按声明优先级排列，选择规则 `first_passing_in_declared_order`（取第一个通过
Gate c1–c7 的候选，**不按 training return 排名**）。候选只允许覆盖
`ppo.{learning_rate,epochs,minibatch_size,gamma,gae_lambda,value_coefficient,max_gradient_norm,clip_epsilon}`
（`ALLOWED_CONTROL_OVERRIDE_KEYS`）；reward representation / λ / policy feature / cadence / batch 在解析期
拒绝。`lr2` 也 7/7 通过但声明优先级更低、变化量级大得多，未选中；3 个 lr≥3e-4 臂训练内 Beta 边界塌缩
（完整记录为失败候选）。

### analytic Beta KL 实现与验证

- 实现：`src/eco_planner/analysis/training.py::beta_kl`（闭式 `KL(Beta(old)||Beta(new))`，float64，
  affine Jacobian 抵消，两维求和，非有限值抛错）；接入
  `src/eco_planner/rl/optimization/update_diagnostics.py::post_update_kl_series`，与 seeded-MC KL、k1/k3
  估计量并存。
- 验证：`tests/training/test_effective_update.py::test_beta_kl_matches_closed_form_and_is_zero_for_identical_parameters`
  （同参数为 0、与闭式一致、形状守卫）。

### E-050/E-051 cadence-PPO 归因结论

- E-050 Gate A：k=5 相对 k=1 的 actor effective update 下降 primary 归因为
  **critic/shared-trunk/global-clipping coupling**（critic 梯度与 pre-clip norm ~4.9×、effective global
  clip coeff 0.0192→0.0039），upstream 为 reward/value scale 与 physical-time credit；actor/state/advantage
  geometry 被排除（z-advantage std ratio 1.000、actor_head policy grad ratio 0.997）。
- E-051 Gate B：global clipping 是直接 proximal 瓶颈；reward/value scale 经 value loss 驱动 critic 梯度与
  pre-clip norm；γ⁵/λ⁵ 是第二部分 upstream；**无任一 override-only 诊断臂达到 k=1 region**，全部
  diagnostic-only 不进入正式 contract。
- E-052 Phase C 据此独立重冻结 `time_consistent_value_scaled`（同时修 value-loss magnitude 与
  physical-time credit，不动 lr），不按 return 选择。

## 协议与冻结契约

单臂 study manifest `configs/experiments/training/e-053-issue106-task-1-gate-t2-reentry.yaml`
（`TrainingGridConfig`），复用 `experiments/training/grid.py::run` 的 Gate F 实现，grid 为单元素
`(lr=1.5e-4, epochs=1, mgn=0.5)`，不构成搜索：

- protocol `experiments/comparison/calibrated.yaml` arm `r0`（`plannerrft_no_energy_calibrated_v1`，
  E-034 冻结 Progress/Comfort 校准）；base job `jobs/training/ppo_conservative`；
- `base_overrides` 注入 E-052 冻结 control：`ppo.value_coefficient=0.1`、`ppo.gamma=0.9509900499000001`、
  `ppo.gae_lambda=0.7737809375`（其余来自 protocol：`target_kl=0.006`、batch=128、
  `transitions_per_environment=8`）；
- canonical cadence `0.1 s × 5 = 0.5 s`（`ClosedLoopCadenceConfig` typed boundary pin）；
- training/runtime seed 0、replay 0；16 scenarios（S/SC × map seeds 0–7）× 8 transitions/env =
  **128 transitions / update**；DDIM5；horizon 80；
- Gate T2 阈值逐值复制自 `configs/experiments/training/grid.yaml`（E-039 冻结 Gate F），`mc_draws=4096`、
  `mc_seed=1000003`。

**命令**：

```powershell
just exp training grid `
  --config configs/experiments/training/e-053-issue106-task-1-gate-t2-reentry.yaml `
  --output-dir outputs/studies/scalar-reward/e-053-issue106-task-1-gate-t2-reentry
```

resolved config 核对（`lr1.5000e-04-epochs1-mgn0.5/resolved_config.yaml`）：cadence `0.1/5/0.5`、reward
`plannerrft_no_energy_calibrated_v1`、`value_coefficient=0.1` / `gamma=0.9509900499000001` /
`gae_lambda=0.7737809375`、lr=1.5e-4 / epochs=1 / mgn=0.5 / target_kl=0.006、batch=minibatch=128、
transitions=8、update_count=50、training/runtime seed 0、replay 0、`execution_steps=5`、
`gradient_diagnostics=false`。

## Gate T2 判定结果

```text
Verdict: PASSED（update_gate_passed=true，selected_config=lr1.5000e-04-epochs1-mgn0.5）
failure_reasons = []
```

| 条件 | 判据 | 实测 | 通过 |
| --- | --- | ---: | :---: |
| c1 median post-update KL | ≥ 1e-6 | **1.0437e-6**（MC） | ✓ |
| c2 KL within target / 无 runaway | ≥90% 且 tail-10 ≤10×median | within=1.00；tail median 6.74e-8；runaway=false | ✓ |
| c3 policy ratio 变化 | ≥ 1e-4 | 1.7194e-3 | ✓ |
| c4 deterministic Beta mean RMS shift | ≥ 0.01 | 0.031113 | ✓ |
| c5 无 Beta boundary collapse | min α,β ≥0.1 且 boundary mass ≤0.2 | min α/β 1.8979/1.9850；mass 0.0225 | ✓ |
| c6 无 behavioral collapse | collision+OOR ≤2 且 ep-len 保留 ≥50% | 0/0；8.0 → 8.0 | ✓ |
| c7 held-out 超噪声 | ≥1 指标 ≥0.1% | distance +0.1058%、energy_total +0.1364% | ✓ |

其他训练诊断（50 updates）：`under_update=false`；`post_update_kl_within_target_fraction=1.0`；
pre-clip gradient norm median `16.373`（post-clip ≤ mgn=0.5）；无 NaN/Inf、无 invalid action、无 Beta
boundary collapse、无 collision/OOR。

## Estimator 一致性（analytic / seeded-MC / k3）

| 估计量 | median | 相对最小值 |
| --- | ---: | ---: |
| analytic Beta KL | 1.3331e-6 | 1.277 |
| seeded-MC KL（primary） | 1.0437e-6 | 1.000 |
| single-draw k3 KL | 1.4957e-6 | 1.433 |

三估计量同量级（max/min = 1.433 ≤ 10×）且有限，对 update strength 给出一致数量级判断；primary
seeded-MC KL 高于冻结 floor `1e-6`。（single-draw k1 median `-2.4e-6` 为 1e-4 量级 batch noise 的有符号
估计，不作 floor 判定。）

## Checkpoint 与 provenance

- initial policy hash `049697739ab5d6e7bd212938bbac82c35215eb9ed14c02557952098a9718d05d`（与
  E-038/E-040/E-048/E-049/E-050/E-051/E-052 一致）；
- final policy hash `39aeb2571cad738c2a8defb016806128fd9c9d26699df748d7465ebb22818a56`
  （**与 E-052 selected control final hash `39aeb2571cad…` 逐值一致** → 同一冻结 control 的独立 matched
  复跑）；
- run commit `6cd818a`（branch `issue83-engery-select`，`git_status_short=[]` clean；由
  `runtime_metadata.json` 记录）；manifest + 回归测试提交 `6cd818a`；上游冻结资产见下；
- EMA 276 tensors / 6,042,628 parameters；runtime device cuda:0、bf16-mixed、seed 0。

## 验证与产物

- 代码级测试：`just test-target tests/training/test_training_workflows.py`（10 passed，含新增
  `test_e053_gate_t2_reentry_manifest_pins_frozen_canonical_control`：单臂 grid、Gate 阈值 == `grid.yaml`、
  组合后 policy 为 `value_coefficient=0.1` / `gamma=0.9509900499000001` / `gae_lambda=0.7737809375`、
  canonical cadence/reward/PPO）。
- 正式产物经只读核验：`git_status_short=[]`、`git_head=6cd818a`、`execution_steps=5`、50 updates、
  initial/final policy hash 符合预期，c1–c7 与 estimator 均有限。
- 正式产物（git ignored）：
  `outputs/studies/scalar-reward/e-053-issue106-task-1-gate-t2-reentry/`
  - `study_manifest.yaml`：resolved 单臂 grid + Gate T2 阈值。
  - `lr1.5000e-04-epochs1-mgn0.5/`：`summary.json`（逐 update 训练诊断）、
    `resolved_config.yaml`、`runtime_metadata.json`、`policy-initial/final/update-NNN.pt`、
    `updates/update-NNN/*.npz`（完整 rollout audit）。
  - `heldout/initial|lr1.5000e-04-epochs1-mgn0.5/`：matched held-out 评测产物（S/SC seeds 16–23）。
  - `summary.json`：逐 arm metrics、Gate T2 判定与选择结果。
  - `analysis.json`、`report.md`、`figures/post-update-kl.*`。

## 结论边界

**支持**：在 canonical k=5 cadence、该 initial policy、该 no-traffic 训练池、calibrated R0 下，E-052 冻结的
canonical PPO control（`time_consistent_value_scaled`）与 E-049 的 Gate T2 匹配协议下 **c1–c7 全通过**：
median seeded-MC post-update KL `1.0437e-6 ≥ 1e-6`（analytic/k3 同量级），ratio 变化 `1.719e-3`、probe RMS
`0.0311`、min α/β `1.898/1.985`、0 collision/OOR、ep-len 8→8、held-out distance +0.106% /
energy_total +0.136%。final policy hash 与 E-052 selected control 逐值一致，确认这是同一 control 的
独立复跑。E-049 的 Gate T2 FAILED 由 canonical PPO control re-freeze（#105）解决，canonical cadence
transfer 在该 control 下成立。

**不支持 / 限制**：

- **c1 余量仍薄**：seeded-MC KL `1.0437e-6` 仅高于 floor `1e-6` 约 4.4%（analytic/k3 ~1.3–1.5e-6 同侧）；
  单点值不应读作精确量，判定以三估计量同侧为准。本结果不证明该 control 对 floor 边界有稳健裕度。
- 单 seed、单 no-traffic 训练池、单 initial policy；不构成跨 seed 总体结论，也不证明 `time_consistent_value_scaled`
  是唯一或最优 control（`lr2` 亦通过 Gate，但未选中）。
- held-out 变化方向不解释为节能或行为改善；仅确认 control 有可测 learned effect。
- Gate T2 通过仅支持「effective-update transfer 成立」，不替代后续 Task 2（R0 vs λ64 canonical
  positive-control / Gate T3）与 Task 3（λ={0,1,2,4,8} dose-response / Gate D）——两者未在本记录执行。
- 未在 #106 完成 Task 2/3 前，不宣称 canonical objective→behavior direction transfer 或 λ dose-response。

## Issue #106 Task 1 provenance 清单

- run commit `6cd818a`、branch `issue83-engery-select`、`git_status_short=[]`（clean run）；上游冻结资产：
  Diffusion Planner 上游源码 `a3a621f0b724c5fa6447f7a2fbaf9e0387bd35df`、revision
  `ae5baf1c57229c53f6309332df960ae27d35333f`、EMA 276 tensors / 6,042,628 parameters；
- 环境：Windows-10-10.0.26200 / Python 3.10.20 / torch 2.12.1+cu126 / CUDA:0 RTX A4000 / Lightning 2.6.6 /
  MetaDrive 0.4.3；bf16-mixed；
- resolved config：`lr1.5000e-04-epochs1-mgn0.5/resolved_config.yaml`（cadence/reward/PPO/scenarios 已核对）；
- checkpoint identity：initial policy `049697739a…`（与 E-038/E-040/E-048/E-049/E-050/E-051/E-052 一致）、
  final policy `39aeb2571cad…`（=E-052 selected control）；
- seeds：training/runtime seed 0、replay 0、16 scenarios S/SC seeds 0–7；DDIM5；
- cadence：0.1 s × 5 = 0.5 s（`ClosedLoopCadenceConfig` typed boundary 校验）；
- commands / artifact path：见上；
- 测试结果：10 passed；failure 状态：无（Gate T2 PASSED 7/7）。
