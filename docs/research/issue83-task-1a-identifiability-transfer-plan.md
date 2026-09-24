# Issue #83 Task 1A — Objective identifiability transfer 执行计划

状态：**已执行**（工作文档）。执行结果与 provenance 见
[E-048 实验记录](../../experiments/records/e-048-issue83-task-1a-objective-identifiability-transfer.md)；
本文件保留预登记方案，不在此维护结果。

任务边界、验收标准与冻结契约的权威来源是 [Issue #83](https://github.com/)（Parent #80；历史前置 #94/#98 已关闭；execution protocol 来自 #96 Task E / ADR 0039）。本文档只记录本次 Task 1A 的执行方案；执行完成后由 `docs/experiments/records/` 的实验记录与对应权威文档取代，不在本文件维护结果。

## 1. 任务定义（来自 Issue #83 Task 1A）

目的：**不重新搜索 reward/PPO**，只验证 #94 冻结的 reward representation（E-034 校准常数 + E-038 band 阈值）在 ADR 0039 canonical cadence（k=5 × 0.1 s = 0.5 s decision interval）的 MDP 时间语义下仍可辨识。

协议：

1. 在 canonical k=5 rollout 上采集新的 fixed source batch；
2. 使用冻结 calibration / band thresholds（E-034 / E-038），不从新 batch 重新推导；
3. 离线比较 calibrated R0、finite stress λ、Energy-only endpoint；
4. 保存 raw / centered / z-normalized advantage、actor-head/lon/lat gradient、component variance 与 multi-substep reward diagnostics。

**Gate T1（复用 #94 Gate C primary threshold，z-form / 实际 PPO preprocessing）**：

```text
cos(g_R0, g_EnergyOnly) <= 0.99            # actor-head 梯度余弦
AND
( normalized-advantage RMSE >= 0.10
  OR sign-flip fraction >= 0.05 )
AND
至少一个 finite stress arm 达到 R0→Energy-only angular separation 的 >= 50%   # 优先复核 λ=64
```

失败 → 停止 #83 正式 sweep，记录 `calibrated representation did not transfer to canonical cadence`；不得通过改 PPO 或增大普通 λ 掩盖。

## 2. 现行机制勘察结论

历史 E-033/E-035/E-038 的旧入口（`just lambda-identifiability` / `just objective-decomposition` / `just experiment reward ...`）已在架构合并中删除，现行等价机制：

| 环节 | 入口 | 实现 |
| --- | --- | --- |
| source batch 采集 | `just exp reward collect` | `scripts/experiments.py:34` → `experiments/reward/runner.py::collect` → `rl/rollout/collection.py::collect` |
| 离线 decomposition + Gate | `just exp credit run` | `experiments/credit/runner.py::run` / `measure`；gate 判定 `experiments/credit/decisions.py::evaluate_gate` |
| 离线分析/报告 | `just exp credit analyze` | `analysis/workflows.py::fixed` |

关键事实：

- **采集协议**（E-038 沿用）：`jobs/training/ppo_conservative`，16 scenarios（S/SC × map seeds 0–7）× 8 transitions/env = 128 transitions，update-0 rollout（无 optimizer/scheduler step），seed 0、replay 0、DDIM5。batch 目录含 `resolved_config.yaml`、`policy-initial.pt`、per-episode NPZ（`reward_substep_*` 审计，shape `[T, CLOSED_LOOP_EXECUTION_STEPS=5]`）、`training-batch.pt`、`sample_index.json`。
- **cadence**：`configs/components/ppo_training.yaml` 已 pin canonical cadence（0.1 s × 5 = 0.5 s），`ClosedLoopCadenceConfig`（`rl/config.py:141`）在 typed boundary 拒绝非 canonical 值。horizon 80 ≥ warmup 0 + 8×5=40，现采集协议 **k=5 兼容，无需改 rollout 语义**。
- **credit 机制**：`measure()` 对每个 arm 用 `reweight`/`energy_only_reward` 重组 per-substep objective，`aggregate_substep_rewards`（sum base/components、min gate）在线/离线共享；`rescore` 用 `scored_arrays` + `energy_score_from_fuel` 重算 progress/comfort/energy；`verify_original_components` 强制 base profile 重建与在线审计 parity（rtol 1e-6）。
- **arm 构造**：`reward_profile(base, λ)`（`rl/reward.py:44`）把 no-energy base 转成 energy profile（name `plannerrft_energy_v1`）。当 base 携带 calibrated-band energy 时，λ arm 与正式 named profile `plannerrft_energy_band_lamλ_v1` **目标等价**（weights 5/5/2/4/λ、E-034 校准、E-038 band 阈值、分母 16+λ）。λ 可从 `diagnostic_config.yaml` 的 arm weight 唯一恢复。
- **gate 统计口径**：runner 中 z-form 结果提升到 pair 顶层，`evaluate_gate` 的 endpoint cosine / RMSE / sign-flip 与 stress angular separation 均为 z-form actor-head 梯度（与 E-038 报告口径一致）。
- **分析契约**：`analysis/workflows.py::fixed` 要求 diagnostics.npz 中所有非 `__gradient_` 数组长度 == sample 数，自动生成分布（component variance 由此获得）。
- **E-038 基准数字**（k=1 / 0.1 s transition，对照用）：endpoint z-form head cosine **0.976536**、z-RMSE **0.771663**、sign-flip **0.257812**；stress angular separation 占 endpoint 比例：λ16 41.50%（cos 0.995945）、**λ64 74.04%（cos 0.987113）**、λ256 91.96%（cos 0.980144）。
- **冻结常数**：E-034 Progress/Comfort（`full_score_delta_m=1.7813475926717124`，comfort 4.157548461641585/3.0/111.70486995152065/0.5）已冻结于 `plannerrft_no_energy_calibrated_v1`；E-038 band 阈值 `46.37086372375488 / 48.7514030456543` ml/km 已冻结于 `plannerrft_energy_band_lam*_v1`。
- **现存 bug（已本地验证）**：`configs/experiments/reward/collection.yaml` 缺 `training.replay_id` override，`compose_job_config` 抛 `MissingMandatoryValue` —— 当前默认采集入口不可用，需随本任务修复。

## 3. 设计决策

| # | 决策 | 理由 |
| --- | --- | --- |
| D1 | 冻结阈值，不从新 batch 推导 calibration/band | Task 1A 检验的是「冻结 representation 是否 transfer」，重新推导检验的是另一个问题（新 batch 上的新 band）。现 credit runner 只有批内推导路径（`calibrate` + `apply_energy_band`），需新增 frozen band 支持（见 Step 1） |
| D2 | source batch 用 `components/reward=plannerrft_no_energy_calibrated_v1` 采集 | 正式 R0 arm；`verify_original_components` 直接对冻结校准做在线/离线 parity；audit 即 calibrated R0 |
| D3 | credit config：`calibration: null`、`energy_band: null`、`frozen_energy_band: {E-038 阈值}` | 承接 D1/D2：progress/comfort 校准来自 batch 自身的冻结 profile，energy 由 frozen band 节点显式给出 |
| D4 | arms = r0 / λ16 / λ64 / λ256 / energy_only（与 E-038 完全一致） | 直接 transfer 复核 E-038 Gate C；λ64 为优先 stress arm。**不加入 λ={1,2,4,8}**：小 λ 不是 1A 的 gate 条件，且 `evaluate_gate` 继承的 stress 单调性检查可能被小 λ 噪声虚假破坏；小 λ 的 objective/guidance effect 留给 Task 2 报告。不因 #83 的较小 λ 降低 gate（Issue 原文要求） |
| D5 | 采集 overrides pin E-039 candidate PPO control（lr=1.5e-4、epochs=1、mgn=0.5、target_kl=0.006、batch=minibatch=128、50 updates/scheduler steps） | lr/epochs/mgn 不影响 update-0 rollout 与离线梯度/优势，但 resolved config 需与 #83 冻结 PPO control 一致（provenance） |
| D6 | 新增诊断数组全部 per-sample 对齐 | `analysis/workflows.fixed` 契约（非梯度数组长度 == sample 数）；与 reward runner 现有 `reward_component_*` / `reward_safety_gate` 保存模式一致 |
| D7 | 实验编号 E-048；配置文件按实验命名 | 当前最高记录 E-047；per-experiment 命名使 resolved config/artifact 可唯一恢复协议 |

## 4. 实现步骤

### Step 1 — frozen band 支持（最小代码改动）

1. `src/eco_planner/reward/calibration.py`：
   - 新增 `FrozenEnergyBand`（strict frozen pydantic model）：`full_score_ml_per_km`、`zero_score_ml_per_km`（均 gt 0；validator 要求 zero > full，与 `EnergyRewardConfig` band 语义一致）；
   - 新增 `apply_frozen_energy_band(profile, band) -> PlannerRFTNoEnergyRewardConfig`：纯 profile 变换（energy.mode→`calibrated_band`、写入两阈值），不读取 episodes；
   - 从 `eco_planner.reward` `__init__` 导出。
2. `src/eco_planner/experiments/credit/config.py`：`CreditStudyConfig` 新增**必填可空**字段 `frozen_energy_band: FrozenEnergyBand | None`（沿用 `calibration`/`energy_band` 的显式 null 风格）；validator：与 `energy_band` 互斥（两者同时非空 → ValueError）。
3. `src/eco_planner/experiments/credit/runner.py`：`run()` 中 `energy_band` 分支后新增 `frozen_energy_band` 分支（`base = apply_frozen_energy_band(base, study.frozen_energy_band)`）。
4. 更新 4 个现有 credit 配置显式补 `frozen_energy_band: null`：`sensitivity.yaml`、`objectives.yaml`、`ablation.yaml`、`energy-band.yaml`。

### Step 2 — component variance + multi-substep reward diagnostics

`credit/runner.py::measure()` 从 rescored episodes 增加数组（全部 per-sample，长度 = transition 数）：

```text
substep_count                       # 每 transition 实际执行子步数（1–5，截断前缀）
reward_component_{ttc,progress,comfort,speed,energy}   # per-transition 分量和（multi-substep sum 聚合）
reward_safety_gate                  # per-transition min gate
```

component variance 由离线分析 `fixed()` 自动生成的分布（含 per_scenario）覆盖；raw/center/z advantage 与 actor-head/lon/lat gradient 已由现有 `diagnostic_variants` + `GRADIENT_GROUPS` 覆盖。per-substep 原始审计以 source batch NPZ 为权威，不在 comparison artifact 重复。

### Step 3 — 实验配置

1. **修复** `configs/experiments/reward/collection.yaml`：补 `training.replay_id=0`（现存 bug）。
2. 新增 `configs/experiments/reward/e-048-collection.yaml`：

```yaml
job: jobs/training/ppo_conservative
overrides:
- runtime.seed=0
- training.replay_id=0
- components/reward=plannerrft_no_energy_calibrated_v1
- components/resources=rtx_a4000
- ppo.learning_rate=1.5e-4        # E-039 candidate / #83 冻结 PPO control
- ppo.max_gradient_norm=0.5
- ppo.target_kl=0.006
- ppo.batch_size=128
- ppo.minibatch_size=128
- training.transitions_per_environment=8
- training.update_count=50
- ppo.scheduler_total_optimizer_steps=50
```

3. 新增 `configs/experiments/credit/e-048-identifiability-transfer.yaml`（结构 = `energy-band.yaml`，但冻结而非推导）：

```yaml
value_target_ddof: 0
advantage_forms: [raw, center, z]
credit_forms: [standard_gae]
quantiles: [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0]
calibration: null          # 冻结校准已由 source batch 的 calibrated profile 携带
energy_band: null
frozen_energy_band:        # E-038 冻结 band 阈值，不从本 batch 重新推导
  full_score_ml_per_km: 46.37086372375488
  zero_score_ml_per_km: 48.7514030456543
objective_gate:
  endpoint_max_actor_head_cosine: 0.99
  min_normalized_advantage_rmse: 0.1
  min_sign_flip_fraction: 0.05
  min_stress_fraction_of_endpoint_separation: 0.5
attribution_gate: null
arms:                      # 与 E-038 完全一致；λ64 为优先 stress arm
- {label: r0, weight: 0.0}
- {label: lambda_16, weight: 16.0}
- {label: lambda_64, weight: 64.0}
- {label: lambda_256, weight: 256.0}
- {label: energy_only, weight: energy_only}
```

### Step 4 — 测试（CPU，沙箱内）

1. `FrozenEnergyBand` 验证：zero ≤ full 拒绝；`frozen_energy_band` 与 `energy_band` 互斥拒绝；`CreditStudyConfig` 缺字段拒绝。
2. **arm 构造与 named profile 等价性回归**（关键契约测试）：从实际 yaml 加载 `plannerrft_no_energy_calibrated_v1` + frozen band → `reward_profile(base, 64)` 与 `plannerrft_energy_band_lam64_v1` 在 synthetic episode 上的 objective 逐值一致（offline 离线 arm ≡ 正式 named profile）。
3. `tests/training/test_fixed_batch.py`：
   - 更新 `test_credit_preserves_ppo_gradients_and_objective_identities` 的 config dict（补 `frozen_energy_band: None`）；
   - 把 `e-048-identifiability-transfer` 加入 `test_reward_and_credit_recompute_without_reference_artifacts` 的配置循环，并断言新数组（`substep_count`、`reward_component_*`、`reward_safety_gate`）存在且 per-sample 对齐、band-rescored energy 生效（synthetic batch 强度 46/48/47/49 ml/km → band 分数含两侧饱和）。
4. 采集配置组合回归（CPU-only）：`e-048-collection.yaml` 与修复后的 `collection.yaml` 均 compose + `parse_training_config` 通过，cadence/seed/reward profile 断言。
5. 运行：`just test-target tests/training/test_fixed_batch.py tests/training/test_credit_assignment.py tests/training/test_reward.py tests/configuration`、`just lint`、`just typecheck`；必要时 `just test`（smoke）。

### Step 5 — 正式运行（沙箱外，GPU + MetaDrive）

前置：当前改动 commit clean 并记录 hash（Issue Gate I 惯例：无 CI 时在实验记录登记本地测试命令与结果）。

```powershell
# 1) canonical k=5 source batch（16 episodes / 128 transitions，calibrated R0 在线审计）
just exp reward collect `
  --config configs/experiments/reward/e-048-collection.yaml `
  --output-dir outputs/studies/scalar-reward/e-048-issue83-task-1a-source-batch

# 2) 离线 identifiability transfer decomposition + Gate T1 判定
just exp credit run `
  --source-dir outputs/studies/scalar-reward/e-048-issue83-task-1a-source-batch `
  --config configs/experiments/credit/e-048-identifiability-transfer.yaml `
  --output-dir outputs/studies/scalar-reward/e-048-issue83-task-1a-identifiability-transfer
```

运行后核对：resolved config 的 `cadence`（0.1×5=0.5）、reward profile、seeds；`runtime_metadata.json` 的 initial policy hash 应与 E-038/E-040 一致（`049697739a…`）——不一致即停下检查（policy 架构变更会破坏 transfer 对照）；`summary.json` 的 `decisions.objective`。

### Step 6 — 判定与记录

- **Gate T1 判定**：读 `summary.json → decisions.objective`（endpoint cosine ≤ 0.99 AND (RMSE ≥ 0.10 OR sign-flip ≥ 0.05) AND ≥1 stress arm ≥50%；同时继承 `evaluate_gate` 的 stress 单调性/总体增加检查，与 E-038 同口径）。
- 通过 → Task 1A 完成，进入 Task 1B（effective-update transfer，另行规划）；失败 → 停止 #83 正式 sweep，按 `attribution` 字段记录失败模式，不通过改 PPO 或增大 λ 掩盖。
- 新增 `docs/experiments/records/e-048-issue83-task-1a-objective-identifiability-transfer.md` 并更新 `docs/experiments/README.md` 索引；记录须包含 Issue Task 5 的 provenance 清单（commit/环境/resolved config/checkpoint identity/seeds/cadence/commands/artifact path/测试结果/failure 状态/可支持与不可支持的结论）与 **E-038(k=1) vs E-048(k=5) 对照表**（endpoint 与 stress 指标逐项）。
- 文档写回（仅在事实变化时）：`docs/agents/contracts/experiments.md`（credit 配置新增 `frozen_energy_band` 节点与新的 per-sample 诊断数组契约）。
- Issue #83 评论：Gate T1 结果 + 指向实验记录。

## 5. 风险与检查点

| 风险 | 检查/应对 |
| --- | --- |
| k=5 下 batch 强度分布不同，endpoint separation 可能变化（甚至 gate 失败） | 这正是 transfer gate 要检验的对象；如实记录，不调整阈值（阈值是 #94 Gate C 冻结口径） |
| initial policy hash 与 E-038 不一致 | 运行后核对；不一致则暂停并排查（不应发生——架构未变） |
| k=5 的 per-transition reward ≈ 5× k=1 scale、component sums 不再被 [0,1] 界定 | z-form 统计不受影响；记录中注明与 E-038 的 raw 数值不可直接比，只比 gate 判定 |
| substep 截断（terminal/truncation 前缀 1–5） | `substep_count` 数组 + source batch per-substep 审计可解释；ADR 0039 语义 |
| 采集/credit 运行需要 GPU+仿真 | 按沙箱要求申请沙箱外执行；不重建环境 |

## 6. Non-goals

- 不重新搜索或修改 PPO 超参、reward representation、band 阈值、cadence；
- 不做 λ={1,2,4,8} 的 identifiability 评定（Task 2 dose-response 范围）；
- 不启动 Task 1B/1C 或任何正式训练；
- 不做历史 artifact 迁移或 E-035/E-038 旧 batch 复用（旧 batch 无 substep 审计与 cadence 节点，已验证不可用）。
