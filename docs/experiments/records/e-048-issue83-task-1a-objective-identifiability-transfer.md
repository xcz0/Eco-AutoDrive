# E-048 Issue #83 Task 1A：objective identifiability transfer 到 canonical k=5 cadence

[返回实验索引](../README.md) · [Task 1A 执行计划](../../research/issue83-task-1a-identifiability-transfer-plan.md) ·
[E-038 源实验](e-038-issue94-task-e-energy-representation.md) · [E-039 PPO control](e-039-issue94-task-f-effective-update-region.md)

**日期 / 类型 / 目的**：2026-09-24 / 正式离线固定批次诊断 / 完成 Issue #83 Task 1A：**不重新搜索
reward/PPO**，只验证 #94 冻结的 reward representation（E-034 calibration + E-038 calibrated-band
阈值）在 ADR 0039 canonical cadence（k=5 × 0.1 s = 0.5 s decision interval）的 MDP 时间语义下仍可
辨识。协议：在 canonical k=5 rollout 上采集新的 fixed source batch；使用冻结 calibration / band
阈值，不从新 batch 重新推导；离线比较 calibrated R0、finite stress λ、Energy-only endpoint；保存
raw/center/z advantage、actor-head/lon/lat 梯度、component variance 与 multi-substep reward
diagnostics。本实验只执行 actor backward，不执行 optimizer/scheduler step，不选择训练 reward。

**代码 / 环境**：Git commit `eaafe5bf911390ef3263bea808aadc40215068a6`（branch `main`，
`git_status_short=[]` clean）；上游源码 `a3a621f0…`；EMA 276 tensors / 6,042,628 parameters。
Windows-10、Python 3.10.20、PyTorch 2.12.1+cu126、CUDA:0（RTX A4000）、Lightning 2.6.6、
MetaDrive 0.4.3；rollout `bf16-mixed`，actor backward float32（无 rollout autocast）。

**进入条件**：E-038 Gate C（k=1 / 0.1 s transition）PASSED（endpoint z-head cosine 0.976536、
sign-flip 0.257812、RMSE 0.771663；λ64/λ256 达 endpoint 分离 74.04%/91.96%）。Task 1A 检查该冻结
表示在 k=5 时间语义下是否 transfer。

## 批次来源与冻结契约

canonical k=5 source batch（`e-048-issue83-task-1a-source-batch/`）：`jobs/training/ppo_conservative`，
`components/reward=plannerrft_no_energy_calibrated_v1`，16 scenarios（S/SC × map seeds 0–7）×
8 transitions/env = **128 transitions / 16 episodes**，update-0 rollout（无 optimizer/scheduler
step），runtime seed 0、replay 0、DDIM5。冻结 PPO control 由 overrides pin（E-039 candidate：
lr=1.5e-4、epochs=1、mgn=0.5、target_kl=0.006、batch=minibatch=128、update_count=50、scheduler=50）；
update-0 rollout 与离线梯度/优势不依赖这些值，pin 只为 provenance 一致。

- cadence：`simulator_step_s=0.1`、`closed_loop_execution_steps=5`、`decision_interval_s=0.5`。
- 冻结校准：E-034 `full_score_delta_m=1.7813475926717124`、comfort
  `4.157548461641585 / 3.0 / 111.70486995152065 / 0.5`（随 source batch 的 calibrated profile
  携带，`verify_original_components` 在线/离线 parity rtol 1e-6）。
- 冻结 band：E-038 `full=46.37086372375488`、`zero=48.7514030456543` ml/km，由 credit 配置
  `frozen_energy_band` 显式给出，**不从新 batch 推导**（`calibration: null`、`energy_band: null`）。
- initial policy hash `049697739ab5d6e7bd212938bbac82c35215eb9ed14c02557952098a9718d05d`、
  frozen planner hash `6a014ec56d24ae0b501779a98e61a1b748e26b75d328184286539d7fe10ef55a`
  与 E-038/E-040 一致 → policy 架构未变，transfer 对照有效。

**命令 / 配置**：

```powershell
just exp reward collect `
  --config configs/experiments/reward/e-048-collection.yaml `
  --output-dir outputs/studies/scalar-reward/e-048-issue83-task-1a-source-batch
just exp credit run `
  --source-dir outputs/studies/scalar-reward/e-048-issue83-task-1a-source-batch `
  --config configs/experiments/credit/e-048-identifiability-transfer.yaml `
  --output-dir outputs/studies/scalar-reward/e-048-issue83-task-1a-identifiability-transfer
```

credit arms 与 E-038 完全一致：`r0 / λ16 / λ64 / λ256 / energy_only`（standard GAE、raw/center/z）；
objective gate 复用 #94 Gate C primary threshold（z-form / 实际 PPO preprocessing）。λ64 为优先
stress arm；不加入 λ={1,2,4,8}（Task 2 dose-response 范围）。

## 计算语义与 multi-substep 诊断

credit runner 对每个 arm 用 `reweight` / `energy_only_reward` 从源 batch 的 per-substep 审计重组
per-transition objective；`frozen_energy_band` 只把 energy 表示切到 `calibrated_band` 并写入冻结阈值，
`rescore` 从 `executed_fuel_proxy_step_energy_ml` + `step_distance_m` 重算 energy 分量。全部 arm 复用
同一 component scores（energy 除外）、safety gate、policy context、动作、old log-prob、critic
current/next value 与 episode boundary；每 arm 在 raw/center/z 下各做一次 full-batch
`loss_objective` backward，无 critic/entropy/clipping/optimizer/scheduler step，backward 后 policy
hash 不变（`policy_unchanged=true`、`optimizer_steps=0`）。

新诊断数组（`diagnostics.npz`，132 个数组全部有限，长度 == sample 数 128）：arm 无关单列
`substep_count`；每 arm 的 `{arm}__reward_component_{ttc,progress,comfort,speed,energy}`
（multi-substep sum 聚合）与 `{arm}__reward_safety_gate`（min gate）。k=5 下全部 128 transitions
的 `substep_count=5`（sum 640，无 terminal/truncation 截断）。component variance 由离线
`analysis.workflows.fixed` 的分布（含 per_scenario）覆盖；per-substep 原始审计以 source batch NPZ
为权威。每个 arm 的 normalized advantage 全批 std 在 `1e-6` 内等于 1、mean 0。

band energy（per substep）mean `0.319100`、std `0.269696`；满分（=1）饱和比例 `0.0%`、零分饱和
`11.7%`——k=5 执行强度分布整体高于 E-038/k=1（E-038：mean 0.447325、std 0.305853，两侧各约
10.16%），本次 none-below-full、仅高消耗侧饱和。

## Gate T1 裁定

```text
Verdict: PASSED（passed = true，failure_reasons = []，attribution = null）
```

- endpoint actor-head cosine `0.910851 ≤ 0.99` ✓
- normalized-advantage RMSE `0.664142 ≥ 0.10` 且 sign-flip `0.187500 ≥ 0.05` ✓
- stress 轨迹单调、总体增加；λ16 `58.79%`、λ64 `85.29%`、λ256 `95.89%` ≥ 50% ✓

**Endpoint R0 vs Energy-only（z-form）**：Pearson/Spearman `0.777721 / 0.819550`、sign-flip
`0.187500`、advantage RMSE `0.664142`、actor-head cosine `0.910851`、norm ratio `2.380373`；
lateral/longitudinal head cosine `0.227853 / 0.986842`。raw 形式 cosine `0.073628`（sign-flip 0、
RMSE 11.03545），center 形式 cosine `0.910851`——分离在 raw 与 z 下都存在，不是 normalization
压缩伪影，也不是正缩放可解释的（正缩放对 z-advantage 严格不变）。

## E-038(k=1) vs E-048(k=5) 对照

| 指标 | E-038 k=1 | E-048 k=5 |
| --- | ---: | ---: |
| endpoint z head cosine | 0.976536 | **0.910851** |
| endpoint z sign-flip | 0.257812 | 0.187500 |
| endpoint z RMSE | 0.771663 | 0.664142 |
| endpoint z Pearson | 0.699924 | 0.777721 |
| endpoint raw head cosine | 0.973894 | 0.073628 |
| endpoint center head cosine | 0.976536 | 0.910851 |
| λ16 head cosine | 0.995945 | 0.968881 |
| λ16 占 endpoint 分离 | 41.50% | **58.79%** |
| λ64 head cosine | 0.987113 | 0.934879 |
| λ64 占 endpoint 分离 | 74.04% | **85.29%** |
| λ256 head cosine | 0.980144 | 0.917924 |
| λ256 占 endpoint 分离 | 91.96% | **95.89%** |
| λ16 / λ64 / λ256 z-Pearson | 0.947023 / 0.827206 / 0.740861 | 0.968751 / 0.885355 / 0.814515 |
| λ16 / λ64 / λ256 z-sign-flip | 0.1094 / 0.1953 / 0.2344 | 0.0391 / 0.1172 / 0.1563 |
| λ16 / λ64 / λ256 z-RMSE | 0.3242 / 0.5856 / 0.7171 | 0.2490 / 0.4770 / 0.6067 |
| band energy per substep mean ± std | 0.447325 ± 0.305853 | 0.319100 ± 0.269696 |
| band 两侧饱和（full/zero） | ~10.16% / ~10.16% | 0.0% / 11.7% |

k=5 下 endpoint 与所有 finite λ arm 的 head cosine 全面低于 k=1，angular separation 更大；λ16 在
k=5 直接越过 50% 门槛（E-038 中仅 41.5%）。冻结 representation 在 canonical cadence 下不仅
transfer，且可辨识度优于 k=1。raw/center/z 三种形式下 k=5 的绝对数值与 k=1 不可直接逐值比较
（per-transition reward ≈ 5× scale、component sums 不再被 [0,1] 界定），只比 gate 判定与方向。

## 验证与产物

- 代码级测试：`tests/training/test_reward.py`、`test_credit_assignment.py`、`test_fixed_batch.py`、
  `tests/configuration/test_scalar_reward.py` 及 `tests/training` + `tests/configuration` 全量
  **312 passed**（含 `FrozenEnergyBand` 校验、frozen/energy band 互斥、credit 缺字段拒绝、
  offline arm ≡ named `plannerrft_energy_band_lam64_v1` objective 逐值一致、e-048 采集配置
  compose + canonical cadence 断言、e-048 credit 端到端链与新诊断数组契约）。Ruff 通过；
  Pyright 0 errors。
- 正式产物只读核验：132 数组全部有限、128 样本、每 arm normalized advantage std 1、policy hash
  前后一致（`04969773…`，与源 batch 一致）、`optimizer_steps=0`、band 阈值与冻结值逐位一致。

产物目录（均 git ignored）：

- `outputs/studies/scalar-reward/e-048-issue83-task-1a-source-batch/`：`resolved_config.yaml`、
  `policy-initial.pt`、`runtime_metadata.json`、`training-batch.pt`、per-episode NPZ（含
  `reward_substep_*` 审计，shape `[T,5]`）、`sample_index.json`、`summary.json`。
- `outputs/studies/scalar-reward/e-048-issue83-task-1a-identifiability-transfer/`：
  `report.md` / `summary.json`（arm/pair/Gate T1 全量统计）、`diagnostics.npz`（advantage 三形式、
  逐参数组梯度、per-arm reward component / safety gate、`substep_count`）、`analysis.json`、
  `diagnostic_config.yaml`（`frozen_energy_band` 阈值）、`runtime_metadata.json`、`sample_index.json`。

## Issue Task 5 provenance 清单

- commit `eaafe5b`、branch `main`、`git_status_short=[]`（clean 运行）；上游 `a3a621f0…`；
- 环境：Windows-10 / Python 3.10.20 / torch 2.12.1+cu126 / CUDA:0 RTX A4000 / Lightning 2.6.6 /
  MetaDrive 0.4.3；
- resolved config：source batch `resolved_config.yaml` + credit `diagnostic_config.yaml`（均在产物目录）；
- checkpoint identity：initial policy `049697739a…` / frozen planner `6a014ec5…`（与 E-038/E-040 一致）；
- seeds：runtime/training seed 0、replay 0、16 scenarios S/SC seeds 0–7；DDIM5；
- cadence：0.1 s × 5 = 0.5 s（`ClosedLoopCadenceConfig` typed boundary 校验）；
- commands：见上；artifact path：见上；
- 测试结果：312 passed、Ruff/Pyright clean；failure 状态：无（Gate T1 PASSED，`failure_reasons=[]`）。

## 结论边界

**支持**：在 canonical k=5 cadence、该 initial policy、该 no-traffic 固定 batch、E-034/E-038 冻结
calibration/band 下，calibrated R0 与 calibrated-band Energy-only 在当前 PPO 预处理（z-form）下
可辨识（endpoint head cosine 0.910851、sign-flip 18.75%），finite λ stress 单调且 λ64/λ256 达
endpoint 分离 85%/96%；冻结 representation **transfer 到 k=5 且可辨识度优于 E-038/k=1**。

**不支持**：不证明训练后行为分离、不证明节能、不选择训练 reward/λ、不构成真实车辆能耗建模；
k=5 raw/center/z 绝对数值与 E-038/k=1 不可逐值比较（只比 gate 判定与方向）；本记录不修改
PPO、reward representation、band 阈值或 cadence。Task 1A 通过后进入 Task 1B
（effective-update transfer），不启动任何正式训练或 sweep。
