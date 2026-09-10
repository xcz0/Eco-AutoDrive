# E-038 Issue #94 Task E：Energy objective 的 calibrated efficiency-band 表示

[返回实验索引](../README.md) · [源实验 E-035](e-035-issue94-task-c-objective-decomposition.md) ·
[归因实验 E-036](e-036-issue94-task-c4-critic-gae-common-term-ablation.md) ·
[门 D 实验 E-037](e-037-issue94-task-d-guidance-control-authority.md)

**日期 / 类型 / 目的**：2026-09-10 / 正式离线固定批次诊断 / 完成 Issue #94 Task E：只修改
Energy objective 的表示（新增 calibrated efficiency-band 模式），不触碰 Progress/Comfort
分量、PPO optimizer 与 E-034 校准规则，使当前训练分布下 R0 与 Energy-only endpoint 在
**实际 PPO 预处理（full-batch z-normalization）之后**可辨识（改变 transition 排序 /
normalized advantage / actor 梯度方向，而非仅 raw reward 幅值），并按 Task C 完整协议重跑
objective decomposition 与 Gate C 判定。本实验只执行 actor backward，不执行 optimizer
step，不选择训练 reward。

**代码 / 环境**：基线 `fe097cd` 加本次未提交改动（表示实现 + 测试 + 实验配置）；
`runtime_metadata.json` 和 `tracked_diff.patch` 记录运行时仓库状态，`source/` 保存实际执行
的模块与 CLI。Windows、Python 3.10.20、PyTorch 2.12.1+cu126、CUDA:0（RTX A4000）；actor
backward 为 float32，无 rollout autocast。

**进入条件**（Task E 为条件任务）：Gate D PASSED（E-037）；Gate C FAILED（E-035
`objective_batch_collinearity`，E-036 强化归因为 `reward/batch collinearity`）。两者均已在
Issue #94 记录，本实验为响应。

**批次来源**：E-035 的源 batch 目录为重构前格式（无 `kind: fixed-batch`），无法被当前
`load_fixed_batch` 加载，因此按 E-033/E-035 同一协议重新采集一批 update-0 作为源 batch
（`e-038-issue94-task-e-source-batch/`）：R0 arm、training seed 0、replay 0、S/SC × map
seeds 0–7、每场景 8 transitions，共 16 episodes / 128 transitions。initial policy hash
`04969773…` 与 frozen planner hash `6a014ec5…` 与 E-033/E-035 一致；采集 seed 确定，
baseline 统计复现 E-035 记录值 6 位以上（本 batch 校准值 `full_score_delta_m=1.7807237`、
`longitudinal=4.1442301`、`jerk=112.0860999` 与 E-035 逐位一致；E-035 记录的 z-pearson
0.997583 / head cosine 0.999713 / RMSE 0.069255 / λ16 cosine 0.999973 均在预分析中复现）。
与 E-035 相同，本记录声称同协议、同 seed、同 hash 的统计复现，不声称逐位复用。

## 表示设计与选择规则（预登记）

**为什么纯缩放无效**：当前 energy score 为 `exp(-I/50)`，I = 执行燃油强度。本 batch 上
`I = 32.5·exp(0.01·v_kmh)` mL/km 精确成立（最大差 2.5e-6），且 I ∈ [44.5, 50.8]（mean
47.72 ± 1.01）——在该窄区间内 `exp(-I/c)` 对 I 近似仿射（score std 仅 0.0078），因此任何
对 `reference_ml_per_km` 的重标定只产生近似 per-arm 仿射变换，transition 排序不变；同时
full-batch z-normalization 对 advantage 的正缩放严格不变（既有测试
`tests/training/test_lambda_identifiability.py::test_positive_scale_is_removed_by_full_batch_normalization`
覆盖该数学）。这就是 E-035 共线无法用 Task B 式重标定修复的机制。

**新表示**：双侧饱和的 calibrated efficiency band（非仿射、在分布内进入饱和段、改变排序
间距）：

```text
score = clip((I_zero − I)/(I_zero − I_full), 0, 1)   若 energy_distance_valid，否则 0.0
I_full / I_zero = 固定源 batch 执行强度分位数（P_full / P_zero），冻结进配置
```

校准/缩放参数全部由预固定校准数据（固定源 batch）确定并冻结，配置携带 expected 冻结值
守卫（确定性推导，`rtol=1e-6`）；无任何 per-PPO-batch 自适应重标定。不涉及真实车辆能耗
建模，保持 smoke-study 边界。R0（no-energy anchor）不动；Progress/Comfort 校准与 PPO
optimizer 不动。

**预登记选择规则**（正式运行前在 scratch 预分析中声明，原文）：

> candidates ordered by increasing saturation; select the first satisfying z-pearson
> ≤ 0.95 AND actor-head cosine ≤ 0.985 AND sign-flip ≥ 0.05 (all on the fixed batch,
> standard GAE, z-form).

候选对（full,zero = 强度分位数；endpoint 指标为校准 R0 vs band Energy-only，z 形式，
standard GAE；scratch 预分析，复用仓库完整管线 load → calibrate → rescore → arms → GAE →
z-norm → actor_backward，policy hash 校验不变）：

| pair | pearson | sign-flip | RMSE | head cosine | λ16 cosine | λ16 占 endpoint 分离 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| (P10,P90) | 0.699924 | 0.25781 | 0.7717 | **0.976536** | 0.995945 | 0.4150 |
| (P15,P85) | 0.694714 | 0.26562 | 0.7783 | 0.978418 | 0.995588 | 0.4515 |
| (P20,P80) | 0.675058 | 0.28125 | 0.8030 | 0.979400 | 0.995663 | 0.4582 |
| (P25,P75) | 0.665282 | 0.27344 | 0.8150 | 0.979599 | 0.995723 | 0.4573 |

四个候选全部满足规则 → 按最小饱和度选中 **(P10, P90)**：`I_full = 46.37086372375488`、
`I_zero = 48.7514030456543`（128 transitions 全部 distance-valid；v = 10.66 ± 0.59 m/s）。

**实现面**（最小改动，默认行为不变）：`EnergyRewardConfig` 新增
`mode: "reference_exponential" | "calibrated_band"` 与 band 阈值字段（band 模式必填、
要求 zero > full > 0，exp 模式禁止携带）；`calibrated_band_score` 纯函数；fixed-batch
`rescore` 在 profile 为 band 模式时从审计的 `executed_fuel_proxy_ml_per_km` +
`energy_distance_valid` 重算 `reward_component_energy`（exp 模式路径不动）；decomposition
配置新增可选 `energy_band` 节（分位数 + 冻结 expected 值 + 容差），runner 在校准后推导
阈值、守卫、切换 profile energy 模式并重打分。arms（R0 / λ / Energy-only）经审计分量自动
继承重打分的 energy；R0（no-energy 权重）不受影响。

**配置 / 命令**：

```powershell
just experiment reward fixed-batch collect `
  --output-dir outputs/studies/scalar-reward/e-038-issue94-task-e-source-batch
just experiment reward objective-decomposition run `
  --source-dir outputs/studies/scalar-reward/e-038-issue94-task-e-source-batch `
  --output-dir outputs/studies/scalar-reward/e-038-issue94-task-e-objective-decomposition `
  --config configs/experiments/reward/objective-decomposition/e-038-task-e.yaml
```

`configs/experiments/reward/objective-decomposition/e-038-task-e.yaml` = `diagnostic.yaml`
的全部协议（λ={16,64,256}、分位、E-034 冻结校准守卫、Gate C 阈值）+ `energy_band` 节。
运行时推导值与冻结 expected 值逐位一致（`46.37086372375488` / `48.7514030456543`）。

**计算语义**：与 E-035 相同——arms 为校准 R0（共享权重 5/5/2/4，分母 16）、
`λ={16,64,256}`（分母 16+λ，energy 分量为 band 重打分值）与 Energy-only endpoint
（`reward = safety_gate × band_energy`，仅作 objective endpoint）。全部 arm 复用同一批
component scores（energy 除外）、safety gate、policy context、动作、old log-prob、critic
current/next value 和 episode boundary；每 arm 在 raw / center-only / z 三种纯诊断
advantage 形式下各做一次 full-batch `loss_objective` backward；不含 critic/entropy loss，
无 clipping/optimizer/scheduler step，backward 后 policy hash 不变。

## C1 Endpoint：R0 vs Energy-only（band）

band 分量统计：mean `0.447325`、std `0.305853`（exp 表示为 0.385 ± 0.0078），两端各
~10.16%（13/128）饱和。R0 arm 与 E-035 逐位一致（reward 0.785863 ± 0.050993、raw
advantage 3.108009 ± 1.437447、head 梯度范数相同）——energy 表示切换未影响 no-energy
anchor。

| Arm | Reward mean ± std | Raw advantage mean ± std | Head grad norm raw/center/z |
| --- | ---: | ---: | --- |
| R0（校准） | 0.785863 ± 0.050993 | 3.108009 ± 1.437447 | 0.18313 / 0.13939 / 0.09697 |
| λ=16 | 0.616594 ± 0.148008 | 2.473973 ± 1.136711 | 0.14699 / 0.11853 / 0.10427 |
| λ=64 | 0.515032 ± 0.242095 | 2.093552 ± 1.039543 | 0.12609 / 0.10662 / 0.10256 |
| λ=256 | 0.467239 ± 0.287056 | 1.914530 ± 1.023134 | 0.11657 / 0.10123 / 0.09894 |
| Energy-only（band） | 0.447325 ± 0.305853 | 1.839938 ± 1.022303 | 0.11268 / 0.09903 / 0.09687 |

| Endpoint 形式 | Pearson | Spearman | Sign flip | Advantage RMSE | Head cosine | Head norm ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| z（当前预处理） | 0.699924 | 0.684044 | 0.257812 | 0.771663 | **0.976536** | 0.998952 |
| raw | 0.699924 | 0.684044 | 0 | 1.629123 | 0.973894 | 0.615321 |
| center-only | 0.699924 | 0.684044 | 0.257812 | 1.022759 | 0.976536 | 0.710448 |

lateral/longitudinal head cosine（z 形式）分别为 0.999737/0.999992：head 整体分离来自两
个分量头梯度范数结构的相对变化（norm ratio 0.2797/0.7475），而非任一单头内部方向。
shared-trunk 梯度仍因 actor-head 零初始化严格为 0，记 `null`，不作为证据。

## C3 λ stress 轨迹

| λ | Head cosine vs R0 | Angular separation (rad) | 占 endpoint 分离比例 | Head norm ratio | z-Pearson | Sign flip | z-RMSE |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 0.995945 | 0.090084 | 41.50% | 1.075309 | 0.947023 | 0.109375 | 0.324233 |
| 64 | 0.987113 | 0.160713 | 74.04% | 1.057651 | 0.827206 | 0.195312 | 0.585566 |
| 256 | 0.980144 | 0.199608 | 91.96% | 1.020286 | 0.740861 | 0.234375 | 0.717098 |

λ 增大时 angular separation 单调增加；λ=64/256 分别达到 endpoint 分离的 74.04%/91.96%
（≥50%）。预分析已预登记的风险——λ16 单独仅 ~41.5%——成立，但 Gate C 只要求至少一个
有限 stress arm 达到 ≥50%，λ64/λ256 满足。

## Gate E 裁定

```text
Verdict: PASSED（gate_c_passed = true，failure_reasons = []，attribution = null）
```

- endpoint actor-head cosine `0.976536 ≤ 0.99` ✓
- normalized-advantage RMSE `0.771663 ≥ 0.10` 且 sign-flip `0.257812 ≥ 0.05`（两项远超
  阈值，endpoint identifiable）✓
- stress 轨迹单调、总体增加、λ64=74.04% 与 λ256=91.96% ≥ 50% ✓

**防作弊条款**：本次通过不依赖关闭 advantage normalization，也不依赖把 reward 乘大常数。
band 是对执行强度的非仿射（双侧饱和）函数，在分布内改变 transition 排序与间距；正缩放
类变换在数学上不可能产生本次测得的指标——per-arm advantage 正缩放对 z-advantage 严格
不变（`test_positive_scale_is_removed_by_full_batch_normalization`），保持 Pearson=1、
sign-flip=0，而实测 z-Pearson 0.699924、sign-flip 0.257812；endpoint 在 raw 形式同样可分
（0.973894 ≤ 0.99），即分离不是 normalization 的伪影，也不是被 normalization 压缩后残留
的方向差。

**C4（critic/GAE ablation）未重跑**，原因：(1) Gate C 本次 PASSED 且无 failure reason，
E-036 式归因消解不再有未决问题——decomposition 产物自带的三形式对照已显示 raw/center/z
均可分（0.9739/0.9765/0.9765），排除 `normalization_suppressed_identifiability`；(2) C4
runner 按校准 profile（无 band 节）自算 arms 并与 reference 的记录数组逐项互检，对 band
reference 的 energy_only arm 按构造必然失配，重跑需扩展 C4 协议本身，超出 Task E
"只改 Energy 表示"的边界。此为记录性缺口，不影响 Gate E 判定（判定输入为 Task C
objective-decomposition 协议本身）。

## 验证与产物

新增/扩展测试 9 项全部通过（`tests/training/test_reward.py` band 数学/分量/配置校验，
`tests/training/test_objective_decomposition.py` band rescore 与 arms 继承、EnergyBand
配置校验、`apply_energy_band` 推导与冻结守卫，`tests/configuration/test_fixed_batch.py`
含 band 节的完整离线链 run_decomposition 集成）；三文件合计 35/35 通过；Ruff 通过；本次
修改模块 Pyright 0 errors（`just typecheck` 在未触碰的 `analysis/guidance.py`、
`rl/tracking.py` 报告环境性 scipy/mlflow 缺包导入错误，与本实验无关、修改前已存在）。

正式产物只读核验：138 个数组全部有限、128 样本、每 arm（5 arms）normalized advantage
sample std 在 `1e-6` 内等于 1、center/z 符号结构相同、policy hash 前后一致
（`04969773…`，与源 batch 一致）、optimizer steps 为 0、band 阈值推导值与冻结值逐位一致。

产物目录：`outputs/studies/scalar-reward/e-038-issue94-task-e-objective-decomposition/`
与 `e-038-issue94-task-e-source-batch/`（均 git ignored）：

- `report.md` / `summary.json`：arm/pair/gate 全量统计与裁定（含
  `energy_band_verification`）。
- `diagnostics.npz` / `sample_index.json`：逐 transition reward/advantage（三种形式）、
  逐 form 逐参数组梯度向量与配对差异。
- `calibration_verification.json` / `diagnostic_config.yaml` / `resolved_config.yaml`
  （band 模式与阈值见 `calibrated_reward.energy`）/ `runtime_metadata.json` /
  `tracked_diff.patch` / `source/`：运行来源与实际执行的源码快照。

**结论边界**：以上结论只支持该 initial policy、该 no-traffic 固定 batch、该校准配置与
预冻结的 (P10, P90) band 阈值；证明的是 band 表示使 Energy objective endpoint 在当前
PPO 预处理下**可辨识**（z 形式 head cosine 0.9765、sign-flip 25.8%），不证明训练后行为
分离、不证明节能、不选择训练 reward 或 λ、不构成真实车辆能耗建模。λ16 单独的分离占比
（41.5%）提示小 λ 下实际训练信号仍可能偏弱，留给后续任务判断。按 Issue #94 顺序，Task
F–H 未执行；未修改全局 reward 默认配置（`plannerrft_*_v1.yaml` profiles 未动），未关闭
Issue #94。
