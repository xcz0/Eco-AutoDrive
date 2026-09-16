# E-040 Issue #94 Task G：objective positive control（R0 vs Rstress 闭环训练）

[返回实验索引](../README.md) · [前置 E-039 Task F](e-039-issue94-task-f-effective-update-region.md) · [reward 对来源 E-038 Task E](e-038-issue94-task-e-energy-representation.md)

**日期 / 类型 / 目的**：2026-09-11 / 正式配对闭环训练对照 / 在 E-039 冻结的有效 PPO
区域内，用 E-038 可辨识 reward 对（校准 R0 vs band-energy λ=64 Rstress）做 matched
闭环 PPO 训练，判定 Issue #94 Task G 的 positive control gate：objective 差异能否
经 PPO 传导为闭环行为的可复现、方向正确的分离。

**裁定：Gate G FAILED（仅 c4 方向条件失败）。** c1/c2/c3/c5 全部通过：guidance
分布可分、闭环行为可分且双 seed 方向复现、无 collapse。但分离方向与 energy 目标
**相反**——双 seed 的 Rstress−R0 paired delta 中 energy intensity 与 mean speed 均为
正（+0.41%/+0.53% ml/km、+1.06%/+1.41% m/s，均超 0.1% matched 噪声界）：Rstress
训练出的 policy 比 R0 **更快且单位里程能耗更高**。λ=64 band-energy objective 在该
协议下没有起到节能 objective 的作用；这是一个真实、可复现的负结果，不是机械故障。

## 运行来源与固定协议

- 代码基线为 `scalar-energy-reward` 分支未提交改动（Task G runner、reward profile
  configs、组合与诊断层，以及本次运行中发现的三处实现修复，见下文）；每个 run
  目录的 `runtime_metadata.json` 与 `resolved_config.yaml` 保存正式训练时的来源。
- Windows、Python 3.10、PyTorch 2.12.1+cu126、MetaDrive 0.4.3，GPU NVIDIA RTX A4000
  （与 E-039 同机）。
- **Arms（E-038 冻结的 identifiable reward 对）**：
  - `r0`：`plannerrft_no_energy_calibrated_v1`——E-034 冻结校准的
    Progress/Comfort（`full_score_delta_m=1.7813475926717124`、comfort
    long `4.157548461641585` / lateral `3.0` / jerk `111.70486995152065` /
    yaw `0.5`），无 energy 权重（分母 16）。
  - `rstress`：`plannerrft_energy_band_lam64_v1`——同套校准分量，energy 分量换为
    E-038 冻结的 `calibrated_band` 双侧饱和表示
    （`band_full_score_ml_per_km=46.37086372375488`、
    `band_zero_score_ml_per_km=48.7514030456543`），`weights.energy=64`（分母 80）。
    λ=64 是 E-038 中达到 R0→Energy-only 梯度角分离 ≥50% 的最小 stress λ
    （endpoint 分离 74.04%；R0-vs-λ64 actor-head cosine 0.987113、sign-flip
    fraction 0.195312）。
- **PPO 冻结自 E-039 Task F 选定 config**：lr=1.5e-04、epochs=1、
  max_gradient_norm=0.5、batch=minibatch=128（16 episodes × 8 transitions）、
  `target_kl=0.006`、`update_count=50`。
- **Matched pairing**：training seeds {0, 1}；同 seed 两 arm 共享同一 initial
  policy（runner 硬校验 hash：seed0 `049697739a…`、seed1 `feee6d85b…`）、
  matched noise/action seed streams、场景池、update 预算与 held-out 矩阵。
- Initial held-out 聚合值双 seed 逐位一致（如 speed 9.998465、energy 47.047213
  ml/km）：policy 的 actor head 权重零初始化、bias 为确定性
  （`rl/policy/model.py` `_initialize_symmetric_actor`），seed 只进入
  value-head/trunk 参数——hash 不同但初始 actor 行为相同，这是 pairing 校验通过
  且 paired 差分从同一起点出发的结构性原因。

```powershell
just experiment training positive-control run --output-dir outputs/studies/scalar-reward/e-040-issue94-task-g-objective-positive-control
```

配置清单：`configs/experiments/training/objective-positive-control.yaml`（arms、
seeds、Gate G 阈值预登记）；实现位于
`src/eco_planner/experiments/training/objective_positive_control/`。

## Gate G 判定与结果

五条件（阈值见 yaml，预登记）：c1 probe guidance-mean paired RMS ≥0.01 且跨 seed
方向一致；c2/c3 separation 指标双 seed paired relative ≥0.1%（E-031 matched 噪声
界）且同向；c4 energy 或 speed delta 双 seed 均为负（energy-heavy arm 应向更慢/
更节能方向移动）；c5 无 collapse（arrive_dest 保持、collision+OOR ≤2、无 stopped、
route completion 保持）。

### 训练机械健康（4 runs 全部 completed，均在 E-039 有效区域内）

| run | MC post-update KL median / max | ratio 变化 | probe RMS | min α / β | boundary mass | pre-clip grad median | collision / OOR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| r0-seed-0 | 1.809e-06 / 3.017e-05 | 1.647e-03 | 0.0339 | 1.908 / 2.000 | 0.022 | 26.1 | 0 / 0 |
| r0-seed-1 | 2.999e-06 / 2.411e-05 | 2.300e-03 | 0.0449 | 1.906 / 1.903 | 0.024 | 30.1 | 0 / 0 |
| rstress-seed-0 | 2.391e-06 / 3.159e-05 | 2.791e-03 | 0.0428 | 1.967 / 1.886 | 0.021 | 17.4 | 0 / 0 |
| rstress-seed-1 | 7.045e-06 / 4.774e-05 | 3.969e-03 | 0.0627 | 1.999 / 1.858 | 0.024 | 19.0 | 0 / 0 |

KL 全部 ≤ target_kl（kl_early_stop 未触发），clip fraction median 0，
post-clip norm 恒 0.5（纯 clip 区域），episode length 全程 8.0，无 Beta 边界
塌缩（min α/β ≥1.86）。c1 的 probe paired：RMS by seed 0.0619 / 0.0828 ≥0.01；
跨 seed cosine 0.9962、sign agreement 1.0。

### Held-out（matched，与 E-031 同协议，24 scenarios）

| seed | policy | mean speed (m/s) | energy intensity (mL/km) | route completion | arrive_dest | collision / OOR / stopped |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | initial | 9.998465 | 47.047213 | 0.9466 | 0.8125 | 0 / 0 / 0 |
| 0 | r0 final | 9.964411 | 46.983970 | 0.9451 | 0.8125 | 0 / 0 / 0 |
| 0 | rstress final | 10.070494 | 47.174267 | 0.9475 | 0.8125 | 0 / 0 / 0 |
| 1 | initial | 9.998465 | 47.047213 | 0.9466 | 0.8125 | 0 / 0 / 0 |
| 1 | r0 final | 9.935536 | 46.936153 | 0.9454 | 0.8125 | 0 / 0 / 0 |
| 1 | rstress final | 10.075349 | 47.185713 | 0.9463 | 0.8125 | 0 / 0 / 0 |

Paired delta（rstress−r0）：

| seed | Δ speed (rel) | Δ energy (rel) | Δ route (rel) |
| --- | ---: | ---: | ---: |
| 0 | +0.1061 m/s（+1.06%） | +0.1903 mL/km（+0.41%） | +0.00238（+0.25%） |
| 1 | +0.1398 m/s（+1.41%） | +0.2496 mL/km（+0.53%） | +0.00093（+0.10%） |

- **c1 PASSED**、**c5 PASSED**（arrive_dest 0.8125 全保持，无 collision/OOR/
  stopped，route completion 保持）。
- **c2/c3 PASSED**：mean_speed 与 energy intensity 双 seed 均超 0.1% 噪声界且同向
  （route_completion seed1 为 +0.10%，在噪声界内不计入分离）。
- **c4 FAILED**：条件要求 energy 或 speed delta 双 seed 均为负，实测**全部为正**
  （`failure_reasons=["c4_direction_matches_objective"]`）。相对共同 initial，
  R0 双 seed speed/energy 均下降（9.964/9.936 m/s、46.98/46.94 mL/km），Rstress
  双 seed 均上升（10.070/10.075 m/s、47.17/47.19 mL/km）——两 arm 从同一起点向
  相反方向移动。

## 首次运行失败与修复（provenance）

首次正式运行 4/4 runs 训练本体全部完成（npz 至 update-049）但在最终
`TrainingRunSummary` 构造时因 pydantic `reward_profile` Literal 只含
`plannerrft_energy_v1`/`plannerrft_no_energy_v1` 而全部写 `failure.json`
（`literal_error`）。诊断确认非 Beta 塌缩后修复三处实现 bug：

1. `rl/artifacts/summaries.py` 两处 `reward_profile` Literal 扩展为 4 名集合；
2. `rl/reward/objectives/plannerrft.py` `_finalize` 硬编码旧 profile 名 → 改用
   `config.name`（此前 `RewardResult.profile_name` 与 npz rollout 元数据对新
   profile 记录失真的通用名）；`RewardProfileName` Literal 与
   `rollout_audit_keys` 成员检查同步扩展；
3. `objective_positive_control/runner.py` 的 `failed_runs` 与 `completed`
   字典键映射错误（成功路径 KeyError）。

回归测试（red/green 验证：临时还原 Literal 后 2 项测试以与 failure.json 相同的
错误签名失败，恢复后通过）：`tests/training/test_reward.py`（profile 名端到端
持久化）、`tests/training/test_ppo.py`（4 profile 参数化的 update summary 持久化）、
`tests/simulation/test_training_tracking.py`（gpu+simulator e2e smoke，band
profile 全链路）。由于首次运行的 npz 元数据内嵌失真 profile 名，4 个 stale run
目录与顶层 summary 整体删除后 clean 重跑；本记录全部数值来自修复后的重跑，run
summary 的 `reward_profile` 字段均为正确新名。

## 结论边界

- **分离是真实且可复现的**（c2/c3）：E-038 的 objective 差异在该闭环 PPO 协议下
  确实传导为超出 matched 噪声界的行为差异。失败仅在 c4 的方向语义。
- **方向相反的机制未被本实验建立**。band 双侧饱和的取值区间为
  [46.371, 48.751] mL/km，而 initial held-out 聚合强度 47.047 已在带内；λ=64
  energy 项与校准 Progress/speed 分量在带内的交互是候选解释，但本实验没有检验，
  不应解读为"band 表示本身错误"或"λ=64 必然反向"。
- **R0 arm 是校准 R0**（`plannerrft_no_energy_calibrated_v1`），与 E-031/E-039
  训练用的未校准 `plannerrft_no_energy_v1` 不同。两 arm 共享同套校准分量使
  paired 差分隔离 energy 项，但单 arm 相对 initial 的绝对移动不能与 E-039 直接
  比较。
- 本实验**不支持任何节能结论**；Task H 的 deterministic + stochastic 行为比较与
  后续以 "positive control 通过" 为前提的 energy-minimization 结论不能以该
  artifact 为基础。
- Issue #94 预设的 fallback（"gradient gate 已通过但闭环不可分 → 返回
  Task D/F"）不适用于本结果：此处闭环**可分**但方向相反，是独立的结果形态，
  Task G 以 Gate G FAILED（c4）结案。

## 验证与产物

- `tests/training/test_objective_positive_control.py` 17 项、新增回归 3 项
  （快速训练测试 49 项通过）、gpu+simulator smoke 2 项（54 s）、`just test`
  9 passed；`ruff check`/`ruff format --check`、`just typecheck`（0 errors）
  通过。
- 正式产物（git ignored）：
  `outputs/studies/scalar-reward/e-040-issue94-task-g-objective-positive-control/`
  - `study_manifest.yaml`：resolved 研究清单（arms + seeds + Gate 阈值）。
  - 4 个 run 目录（`r0-seed-{0,1}`、`rstress-seed-{0,1}`）：
    `summary.json`（逐 update 训练诊断）、`policy-initial/final.pt`、
    `policy-update-000..049.pt`、`training-state.ckpt`、`updates/update-NNN/*.npz`
    （完整 rollout audit）、`resolved_config.yaml`、`runtime_metadata.json`。
  - `heldout/seed-{0,1}/{initial,r0,rstress}/`：matched held-out 评测产物。
  - `summary.json`：per-run metrics、paired probe/held-out 分析、Gate G 判定
    （`gate_g_passed=false`、`failure_reasons=["c4_direction_matches_objective"]`）。

Task G 已完成（Gate G FAILED on c4，真实负结果）；本次改动未提交、未推送，
GitHub Issue #94 的评论报告另行执行。
