# E-039 Issue #94 Task F：标准 PPO effective-update region 搜索

[返回实验索引](../README.md) · [前置 E-038 Task E](e-038-issue94-task-e-energy-representation.md)

**日期 / 类型 / 目的**：2026-09-11 / 正式训练网格搜索 / 在 R0 reward objective 与 matched
fixed-batch 协议下，扫描 PPO 优化器区域（learning rate × epochs × max gradient norm），
判定哪个 config 使 policy 产生**有效且稳定**的 per-update 变化，完成 Issue #94 Task F
（Gate F）。不修改 reward 组合、guidance 或执行方式；不训练 band arm。

**裁定：Gate F PASSED。** 选定 config `lr=1.5e-04, epochs=1, max_gradient_norm=0.5`
（arm `lr1.5000e-04-epochs1-mgn0.5`）。有效区域边界为 **epochs=1 且
lr ∈ (5e-05, 1.5e-04]**：lr=5e-05 仍低于有效 update 阈值，lr=1.5e-04 在 epochs=2
（以及 lr=4.5e-04 全部 epochs）训练内崩溃。三个 lr=1.5e-04/epochs=1 arms 通过全部
7 条 Gate F 条件，选择规则（最小 lr → epochs → mgn）取 mgn=0.5。

## 运行来源与固定协议

- 代码基线为 `scalar-energy-reward` 分支未提交改动（Task F runner、诊断层与
  `evaluation/artifacts/models.py` 的 `RuntimeMetadata.git_branch` 字段修复，见下文）；
  每个 arm 目录的 `runtime_metadata.json`、`tracked_diff.patch` 与 `source/` 保存该 arm
  正式训练时的来源。
- Windows、Python 3.10、PyTorch 2.12.1+cu126、MetaDrive 0.4.3，GPU NVIDIA RTX A4000
  （`configs` 资源名与实际设备一致）。
- 训练协议 matched 于 E-031/E-032 的 `fixed-batch.yaml` control override：R0 reward
  `plannerrft_no_energy_v1`（a1 arm；Task E band 表示只改 energy 分量，E-038 已验证 R0
  arm 逐位一致，故 R0 objective 不受 Task E 影响）、training seed=0、batch=128
  （16 episodes × 8 slots）、minibatch=128（每 epoch 恰一个 minibatch）、
  `target_kl=0.006`、`update_count=50`。
- 网格：lr ∈ {1.6301e-05（E-031 control）, 5e-05, 1.5e-04, 4.5e-04} × epochs ∈ {1, 2}
  × max_gradient_norm ∈ {0.5, 1.0, 2.0}，共 24 arms；全部 arms 共享同一
  initial policy hash（`049697739a…`，由 runner 校验）。scheduler 总步数随 epochs 同步
  覆盖（`scheduler_total_optimizer_steps = update_count × epochs`）。
- Arm 训练失败（非有限 rollout 量）显式记录 `failure.json` 并继续网格；这本身是
  Gate F 条件 5（无 Beta boundary collapse）的直接证据，不是被掩盖的缺项。

```powershell
just experiment training effective-update run --output-dir outputs/studies/scalar-reward/e-039-issue94-task-f-effective-update-region
```

配置清单：`configs/experiments/training/effective-update.yaml`（网格、Gate 阈值与选择
规则）；实现位于 `src/eco_planner/experiments/training/effective_update/`。runner 幂等：
已存在 `summary.json`/`failure.json` 的 arm 跳过训练，分析阶段可在不重训的情况下重跑。

## KL 测量语义（本实验的方法学发现）

Gate F 条件 1/2 以 "median approximate KL" 判定，但两个现成测量在该协议下都不可直接
使用，本实验因此改用离线重算：

1. **训练内 torchrl `kl_approx` 在 optimizer step 之前测量**（loss forward 时）。本网格
   batch=minibatch（每 epoch 一个 minibatch），epochs=1 时它测量的是
   `KL(old || policy-at-update-start)`，即结构性地 ≈ 0（±1e-8），与 update 大小无关。
   E-031/E-032 报告的 KL≈1e-8 主要是该测量伪影；其 under-update 的真实证据来自
   ratio≈1 与 parameter delta≈噪声。本实验保留该量并改名为 `pre_update_kl`。
2. **post-update 单抽样 k1**（对持久化的 128 个 on-policy 动作用
   `(old_log_prob − new_log_prob).mean()`，与 torchrl 同公式）的 batch 噪声约
   1e-4，远高于 1e-6 的 gate 下限，无法分辨门限（本实验实测其 median 为
   -3.2e-6…+2.8e-8，被噪声主导）。

**主估计量**：rollout NPZ 持久化了完整 policy context 与 old Beta 参数
（`beta_alpha`/`beta_beta`），因此对同一 estimand `E_old[old_log_prob − new_log_prob]`
用显式 seed 的 Monte-Carlo 在 old 分布上积分（每个 context 4096 抽样，seed
`1_000_003 + update_index`），加载 `policy-update-NNN.pt` 前向得到 new 参数。该估计
将 median 的分辨率降到 ~3e-7，可与 1e-6 门限比较；二次型 k3
（`½E[log_ratio²]`）作为独立交叉验证（小 update Fisher 区域内 k3≈KL，两者一致：
lr=1.5e-04 arms k3 median ≈ 1.0–1.2e-6 vs MC KL median 1.16–1.53e-6）。

## Gate F 判定与结果

Gate 条件（阈值见 yaml）：c1 median post-update KL ≥ 1e-6；c2 ≥90% updates KL ≤
target_kl 且无 runaway（tail 10 updates median > 10× 总 median）；c3 policy ratio 变化
（median of max(|mean−1|, std)）≥ 1e-4；c4 deterministic Beta-mean probe RMS shift ≥
0.01；c5 Beta 参数 ≥0.1 且 boundary mass ≤0.2；c6 collision+OOR ≤2 且 episode length
保持 ≥50%；c7 held-out 非安全指标相对变化 ≥0.1%（E-031 matched evaluation noise）。
under-update 分类 = KL、ratio、RMS 三者均低于各自下限。

24 arms 结果（每 arm 50 updates；完整逐 arm 数据见 `summary.json`）：

| 区域 | arms | 结果 |
| --- | ---: | --- |
| lr=1.6301e-05（control，全部 epochs/mgn） | 6 | **under_update=True**：MC KL median ~±1e-9，ratio 2.8–7.5e-5，RMS 4–9e-4；c1/c3/c4 全失败。E-031 under-update 结论在修正 KL 测量后成立 |
| lr=5e-05（全部 epochs/mgn） | 6 | KL median 2e-9–8e-8 < 1e-6，c1 失败（epochs=1 另有 c4 失败）；ratio 1.4–5.0e-4 已过 c3，但 update 强度仍不足 |
| lr=1.5e-04, epochs=1（mgn 0.5/1/2） | 3 | **全部 7 条通过**（见下表） |
| lr=1.5e-04, epochs=2；lr=4.5e-04（全部） | 9 | 训练失败：`RuntimeError: rollout host tensor 'old_joint_guidance_log_prob' contains non-finite values`（Beta 边界塌缩导致 log-prob 非有限）——条件 5 的直接失败证据 |

三个通过 arms 的关键数值（c1–c6 来自训练诊断，c7 来自 matched held-out 评测）：

| arm | MC KL median / max | ratio 变化 | probe RMS | min α / β | boundary mass | collision/OOR | held-out 超噪声指标 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| lr1.5e-04-epochs1-mgn0.5 | 1.160e-06 / 2.67e-05 | 1.409e-03 | 0.0289 | 1.912 / 1.990 | 0.022 | 0 / 0 | mean_speed_mps（+0.14%） |
| lr1.5e-04-epochs1-mgn1 | 1.529e-06 / 2.90e-05 | 1.495e-03 | 0.0310 | 同量级 | 0.022 | 0 / 0 | mean_speed_mps（+0.15%） |
| lr1.5e-04-epochs1-mgn2 | 1.406e-06 / 2.98e-05 | 1.548e-03 | 0.0322 | 同量级 | 0.022 | 0 / 0 | mean_speed_mps（+0.15%） |

三个 arms 的 KL 全部 ≤ target_kl（within fraction = 1.00），tail median ≈ 0（无
runaway），kl_early_stop 均未触发。选定 arm 的其他指标：pre-clip grad norm median
28.63（post-clip 恒为 mgn=0.5，即该 arm 实际处于纯 clip 区域），clip fraction median
0（ratio 远未触及 PPO clip），value loss 7.13 → 6.46，explained variance ~0
（50 updates 内 critic 尚未拟合），episode length 全程 8.0 无坍塌。

选定 arm 的 held-out（matched，与 E-031 同协议，24 scenarios）：

| 指标 | initial policy | final policy | 相对变化 |
| --- | ---: | ---: | ---: |
| mean speed (m/s) | 9.998465 | 9.984466 | 0.14% |
| energy intensity (mL/km) | 47.047213 | 47.021920 | 0.05% |
| route completion | — | — | 0.011% |
| arrive_dest fraction | 0.8125 | 0.8125 | — |
| collision / OOR | 0 / 0 | 0 / 0 | — |

条件 7 仅由 mean_speed_mps 超出 0.1% 噪声界支撑（energy 类指标变化 0.05% 在噪声界
内）；方向为速度小幅下降（R0 无 energy 项，不解释为节能效应）。

## 结论边界

- "Effective update" 的判定是**机械性**的：per-update KL 刚过 1e-6 下限、ratio/RMS/probe
  可测移动、held-out 速度变化刚过噪声界。本实验不声称学习到了有意义的行为，也不支持
  任何节能结论；R0 objective 下 held-out 变化本身就非目标。
- 通过 arms 的 per-update KL（1.16e-6）只是边际过线；MC estimator 的 median 分辨率
  ~3e-7 足以支撑该判定，但不应把 1.16e-6 解读为精确点值。
- 有效区域结论限于本协议（batch=128 单 minibatch、50 updates、R0、seed 0、BF16）。
  epochs=2 在 lr=1.5e-04 下的失败说明该边界不是平滑的 "更大 lr 皆可" 区域；
  lr∈(5e-05, 1.5e-04] 的内部未加密采样，不排除更低 lr 在更长训练下也有效。
- 失败 arms 的崩溃方式与 E-028/E-030 观察到的 Beta 边界塌缩一致：高有效 step 下
  concentration 快速退化使 log-prob 非有限。条件 5 通过的 arms 的 min α/β 仍 ≥1.9，
  与初始 2.0 相比变化很小。

## 验证与产物

- `tests/training/test_effective_update.py` 18 项通过（含 post-update KL 重算的
  真策略集成测试：identical checkpoint → KL=0，扰动 checkpoint → KL>1e-3）；
  `just lint`、`just typecheck` 通过。
- 运行中发现并修复仓库现有 bug：`evaluation/artifacts/models.py` 的 `RuntimeMetadata`
  缺 `git_branch` 字段（`collect_repository_metadata` 新增字段未同步），任何
  evaluation run 在 HEAD 会崩溃；修复后 `tests/evaluation/test_engine.py` 3 项通过。
  首次正式运行中 initial held-out 评测曾因此中断，修复后 runner 幂等重入完成。
- 正式产物（git ignored）：
  `outputs/studies/scalar-reward/e-039-issue94-task-f-effective-update-region/`
  - `study_manifest.yaml`：resolved 研究清单（网格 + Gate 阈值）。
  - 15 个完成 arm 目录：`summary.json`（逐 update 训练诊断）、
    `policy-initial/final/update-NNN.pt`、`updates/update-NNN/*.npz`（完整 rollout
    audit）、`resolved_config.yaml`、`runtime_metadata.json`。
  - 9 个失败 arm 目录：`failure.json`（崩溃原因）。
  - `heldout/initial/` 与 3 个候选 `heldout/<label>/`：matched held-out 评测产物。
  - `summary.json`：逐 arm metrics（pre/post-update KL、ratio、RMS、Beta、行为、
    held-out 相对变化）、Gate F 判定与选择结果。

Task F 已完成（Gate F PASSED）；本次改动未提交、未推送，GitHub Issue #94 的评论
报告另行执行。
