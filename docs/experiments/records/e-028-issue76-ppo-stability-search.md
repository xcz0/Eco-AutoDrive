# E-028 Issue #76：reset 修复后的 PPO 稳定训练超参数分层搜索

[返回实验索引](../README.md)

**日期 / 类型 / 目的**：2026-08-31 至 2026-09-01 / 远程训练机（rtx_a4000）正式搜索 / Issue #76：在固定 reward（`metadrive_builtin_v1`）下，通过 P0/P1 基线复验 + Optuna Stage A/B/C 分层搜索确认 PPO 训练稳定超参数区域，识别主要失败模式，为后续 reward 实验建立可信基线。不评估能耗 reward、不比较 reward profile、不证明 reward improvement。

**状态**：完成。P0/P1 在 reset 修复后均稳定（E-026 式退化不复现）；Stage A 24 trials 得到稳定区域；Stage B top 5 × 3 seeds × 50 updates 全部通过；Stage C top 2 × 3 seeds × 100 updates 中 **config-0001（epochs=1, lr=1.63e-5）为唯一跨 seed 长程稳定候选**，config-0003（epochs=3）在 seed 0 的 100 updates 后出现 Beta 边界塌缩。主要失败模式为 epochs=3 与 lr 联合把 guidance Beta 分布推向 ±1 边界。

## 前置：reset bug 修复与 P0/P1 复验

Stage A 首次运行被 MetaDrive reset 语义 bug 阻断：step 过一次后对同一 map/seed reset 时，未清除的 `engine.external_actions` 在 reset 内部 `taskMgr.step()` 中重放旧世界轨迹，车辆从旧轨迹位置而非出生点开始（诊断过程见 Issue #76 第 3 条评论）。修复为 `TrajectoryMetaDriveEnv.reset` 前清除 `engine.external_actions`（commit `bbf1365`，回归测试 `tests/simulation/test_closed_loop.py::test_same_scenario_reset_restores_spawn_after_trajectory_step`）。

修复后 P0/P1 复验（2026-08-31，20 updates × training seed 0，`metadrive_builtin_v1`）：

| 指标 | P0 (smoke) | P1 (conservative) |
| --- | ---: | ---: |
| out_of_road_count（20 updates 累计） | 0 | 0 |
| collision_count | 0 | 0 |
| episode_count / update | 2（全程不变） | 16（全程不变） |
| mean_episode_length | 16.0（全程不变） | 16.0（全程不变） |
| entropy（首→末） | 1.1375 → 1.3418（未塌缩） | 1.1361 → 1.1369（几乎不变） |

P0 不再复现 E-026 式退化。E-026 的退化与 E-027 的"update 强度过大"归因均被推翻，结论边界修正已写入对应记录。产物：

- P0: `outputs/training/ppo/2026-08-31/21-56-45-seed-0-replay-0/`
- P1: `outputs/training/ppo/2026-08-31/21-59-48-seed-0-replay-0/`

## 代码、环境与配置

- 基础 commit：`bbf1365c21ad060cbe87be0a3691544b3644a2c4`；运行包含本会话未提交的工作流修复（见下文"工作流修复"），涉及 `scripts/studies/ppo_stability.py`、`configs/studies/ppo/stability.yaml`、`tests/configuration/test_studies.py`。
- Windows-10 build 26200；Python 3.10.20；PyTorch 2.12.1+cu126；Lightning 2.6.5；MetaDrive 0.4.3；Optuna（TPE + MedianPruner + SQLite）；CUDA `cuda:0`（RTX A4000）；BF16 mixed precision；资源 profile `rtx_a4000`（`rollout_worker_count=4`）。
- study 配置：`configs/studies/ppo/stability.yaml`，base `jobs/training/ppo/conservative`；reward 固定 `metadrive_builtin_v1`；frozen planner hash 前后一致（`146f582d...22ced6`），仅 Exploration Policy 更新。
- 搜索空间：lr 1e-5~2e-4（log）、epochs {1,2,3}、batch {128,256,512}、minibatch {64,128,256}、target_kl 0.005~0.03（log）；sampler seed 760024；固定 gamma=0.99、gae_lambda=0.95、clip_epsilon=0.2、entropy_coefficient=0.01、value_coefficient=0.5、guidance range 默认值。
- batch 通过独立 scenario 数扩展（batch=128→S/SC seeds 0–3，256→seeds 0–7，512→seeds 0–15），`transitions_per_scenario=16`，不延长单条 trajectory。
- 闭环评测（Stage B/C 每 run 的 initial/final checkpoint）：S/SC held-out map seeds 16–23、100 transitions/scenario、evaluation seed 760025、reward-independent 门槛（collision 不增加、episode/route/distance/speed retention ≥ 0.90）。

命令：

```powershell
just ppo-stability-stage-a outputs/studies/ppo-stability/issue76
just ppo-stability-stage-b outputs/studies/ppo-stability/issue76
just ppo-stability-stage-c outputs/studies/ppo-stability/issue76
just summarize-ppo-stability outputs/studies/ppo-stability/issue76
```

## Stage A：Optuna 稳定区域搜索（24 trials × 30 updates，training seed 0）

trial 状态：17 complete（stable）/ 5 pruned / 2 fail。

- 5 个 pruned 全部是 `invalid_batch_minibatch_combination`（batch=128 + minibatch=256 不可整除组合），均为采样阶段剪枝，不消耗训练预算。
- 2 个 fail（trial 22、23）：lr 1.41e-4 / 1.69e-4 × epochs=3，训练结束后 policy probe 采样触发 `ValueError: guidance action must be strictly inside (-1, 1)`——Beta guidance 分布被推向 ±1 边界（Issue 阶段 0 定义的失稳模式）。
- 17 个 stable trial 覆盖：lr 1.03e-5 ~ 1.06e-4、batch 128/256/512、epochs 1/2/3 全部有稳定代表；稳定 trial 的 out_of_road_fraction=0、episode length retention ≥ 0.94、approx_kl 全程 ~1e-5 量级、无 KL early-stop 失控。
- 唯一失败模式集中在 epochs=3 × lr ≥ 1.4e-4；稳定上界附近（trial 8: lr=1.06e-4 × epochs=2）仍稳定。

## Stage B：top 5 多 seed 复验（5 configs × seeds [0,1,2] × 50 updates + 闭环评测）

top 5 configs（Stage A trials 0–4）全部 15 runs complete；initial vs final 闭环评测全部 passed（collision 未增加、episode_length_retention=1.0、route_progress_retention ≈ 1.0）。排名 `ranked_config_ids=[3,1,0,4,2]`，晋级 Stage C：config 3 与 config 1。

| config | batch | minibatch | epochs | lr | target_kl |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 256 | 128 | 3 | 3.2023e-5 | 0.00517 |
| 1 | 128 | 128 | 1 | 1.6301e-5 | 0.00600 |

## Stage C：top 2 长程确认（2 configs × seeds [0,1,2] × 100 updates + 闭环评测）

| config | seed | 状态 | 说明 |
| ---: | ---: | --- | --- |
| 3 | 0 | **failed** | 100 updates 训练完成后最终 policy probe 触发 `ValueError: guidance action must be strictly inside (-1, 1)`（Beta 边界塌缩，与首次尝试确定性复现） |
| 3 | 1 | complete | 评测 passed |
| 3 | 2 | complete | 评测 passed |
| 1 | 0 | complete | 评测 passed |
| 1 | 1 | complete | 评测 passed |
| 1 | 2 | complete | 评测 passed |

`ranked_config_ids=[1]`：config-0001 是唯一完成 3 seeds × 100 updates 且全部评测通过的候选（worst episode length retention = 1.0）。config-0003 在 50 updates（Stage B）时 3 个 seed 全稳定，但在 100 updates 时 seed 0 失稳，不具备长程稳健性。

**最终稳定候选（config-0001，Stage A trial 1）**：`batch=128、minibatch=128、epochs=1、lr=1.6301e-5、target_kl=0.00600`（base conservative，其余参数同 P1）。

## 失败模式与稳定区域

- 主要失败模式：**epochs=3 与 lr 联合将 guidance Beta 分布推向 ±1 边界**，在训练结束的 policy probe 采样中越界报错。30 updates 内仅在 lr ≥ 1.4e-4 × epochs=3 出现；100 updates 长程下即使 lr=3.2e-5（config-0003 seed 0）也可出现。epochs=1 候选未出现该模式。
- 稳定区域（30 updates × seed 0 证据 + 50/100 updates 多 seed 复验）：**lr 1e-5 ~ 1e-4，epochs 1–2（或 epochs=3 仅在低 lr 短程内）**；batch 128/256/512 与 minibatch 组合在该区域内未见稳定性差异。
- E-026 式 out-of-road 阶跃退化在 reset 修复后全程未复现（Stage A/B/C 所有 run 的 out_of_road_fraction=0）。

## 工作流修复（本会话，均已通过测试与 lint/typecheck）

1. **clip_fraction hard-prune 禁用**（配置缺陷）：稳定基线 P1 的 clip_fraction 全程 0.67–0.74，原 0.50 阈值会把所有稳定配置剪枝（首次 Stage A 24/24 全剪枝）。`PruningConfig.clip_fraction` 改为 `StrictFloat | None`，`stability.yaml` 设 `null` 并注释原因；clip_fraction 保持监控上报。测试：`test_stability_monitor_reports_registered_domain_prune_reasons`。
2. **Stage A 单 trial 失败不终止 study**：`study.optimize(..., catch=(Exception,))`，失败记录为 fail trial + failure 原因（背景：Beta 边界 ValueError 曾崩掉整个 study）。
3. **参数重要性 ImportError 捕获**：optuna fANOVA 需要 sklearn（未安装），汇总记录 `parameter_importance_error` 而非崩溃。参数重要性在 Issue 中仅为辅助项，本记录不提供。
4. **Stage B/C 单 run 失败不终止 stage**（本次 Stage C 首跑实证后修复）：`_run_validation` 增加 `except Exception` 分支，失败 run 记录为 `state=failed` + 原因，其余 runs 继续。首次 Stage C 运行中 config-0003/seed-0 的 Beta 边界 ValueError 曾终止整个 stage（且 `stage_root.mkdir(exist_ok=False)` 使其不可续跑）；修复后同配置同 seed 确定性复现失败并被如实记录。测试：`test_stability_study_sqlite_continuation_and_validation_ranking` 扩展 failed record 排除断言。

无效/中断产物归档（本地，gitignored）：

- `outputs/studies/ppo-stability/issue76-invalid-pre-reset-fix/`：reset bug 修复前的 Stage A（6 trials 全部无效）。
- `outputs/studies/ppo-stability/issue76-invalid-clip-prune/`：clip 误剪枝的 Stage A 首跑（24/24 pruned）。
- `outputs/studies/ppo-stability/issue76-crashed-stage-c-attempt/`：Stage C 首跑（config-0003/seed-0 Beta 边界崩溃，保留其 100 updates 训练产物作为失败证据）。

## 结论边界

本记录支持：

- reset 修复（`bbf1365`）后，P0 smoke 与 P1 conservative 在 20 updates × seed 0 下均稳定；E-026 式退化归因于 reset bug 而非 PPO update 强度（E-026/E-027 的归因修正见各自记录）。
- 在 `metadrive_builtin_v1`、no-traffic S/SC 场景、`transitions_per_environment=16`、本机环境下：lr 1e-5~1e-4 为 30 updates 内的稳定区域；epochs=3 与较高 lr 联合是主要失败模式（Beta guidance 分布边界塌缩）。
- config-0001（batch=128、minibatch=128、epochs=1、lr=1.6301e-5、target_kl=0.00600）是唯一通过 3 training seeds × 100 updates 训练 + reward-independent 闭环评测（held-out map seeds 16–23）的候选，可作为后续 reward 实验的 PPO 稳定基线起点。
- config-0003（epochs=3）在 50 updates × 3 seeds 稳定但 100 updates × seed 0 失稳，不具长程稳健性——Stage B 的 50 updates 复验不足以确认 epochs=3 配置的长程稳定性。

本记录不支持：

- reward 优劣、节能改善或 builtin vs energy 比较（固定单一 reward profile）。
- 稳定区域外推到有 traffic 场景、更长训练（>100 updates）、其他 reward profile 或其他机器/精度。
- 超参数因果解释——搜索为 TPE 观测性筛选，且参数重要性因 sklearn 缺失不可用（`parameter_importance_error` 已记录）。
- Beta 边界塌缩的深层机制（gradient 竞争或 guidance sensitivity）——Issue 的条件机制诊断针对"E-026 式退化仍复现"的情形，该情形已被 reset 修复消除；epochs=3 失败模式通过参数关联与 config-0001 的选择规避，未做机制实验。

## 产物

- Optuna study：`outputs/studies/ppo-stability/issue76/study.db`（24 trials）、`study_manifest.yaml`、`stage-a-summary.json`（汇总刷新）
- Stage A：`outputs/studies/ppo-stability/issue76/stage-a/trial-0000..0023/`（每 trial 含 resolved_config、summary、checkpoints、trial.json）
- Stage B：`outputs/studies/ppo-stability/issue76/stage-b/config-000{0..4}/seed-{0,1,2}/`（含 evaluation/initial、evaluation/final、comparison.json）+ `summary.json`
- Stage C：`outputs/studies/ppo-stability/issue76/stage-c/config-000{1,3}/seed-{0,1,2}/` + `summary.json`
- P0/P1 复验：`outputs/training/ppo/2026-08-31/21-56-45-seed-0-replay-0/`、`outputs/training/ppo/2026-08-31/21-59-48-seed-0-replay-0/`

Issue #76 验收完成（机制诊断项因退化不复现而不适用；参数重要性因 sklearn 缺失缺失，见结论边界）。
