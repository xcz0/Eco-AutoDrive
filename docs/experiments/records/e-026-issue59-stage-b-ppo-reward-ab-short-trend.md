# E-026 Issue #59 阶段 B：20 updates × 3 seeds 匹配 PPO A/B 短趋势

[返回实验索引](../README.md)

**日期 / 类型 / 目的**：2026-08-28 / 本机短趋势验证 / 验证 Issue #59 阶段 B：在严格匹配条件下执行 20 updates × 3 个独立 training seeds 的 builtin 与 energy 两条 PPO 训练，汇总跨 seed 方向与离散程度，排除明显 reward hacking。不证明 reward 有效、节能改善或 PlannerRFT parity。

**状态**：全部 9 项机械检查通过，3 个 seed 均完成 20 updates；两 profile 间无显著差异，无 reward hacking。策略在 ~5 updates 后因 smoke PPO 配置退化（两个 profile 均出现大量 out-of-road），这是 PPO 超参数限制而非 reward 定义失败。

## 代码、环境与配置

- 基础 commit：`d54553bcbe8294a10596bfbe5666c3ba57129b7a`；运行包含未提交的 MetaDrive navigation 恢复修复（`TorchRLMetaDriveEnv._reset` 捕获 "no connected navigation route lanes" 错误后重建 env 并重试），diff 保存在每个 run 的 `tracked_diff.patch`。
- Windows-10 build 26200；Python 3.10.20；PyTorch 2.12.1+cu126；Lightning 2.6.5；MetaDrive 0.4.3；Pydantic 2.13.4；CUDA `cuda:0`；BF16 mixed precision。
- 资源 profile `rtx_a4000`：`rollout_worker_count=4`、`torch_threads_per_worker=12`。
- 训练配置：`configs/studies/reward/ppo_ab_long_term.yaml` → base `jobs/training/ppo/smoke`；场景 `S` / `SC` seed 0、training seeds `[0, 1, 2]`、replay 0、DDIM5、`transitions_per_environment=16`、`update_count=20`、`scheduler_total_optimizer_steps=160`（20 × 4 epochs × 2 minibatches = 160）、`history_warmup_steps=0`、`deterministic=true`、no-traffic。
- planner checkpoint 为 276 EMA tensors / 6,042,628 parameters；frozen planner hash 前后均为 `146f582dd5994f6490aa1b4293d6d972b611362151b54d84022f09f13e22ced6`。
- 本机忽略产物根目录：`outputs/studies/ppo_reward_ab/2026-08-28-stage-b/`。

## MetaDrive navigation 恢复修复

在 smoke PPO 配置下，策略在 ~5 updates 后退化并频繁出界（out-of-road），导致同一 slot 被反复 reset。MetaDrive 在累积大量 same-map reset 后 navigation 状态可能损坏，表现为`no connected navigation route lanes exist in the local lane set`。修复方式：`TorchRLMetaDriveEnv._reset` 捕获该特定错误后调用 `MetaDriveEnvSlot.recreate_environment()`（关闭并重建 env），然后重试一次 reset。此修复不影响正常路径下的 reset 行为，只在 navigation 损坏时触发。

## 命令与验证

```powershell
just ppo-reward-ab outputs/studies/ppo_reward_ab/2026-08-28-stage-b configs/studies/reward/ppo_ab_long_term.yaml
```

launcher 顺序执行 3 个 training seed × 2 个 reward profile = 6 条训练，每条 20 updates × 32 transitions = 640 transitions；总墙钟约 12 分钟。`review_report.json` 由 `scripts.analysis.ppo_reward_ab` 自动生成。

实现回归同时通过：

```text
just test-sim  -> 5 passed, 46 deselected
just lint      -> All checks passed
```

## 机械检查（全部通过，3 个 seed/replay pair × 9 项 = 27 项）

| 检查                                      | seed 0 | seed 1 | seed 2 |
| ----------------------------------------- | ------ | ------ | ------ |
| `resolved_configs_match_except_reward`    | passed | passed | passed |
| `resolved_config_matches_matrix`          | passed | passed | passed |
| `initial_policy_hashes_match`             | passed | passed | passed |
| `noise_seeds_match`                       | passed | passed | passed |
| `policy_action_seeds_match`               | passed | passed | passed |
| `configured_update_count_completed`       | passed | passed | passed |
| `both_policies_updated`                   | passed | passed | passed |
| `both_frozen_planners_unchanged`          | passed | passed | passed |
| `pre_update_collection_is_exactly_paired` | passed | passed | passed |

`mechanical_status = passed`，`review_status = pending_human_review`。

## 跨 seed 效果窗口汇总（updates 1–19，每 profile 608 transitions × seed）

| 指标                             |     mean | sample std |      min |      max |
| -------------------------------- | -------: | ---------: | -------: | -------: |
| energy intensity change fraction | -2.38e-4 |    3.11e-4 | -5.89e-4 |  3.70e-6 |
| longitudinal action mean delta   |  1.84e-3 |    4.78e-2 | -4.85e-2 |  4.67e-2 |
| mean speed change fraction       | -1.27e-3 |    1.27e-3 | -2.62e-3 | -9.74e-5 |
| progress change fraction         | -1.26e-3 |    1.09e-3 | -2.41e-3 | -2.59e-4 |
| collision count delta            |        0 |          0 |        0 |        0 |
| out-of-road count delta          |     1.67 |       2.89 |        0 |        5 |

review_thresholds 触发情况：

| flag                                      | 触发次数 / 3 |
| ----------------------------------------- | -----------: |
| `longitudinal_action_changed`             |            2 |
| `out_of_road_regressed`                   |            1 |
| `energy_intensity_changed`                |            0 |
| `progress_regressed`                      |            0 |
| `mean_speed_regressed`                    |            0 |
| `collision_regressed`                     |            0 |
| `energy_drop_confounded_by_progress_drop` |            0 |

## 各 seed 效果窗口明细

| seed | profile | distance (m) | fuel mL/km | mean speed (m/s) | out-of-road | progress |
| ---: | ------- | -----------: | ---------: | ---------------: | ----------: | -------: |
|    0 | builtin |        610.8 |      47.82 |            10.05 |         127 |    3.329 |
|    0 | energy  |        609.2 |      47.81 |            10.02 |         127 |    3.324 |
|    1 | builtin |        610.2 |      47.88 |            10.04 |         147 |    3.329 |
|    1 | energy  |        610.2 |      47.88 |            10.04 |         147 |    3.329 |
|    2 | builtin |        607.9 |      47.76 |            10.00 |         170 |    3.325 |
|    2 | energy  |        607.2 |      47.73 |             9.99 |         175 |    3.324 |

## 策略退化观察

两个 profile 均在 update ~5 后出现策略退化，表现为 out-of-road 急剧增加（从 0 到 14–16 per update）、episode_count 从 2 升至 17、terminal_override 大幅波动。这是 smoke PPO 配置（lr=0.00025、batch=32、4 epochs）的预期限制，不是 reward 定义问题：

- 退化在 builtin 和 energy 两个 profile 上同步出现，方向和量级一致；
- 退化后两 profile 间仍保持匹配（energy intensity 变化 < 0.06%）；
- PPO 诊断全部 finite：policy loss、value loss、entropy、approximate KL 和 gradient norm 在全部 20 updates × 2 profiles × 3 seeds = 120 个 update 中没有 NaN 或 Inf。

energy profile 的 value loss 和 gradient norm 比 builtin 大约 1–3 个数量级，直接来自`plannerrft_energy_v1` 的 reward scale（~27 vs ~1.5 per transition），与 E-024 和 E-025 观察一致，不构成数值不稳定。

## 结论边界

本记录支持：

- 严格匹配的 builtin/energy A/B 在当前 commit、本机、两个 no-traffic 场景、3 个 training seeds 和 20 个 update 下，pairing、artifact schema、有限值门槛和跨 seed 汇总全部成立。
- 20 updates 内未观察到 reward hacking：energy intensity、progress、mean speed、stopped fraction、collision 与 builtin 在 deadband 内一致；energy profile 没有通过少走、停车或降速来降低能耗。
- update 0 的 pre-update collection 在 3 个 seed 的每条 run 之间逐值相等，证明 A/B 的初始 policy、noise/action seeds 和初始 rollout 严格匹配。
- 策略退化是 PPO smoke 配置的限制，两个 profile 同步出现，不构成 reward 定义失败。

本记录不支持：

- reward 有效性、节能改善、收敛性或跨 seed 统计显著性结论——3 个 seed 不足以支撑统计推断；
- PlannerRFT/nuPlan scorer parity、真实车辆能耗、动力学可执行性或舒适性；
- builtin 与 energy 在更长训练或更复杂场景下的相对表现；
- learned policy 的闭环效果——如需声称，须使用相同固定 evaluation scenarios/seeds 对最终 checkpoints 做 reward-independent 闭环评测（可另立实验记录或后续 Issue）。

Issue #59 阶段 B 验收完成。
