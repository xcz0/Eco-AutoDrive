# E-025 Issue #59 阶段 A：`metadrive_builtin_v1` / `plannerrft_energy_v1` 匹配 PPO A/B 机械门控

[返回实验索引](../README.md)

**日期 / 类型 / 目的**：2026-08-27 / 本机机械门控 / 验证 Issue #59 阶段 A：
使用现有 `configs/studies/reward/ppo_ab.yaml` 在严格匹配条件下执行 builtin 与 energy 两条
PPO 训练，确认 pairing、finite、artifact schema 和短期方向，并排除明显退化或 reward
hacking。不证明 reward 有效、节能改善或 PlannerRFT parity。

**状态**：阶段 A 全部机械检查通过，无退化标志；Issue #59 阶段 B 仍未完成。

## 代码、环境与配置

- 基础 commit：`a9ee01df8377a65b6a568a0f8b32e01959fe9b9f`；工作区干净，
  `git_status_short=[]`，`tracked_diff.patch` 为空。本地未提交改动只有 gitignored 的
  `.env`（`MACHINE_NAME=rtx_a4000`），不进入运行时 resolved config。
- Windows-10 build 26200；Python 3.10.20；PyTorch 2.12.1+cu126；Lightning 2.6.5；
  MetaDrive 0.4.3；Pydantic 2.13.4；CUDA `cuda:0`；BF16 mixed precision。
- 资源 profile `rtx_a4000`：`rollout_worker_count=4`、`torch_threads_per_worker=12`。
- 训练配置：`configs/studies/reward/ppo_ab.yaml` → base `jobs/training/ppo/smoke`；
  场景 `S` / `SC` seed 0、training seed 0、replay 0、DDIM5、`transitions_per_environment=16`、
  `update_count=4`、`history_warmup_steps=0`、`deterministic=true`、no-traffic。
- planner checkpoint 为 276 EMA tensors / 6,042,628 parameters；frozen planner hash 前后均为
  `146f582dd5994f6490aa1b4293d6d972b611362151b54d84022f09f13e22ced6`。
- 初始 policy hash（两条 run 共享）：`049697739ab5d6e7bd212938bbac82c35215eb9ed14c02557952098a9718d05d`。
- noise seeds `[2140815137, 4177475342]`、policy action seeds `[2920704928, 3069229606]`。
- 本机忽略产物根目录：`outputs/studies/ppo_reward_ab/2026-08-27-stage-a/`。

## 命令与验证

```powershell
just ppo-reward-ab outputs/studies/ppo_reward_ab/2026-08-27-stage-a
```

launcher 顺序执行 `metadrive_builtin_v1`（A）与 `plannerrft_energy_v1`（B）各一次训练，
每条训练 4 个 update × 32 transitions = 128 transitions；总墙钟约 1 分钟。`review_report.json`
由 `scripts.analysis.ppo_reward_ab` 自动生成。

## 机械检查（全部通过）

| 检查 | 结果 |
| --- | --- |
| `resolved_configs_match_except_reward` | passed |
| `resolved_config_matches_matrix` | passed |
| `initial_policy_hashes_match` | passed |
| `noise_seeds_match` | passed |
| `policy_action_seeds_match` | passed |
| `configured_update_count_completed` | passed（两条均 4 updates） |
| `both_policies_updated` | passed（A: `0496…`→`66579d8b…`；B: `0496…`→`76288cd8…`） |
| `both_frozen_planners_unchanged` | passed |
| `pre_update_collection_is_exactly_paired` | passed（update 0 的全部 common pair 字段逐值相等） |

`mechanical_status = passed`，`review_status = pending_human_review`。

## 效果窗口指标（updates 1–3，96 transitions × profile）

| 指标 | builtin (A) | energy (B) | 变化 |
| --- | ---: | ---: | ---: |
| longitudinal action mean (`g[:,1]`) | -0.020527 | -0.027284 | -0.006756 |
| fuel proxy mL/km | 48.7945 | 48.7960 | +3.06e-5 (fraction) |
| mean speed m/s | 11.2368 | 11.2372 | +4.04e-5 (fraction) |
| route progress delta | 0.61319 | 0.61319 | +6.03e-6 (fraction) |
| stopped fraction | 0.0 | 0.0 | 0 |
| collision count | 0 | 0 | 0 |
| out-of-road count | 0 | 0 | 0 |
| distance m | 107.873 | 107.877 | — |

review_thresholds 全部未被触发：`longitudinal_action_changed=false`、
`energy_intensity_changed=false`、`progress_regressed=false`、
`mean_speed_regressed=false`、`collision_regressed=false`、
`out_of_road_regressed=false`、`energy_drop_confounded_by_progress_drop=false`。
没有任何“少走、停车或降速降耗”的迹象。

## PPO 诊断（全部 finite）

| update | profile | policy loss | value loss | entropy | approx KL | max pre-clip grad |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | builtin | 0.1821 | 0.0246 | 1.1375 | 7.62e-4 | 0.447 |
| 1 | builtin | 0.3395 | 0.0253 | 1.1430 | 8.99e-4 | 0.786 |
| 2 | builtin | 0.1232 | 0.0185 | 1.1512 | 1.76e-3 | 1.113 |
| 3 | builtin | 0.2321 | 0.0253 | 1.1552 | -2.23e-4 | 1.300 |
| 0 | energy | 0.1819 | 16.4852 | 1.1374 | 8.69e-4 | 14.959 |
| 1 | energy | 0.3328 | 14.2304 | 1.1474 | 1.82e-3 | 32.860 |
| 2 | energy | 0.1174 | 13.3450 | 1.1678 | 4.74e-3 | 59.205 |
| 3 | energy | 0.2146 | 14.9905 | 1.1788 | -7.06e-4 | 71.191 |

energy profile 的 value loss 与 gradient norm 比 builtin 大约三个数量级，直接来自
`plannerrft_energy_v1` 的 reward scale（每个 transition 总 reward ≈ 27，而 builtin ≈ 1.5），
不构成数值不稳定：KL、entropy、clip_fraction 与 policy loss 量级与 builtin 一致，
PPO 全部 4 个 update 都成功完成且 summary 全字段 finite。该量级差异是 reward 设计的预期结果，
不应被解释为训练不稳定；后续阶段 B 若需更稳定 value 学习，应通过 PPO 超参数（如 value loss
coefficient 或 reward normalization）显式调整，而非在 reward 公式中隐式缩放。

## 结论边界

本记录支持：

- 严格匹配的 builtin/energy A/B 在当前 commit、本机、两个 no-traffic 场景、1 个 training seed
  和 4 个 update 下，pairing、artifact schema、有限值门槛和短期方向门控全部成立。
- 4 个 update 内未观察到 reward hacking：energy intensity、progress、mean speed、stopped
  fraction、collision、out-of-road 与 builtin 在 deadband 内一致。
- update 0 的 pre-update collection 在两条 run 之间逐值相等，证明 A/B 的初始 policy、
  noise/action seeds 和初始 rollout 严格匹配。

本记录不支持：

- reward 有效性、节能改善、收敛性或跨 seed 稳定性结论——阶段 A 只有 1 个 seed、4 个 update；
- PlannerRFT/nuPlan scorer parity、真实车辆能耗、动力学可执行性或舒适性；
- builtin 与 energy 在更长训练或更复杂场景下的相对表现。

阶段 A 通过，Issue #59 阶段 B 仍需独立登记。如阶段 B 出现 energy proxy 下降伴随
progress/speed 崩溃，按 Issue 要求记录为 reward 定义失败并停止扩大实验。
