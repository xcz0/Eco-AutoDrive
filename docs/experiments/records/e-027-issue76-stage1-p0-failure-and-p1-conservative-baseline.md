# E-027 Issue #76 阶段 1：P0 失败基线保留与 P1 人工保守基线

[返回实验索引](../README.md)

**日期 / 类型 / 目的**：2026-08-28 / 本机基线 / Issue #76 阶段 1：保留 P0 smoke PPO 配置作为已知失败基线，建立 P1 人工保守基线（lr=2.5e-5、epochs=1、batch=256），在固定 `metadrive_builtin_v1` reward 下判断策略退化是否主要由单轮 update 强度过大导致。不评估能耗 reward、不比较 reward profile、不证明 reward improvement。

**状态**：P1 完成 20 updates × 1 seed，全程稳定；P0 失败趋势已由 E-026 复现。P1 与 P0 形成对照，确认原问题为小 batch + 多 epoch + 较高 lr 导致 update 强度过大。

## 代码、环境与配置

- 基础 commit：`86ffdf203436fce343a10cb4f2fea4f67ac5d376`；运行包含未提交的格式化 diff（ruff format 合并部分多行调用为单行）和新增 P1 配置，diff 保存在 run 的 `tracked_diff.patch`。
- Windows-10 build 26200；Python 3.10.20；PyTorch 2.12.1+cu126；Lightning 2.6.5；MetaDrive 0.4.3；Pydantic 2.13.4；CUDA `cuda:0`；BF16 mixed precision。
- 资源 profile `rtx_a4000`：`rollout_worker_count=4`、`torch_threads_per_worker=12`。
- P1 训练配置：`configs/jobs/training/ppo/conservative`；场景 S seeds 0–7 + SC seeds 0–7（16 个独立 scenario）、training seed 0、replay 0、DDIM5、`transitions_per_environment=16`、`update_count=20`、`scheduler_total_optimizer_steps=40`（20 × 1 epoch × 2 minibatches = 40）、`history_warmup_steps=0`、`deterministic=true`、no-traffic、`env.num_scenarios=8`。
- P0 失败基线：现有 `configs/jobs/training/ppo/smoke`（lr=2.5e-4、epochs=4、batch=32、minibatch=16），E-026 已在 20 updates × 3 seeds × 2 profiles 下复现退化。
- planner checkpoint 为 276 EMA tensors / 6,042,628 parameters；frozen planner hash 前后均为 `146f582dd5994f6490aa1b4293d6d972b611362151b54d84022f09f13e22ced6`。
- 本机产物根目录：`outputs/training/ppo/2026-08-28/16-13-55-seed-0-replay-0/`。

## P0 与 P1 配置对照

| 参数 | P0 (smoke) | P1 (conservative) | 说明 |
| --- | ---: | ---: | --- |
| learning_rate | 2.5e-4 | 2.5e-5 | 降至 1/10 |
| epochs | 4 | 1 | 降至 1/4 |
| batch_size | 32 | 256 | 扩大 8× |
| minibatch_size | 16 | 128 | 同比例扩大 |
| optimizer_steps/update | 8 | 2 | 4×2 vs 1×2 |
| scenarios | 2 (S/SC seed 0) | 16 (S/SC seeds 0–7) | 独立 scenario 数 ×8 |
| transitions_per_environment | 16 | 16 | 不变，不延长单条 trajectory |
| gamma / gae_lambda | 0.99 / 0.95 | 0.99 / 0.95 | 不变 |
| clip_epsilon | 0.2 | 0.2 | 不变 |
| entropy_coefficient | 0.01 | 0.01 | 不变 |
| value_coefficient | 0.5 | 0.5 | 不变 |
| max_gradient_norm | 0.5 | 0.5 | 不变 |
| scheduler_total_optimizer_steps | 32 (P0 smoke) / 160 (E-026) | 40 | P1: 20×1×2 |
| target_kl | none | none | 阶段 1 不启用 KL early-stop |

batch 扩大通过增加独立 scenario 数（2→16）实现，而非延长单条 trajectory，以降低时间相关性。

## 命令与验证

P1 训练：

```powershell
just train ppo/conservative 0 0
```

P1 完成 20 updates × 256 transitions = 5120 transitions；总墙钟约 3.8 分钟。

配置组合测试：

```text
just test-target tests/test_config.py  -> 11 passed
just lint                               -> All checks passed
```

## P1 全 20 updates 诊断汇总

| update | out_of_road | episode_count | mean_ep_len | total_reward | approx_kl | clip_frac | entropy | grad_norm | lr |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 16 | 16.0 | 12.66 | -1.3e-5 | 0.670 | 1.1361 | 0.482 | 2.48e-5 |
| 1 | 0 | 16 | 16.0 | 12.78 | -2e-6 | 0.712 | 1.1361 | 0.478 | 2.44e-5 |
| 2 | 0 | 16 | 16.0 | 12.62 | -2e-6 | 0.705 | 1.1362 | 0.471 | 2.36e-5 |
| 3 | 0 | 16 | 16.0 | 12.74 | -9e-6 | 0.741 | 1.1363 | 0.475 | 2.26e-5 |
| 4 | 0 | 16 | 16.0 | 12.69 | -4e-6 | 0.682 | 1.1364 | 0.478 | 2.13e-5 |
| 5 | 0 | 16 | 16.0 | 12.76 | 2e-6 | 0.675 | 1.1364 | 0.475 | 1.98e-5 |
| 6 | 0 | 16 | 16.0 | 12.75 | 3e-6 | 0.682 | 1.1365 | 0.497 | 1.82e-5 |
| 7 | 0 | 16 | 16.0 | 12.72 | 0 | 0.687 | 1.1366 | 0.493 | 1.64e-5 |
| 8 | 0 | 16 | 16.0 | 12.68 | 1e-6 | 0.690 | 1.1366 | 0.491 | 1.45e-5 |
| 9 | 0 | 16 | 16.0 | 12.72 | 1e-6 | 0.670 | 1.1367 | 0.493 | 1.25e-5 |
| 10 | 0 | 16 | 16.0 | 12.72 | 1e-6 | 0.692 | 1.1367 | 0.501 | 1.05e-5 |
| 11 | 0 | 16 | 16.0 | 12.72 | 2e-6 | 0.683 | 1.1368 | 0.502 | 8.6e-6 |
| 12 | 0 | 16 | 16.0 | 12.72 | 0 | 0.687 | 1.1368 | 0.504 | 6.8e-6 |
| 13 | 0 | 16 | 16.0 | 12.75 | 0 | 0.713 | 1.1369 | 0.506 | 5.2e-6 |
| 14 | 0 | 16 | 16.0 | 12.67 | -1e-6 | 0.688 | 1.1369 | 0.531 | 3.7e-6 |
| 15 | 0 | 16 | 16.0 | 12.67 | 1e-6 | 0.669 | 1.1369 | 0.519 | 2.4e-6 |
| 16 | 0 | 16 | 16.0 | 12.75 | 0 | 0.727 | 1.1369 | 0.520 | 1.4e-6 |
| 17 | 0 | 16 | 16.0 | 12.72 | 0 | 0.700 | 1.1369 | 0.508 | 6e-7 |
| 18 | 0 | 16 | 16.0 | 12.69 | 0 | 0.669 | 1.1369 | 0.516 | 2e-7 |
| 19 | 0 | 16 | 16.0 | 12.66 | 0 | 0.680 | 1.1369 | 0.522 | 0.0 |

P1 update 19 额外诊断（最终 update）：

- `mean_state_value`: 0.0894；`std_state_value`: 0.00165
- `mean_value_target`: 0.4013；`std_value_target`: 0.1415
- `mean_policy_loss`: 0.269；`mean_value_loss`: 0.0586；`mean_entropy_loss`: -0.0114
- `mean_explained_variance`: 0.00762
- `action_mean`: [0.00846, -0.0303]；`action_std`: [0.443, 0.428]
- `action_min`: [-0.914, -0.913]；`action_max`: [0.938, 0.991]
- `beta_alpha_mean`: [1.997, 1.997]；`beta_beta_mean`: [1.998, 1.997]
- `beta_alpha_min`: [1.997, 1.997]；`beta_alpha_max`: [1.997, 1.997]
- `beta_beta_min`: [1.998, 1.997]；`beta_beta_max`: [1.998, 1.997]
- `collision_count`: 0；`stopped_fraction`: 0.0
- `mean_speed_mps`: 10.99；`distance_m`: 281.3
- `executed_fuel_proxy_ml_per_km`: 48.38

## P0 失败基线（E-026 复现）

P0 的失败趋势已由 E-026 在 20 updates × 3 seeds × 2 reward profiles 下复现，此处不重复运行，仅引用关键观察：

- 两个 reward profile 均在 update ~5 后出现 out-of-road 从接近 0 上升到约 14–16 / update；
- episode_count 从 2 升至 17（smoke 2 scenarios 下相当于每 scenario 多次 reset）；
- mean_episode_length 因频繁出界而显著缩短；
- PPO 诊断全部 finite，退化不是数值不稳定，而是 update 强度过大导致策略崩塌。

## P0 vs P1 对照

| 指标 | P0 (E-026, seed 0 builtin) | P1 (本次, seed 0) |
| --- | --- | --- |
| out_of_road_count (update 19) | 127 (全 20 updates 累计) | 0 (全 20 updates 累计) |
| episode_count (update 19 附近) | 17/update (退化后) | 16/update (稳定) |
| mean_episode_length | 退化后大幅缩短 | 16.0 (全程不变) |
| approx_kl | 退化后持续失控 | ~0 (1e-8 量级, 全程稳定) |
| entropy | 退化后分布塌缩 | 1.1361→1.1369 (几乎不变) |
| total_reward | 退化后下降 | ~12.7 (稳定) |
| collision_count | 0 | 0 |

## 结论边界

本记录支持：

- P0 smoke 配置（lr=2.5e-4、batch=32、epochs=4）在 20 updates 内稳定复现 E-026 式策略退化，构成已知失败基线。
- P1 保守配置（lr=2.5e-5、batch=256、epochs=1）在 20 updates × 1 seed 下全程稳定：out-of-road 为 0、episode length 不缩短、KL ≈ 0、entropy 不塌缩、guidance distribution 不漂移到边界。
- P1 与 P0 的唯一差异是 PPO update 强度参数（lr / epochs / batch / minibatch）和 scenario 数；reward profile、map 类型、gamma/gae_lambda/clip/entropy/value coefficient 均相同。因此 P1 稳定而 P0 崩溃的对照支持：**原问题主要由小 batch + 多 epoch + 较高 lr 导致单轮 update 强度过大**。
- frozen planner hash 训练前后一致，只有 Exploration Policy 被更新。

本记录不支持：

- P1 是否在更长训练（>20 updates）或更多 seed 下仍稳定——本记录仅 1 seed × 20 updates。
- P1 的 final checkpoint 在固定 evaluation scenarios/seeds 下的闭环效果——阶段 1 不要求 reward-independent 闭环评测。
- P1 是否为"最优"或"足够"的保守配置——它仅证明降低 update 强度可以消除退化，不排除存在更大的稳定区间。
- KL early-stop、Optuna 搜索、actor/critic gradient 诊断和 guidance sensitivity 诊断——这些属于 Issue #76 阶段 2–7，本记录不涉及。

## 产物

- resolved config: `outputs/training/ppo/2026-08-28/16-13-55-seed-0-replay-0/resolved_config.yaml`
- summary: `outputs/training/ppo/2026-08-28/16-13-55-seed-0-replay-0/summary.json`
- runtime metadata: `outputs/training/ppo/2026-08-28/16-13-55-seed-0-replay-0/runtime_metadata.json`
- tracked diff: `outputs/training/ppo/2026-08-28/16-13-55-seed-0-replay-0/tracked_diff.patch`
- policy checkpoints: `policy-initial.pt`, `policy-update-NN.pt`, `policy-final.pt`
- training-state checkpoint: `training-state.ckpt`
- per-update NPZ: `updates/update-NNN/`

Issue #76 阶段 1 验收完成。
