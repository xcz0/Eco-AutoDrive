# E-030 Issue #81 Task D：标准 PPO 目标下的 R0 / Rλ=1 transfer gate

[返回实验索引](../README.md)

**日期 / 类型 / 目的**：2026-09-07 / transfer gate（机械有效性与短程优化稳定性门控）/
完成 Issue #81 Task D：验证 E-028 最终稳定候选 PPO 配置（config-0001）能迁移到
matched 协议下的 R0（`plannerrft_no_energy_v1`）与 Rλ=1（`plannerrft_energy_v1`）
两条训练臂，且不引入新的机械或数值问题。本 gate 不证明 energy improvement。

**代码**：`8ba897fe7a9a0fef2b2a4ea59e13bd8453f57ac3`（Fix PPO log-probability
broadcasting，8ba897f），运行时工作区干净（`runtime_metadata.json` 的
`git_status_short` 为空，`tracked_diff.patch` 为空文件）。**本运行是 shape 广播修复后
的第一次 transfer gate**：此前在 `a80d910` 下完成的同名运行（两臂 20 updates、判据
逐项通过）经核验确认受 pre-existing 交叉配对 PPO 目标缺陷影响，产物已整体改名保留于
`outputs/studies/scalar-reward/e-030-transfer-gate-invalid-cross-paired-ppo/`，不作为
gate 证据；缺陷登记与修复证据见 Issue #81 评论及 8ba897f 回归测试。上游源码与模型
checkpoint 见[共同资产](../README.md#共同资产)。

**环境 / 模型**：Windows 10（10.0.26200）；Python 3.10.20；PyTorch 2.12.1+cu126；
Lightning 2.6.5；MetaDrive 0.4.3；resource profile `rtx_a4000`，单卡 `cuda:0`、
`bf16-mixed`，vector rollout 4 workers。checkpoint 为 `checkpoints/DP-Origin/model.pth`
（EMA 276 tensors / 6,042,628 parameters，`frozen_planner_hash` 全程
`6a014ec5…` 不变）。

**配置**：protocol manifest `configs/experiments/scalar_reward/protocol.yaml` 组合
`jobs/training/ppo_conservative` + `components/reward=<arm profile>`；训练场景池为
S/SC map seeds 0–7（16 scenarios，held-out 池的子集且与其不相交）；DDIM5
（`ddim_stochasticity=0`）；runtime seed 0、`training.replay_id=0`；planner noise 与
policy action 的每场景派生 seed 记录于 `summary.json` 的 `noise_seeds` /
`policy_action_seeds`。E-028 config-0001 超参经 override 透传：
`ppo.learning_rate=1.6301e-05`、`ppo.target_kl=0.006`、`ppo.batch_size=128`、
`training.transitions_per_environment=8`、`ppo.scheduler_total_optimizer_steps=20`；
其余为 conservative 默认（`epochs=1`、`minibatch_size=128`、`gamma=0.99`、
`gae_lambda=0.95`、`clip_epsilon=0.2`、`entropy_coefficient=0.01`、
`value_coefficient=0.5`、`update_count=20`）。batch 组成契约为
`16 scenarios × 8 transitions = 128`，与 E-028 搜索时 config-0001 的
`8 scenarios × 16 transitions` 组成不同（协议场景池固定为 16 个所致），这是本 gate 与
E-028 原组合的已知差异点。

**命令**：

```powershell
just scalar-reward train --arm a1 --output-dir outputs/studies/scalar-reward/e-030-transfer-gate/a1-r0 `
  --override ppo.learning_rate=1.6301e-05 --override ppo.target_kl=0.006 `
  --override ppo.batch_size=128 --override training.transitions_per_environment=8 `
  --override ppo.scheduler_total_optimizer_steps=20
# a2 同理换 --arm a2 与 .../a2-rlam1 输出目录
```

a1 运行约 144 s、a2 约 146 s；两臂均 `status: completed`，runner 在训练前通过全部
协议校验（seed、replay、reward profile、sampler、场景池）。

**结果**（两臂各 20 updates × 128 transitions = 2560，每 update 1 个 optimizer step）：

| Gate 判据 | a1 R0 | a2 Rλ=1 |
| --- | --- | --- |
| 全部 configured updates 完成 | 20/20，`total_transitions=2560` | 20/20，`total_transitions=2560` |
| frozen planner hash 不变 | `6a014ec5…` == after | 同左 |
| Exploration Policy 更新 | `0496977…` → `5db2902…` | `0496977…` → `388508f…` |
| PPO loss / value loss / entropy / approx KL / grad norm finite | 全部 finite | 全部 finite |
| approx KL / clip fraction | \|KL\| ≤ 3.2e-8；clip frac = 0.0000 | \|KL\| ≤ 2.9e-8；clip frac = 0.0000 |
| policy ratio | mean ≈ 1.0000，max ≤ 1.0004 | mean ≈ 1.0000，max ≤ 1.0004 |
| Beta boundary / invalid action | action ∈ [-0.9802, 0.9862] ⊂ (-1, 1)；α,β ≥ 1.9995 | 同左 |
| probe boundary mass（诊断） | 0.020996 before == after（Beta(2,2) 基线） | 同左 |
| collision / out-of-road | 0 / 0（20 updates 累计） | 0 / 0 |
| episode length | 恒为 8（无 collapse） | 恒为 8 |
| stopped fraction | 0.0000（全部 updates） | 0.0000 |
| mean speed / progress delta / distance | 10.58–10.76 m/s；0.849→0.850；136.5–136.9 m | 同左 |
| total_reward（首→末 update） | 111.80 → 111.86 | 108.12 → 108.18 |

value loss 首末 update 分别为 7.133 → 7.134（a1）与 6.672 → 6.673（a2）；entropy 恒为
1.13611；explained variance ≈ -1.2e-4；pre-clip grad norm 4.66–4.93；无 KL early stop；
LR schedule 20 步线性走完（末 update `final_learning_rate=0`）。fuel proxy 全程
47.77–47.83 mL/km。

**reward profile 语义交叉验证**：首 update 分量均值 ttc=1.0、progress≈0.9950、
comfort=0、speed=1.0、energy≈0.3851，则 R0 total = (5+4.975+4)/16 × 128 = 111.80，
Rλ=1 total = (5+4.975+4+0.3851)/17 × 128 = 108.12，与实测逐位一致——两 profile 的
唯一目标差异确为 energy 项与归一化分母（a1 的 energy 分量作为未加权 audit 诊断记录，
数值与 a2 相同，因两臂 seed/rollout 一致）。

**修复生效证据**：本次运行 `mean_clip_fraction = 0.0000` 与
`policy_ratio ≈ 1.0000±0.0004` 自洽；受缺陷影响的旧运行在相同配置下呈现
clip fraction 0.65–0.74 与 KL ~1e-8 并存的矛盾签名（见 Issue #81 缺陷登记）。

**验证**：运行前 `just test-target
tests/training/test_ppo.py::test_ppo_pairs_each_action_with_its_behavior_log_probability`
1 passed（8ba897f 回归测试，证明逐样本 importance ratio 配对）；两臂训练由
`scalar-reward train` 协议校验放行后完成。本记录为运行登记，无代码改动。

**产物**：`outputs/studies/scalar-reward/e-030-transfer-gate/`（git 忽略），含
`a1-r0/` 与 `a2-rlam1/`；每臂含 `resolved_config.yaml`、`summary.json`、
`runtime_metadata.json`、`tracked_diff.patch`（空）、`policy-initial.pt` /
`policy-final.pt`、20 个 `policy-update-NNN.pt`、`training-state.ckpt` 与
`updates/update-000..019/` 的 16 slot episode audit NPZ。受缺陷影响的旧运行保留于
`outputs/studies/scalar-reward/e-030-transfer-gate-invalid-cross-paired-ppo/`
（git_head `a80d910`）。

**结论边界**：本记录支持：E-028 config-0001 PPO 配置在标准（修复后）PPO 目标下可
迁移到 matched 协议的 R0 与 Rλ=1 训练臂——20 updates 内机械有效（updates 完成、
frozen planner 不变、policy 更新、诊断 finite、无 Beta boundary / invalid action）且
短程稳定（无 collision / out-of-road / episode-length collapse，无 stopped / speed /
progress 异常，无 reward hacking 迹象）；Issue #80 可进入正式 reward ablation。不
支持：任何 energy improvement 或 R0 vs Rλ 行为差异结论（两臂 rollout 统计在 20
updates 内几乎相同，per-update KL ~1e-8 量级，policy 位移极小）；超过 20 updates 的
长程稳定性（由 E-028 在其组成下 100 updates 的 3-seed 证据支持，非本 gate 范围）；
真实车辆能耗结论（energy 仍为 fuel proxy）；以及与 batch 组成不同（8×16）的 E-028
原运行的逐值可比性。
