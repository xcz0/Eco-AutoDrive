# E-032 Issue #82 Task B：Energy weight coarse sweep（λ = 1/2/4/8）

[返回实验索引](../README.md)

**日期 / 类型 / 目的**：2026-09-07 / 正式运行（coarse sweep）/ 完成 Issue #82
Task B：在 matched 协议下固定其他全部因素，仅改变 energy weight
`λ ∈ {1, 2, 4, 8}`（reward `Rλ = Gate × (5·TTC + 5·Progress + 2·Comfort +
4·Speed + λ·Energy)/(16+λ)`，即 `reward.weights.energy`），每 λ 1 training
seed × 50 updates 训练并对 final checkpoint 做统一 reward-independent held-out
evaluation，为 Task C 的 trade-off 分析与 Day 3 候选筛选提供 artifact。本任务不调整
PPO 超参，不以 training return 选择 reward，也不证明任何 λ 更"好"。

**代码**：`630a736`（完成 Issue #82 Task A，630a736）；运行时无未提交 diff
（各 run 的 `tracked_diff.patch` 为 0 字节，`git status --short` 干净）。上游源码与
模型 checkpoint 见[共同资产](../README.md#共同资产)。

**环境 / 模型**：Windows 10（10.0.26200）；Python 3.10.20；PyTorch 2.12.1+cu126；
Lightning 2.6.5；MetaDrive 0.4.3；resource profile `rtx_a4000`，单卡 `cuda:0`、
`bf16-mixed`，vector rollout 4 workers。checkpoint 为 `checkpoints/DP-Origin/model.pth`
（EMA 276 tensors / 6,042,628 parameters，`frozen_planner_hash` 全程
`6a014ec5…` 不变）。

**配置**：protocol manifest `configs/experiments/scalar_reward/protocol.yaml` 组合
`jobs/training/ppo_conservative` + `components/reward=plannerrft_energy_v1`（arm
`a2`），λ 经 `--override reward.weights.energy=<λ>.0` 逐 run 设置（其余权重
5/5/2/4 固定，分母 `16+λ` 由权重和属性自动成立）；训练场景池 S/SC map seeds 0–7
（16 scenarios，与 held-out 池 16–23 不相交）；DDIM5（`ddim_stochasticity=0`）；
training seed `0`（四个 λ run 共用，`training.replay_id=0`）；E-028 config-0001
超参经 override 透传且四 run 完全一致：`ppo.learning_rate=1.6301e-05`、
`ppo.target_kl=0.006`、`ppo.batch_size=128`、`training.transitions_per_environment=8`、
`training.update_count=50`（Issue 预算 30~50 的上限，四 λ 相同预算）、
`ppo.scheduler_total_optimizer_steps=50`；其余 conservative 默认（`epochs=1`、
`minibatch_size=128`、`gamma=0.99`、`gae_lambda=0.95`、`clip_epsilon=0.2`、
`entropy_coefficient=0.01`、`value_coefficient=0.5`）。batch 组成契约为
`16 scenarios × 8 transitions = 128`（与 E-030/E-031 相同）。四个 run 的
`resolved_config.yaml` 经逐行 diff 仅 `reward.weights.energy` 一行不同。held-out
evaluation 复用协议 `policy_job`（`jobs/evaluation/no_traffic_heldout_policy`）：
S/SC seeds 16–23（16 episodes）、no-traffic、300 步 horizon、0 warmup、DDIM5、
runtime seed 760025、`env.num_scenarios=24`、guidance `orthogonal_policy`、
policy action 固定 Beta mean——与 E-029 A0 / E-031 A1 逐项 matched。

**命令**：

```powershell
just scalar-reward train --arm a2 --training-seed 0 `
  --output-dir outputs/studies/scalar-reward/e-032-issue82-task-b-lam-sweep/a2-rlam<λ>-seed-0 `
  --override reward.weights.energy=<λ>.0 `
  --override ppo.learning_rate=1.6301e-05 --override ppo.target_kl=0.006 `
  --override ppo.batch_size=128 --override training.transitions_per_environment=8 `
  --override training.update_count=50 --override ppo.scheduler_total_optimizer_steps=50
# 评测：just scalar-reward evaluate-policy --arm a2 --checkpoint final `
#   --checkpoint-path <run>/policy-final.pt --output-dir <run>/evaluation/final
```

训练每 run 约 367–370 s，评测每 run 约 119–121 s；四个训练 run 与四个评测 run 均
`status: completed`，runner 协议校验（seed namespace、replay、reward profile、
sampler、场景池、`num_scenarios` 覆盖、评测 policy/model/map_query_radius 匹配）
全部放行。

**结果**（4 runs × 50 updates × 128 transitions = 6400 transitions each，每 update
1 个 optimizer step，LR schedule 50 步线性走完）：

Task B 匹配判据逐项核验：

| Task B 判据 | 核验结果 |
| --- | --- |
| 所有 λ 使用同一 initial policy checkpoint | 四 run `initial_policy_hash` 均为 `04969773…`，与 E-031 A1 seed-0 的 initial 完全一致（同 training seed ⇒ 同初始化，E-030 已有实证） |
| scenario/map seed、planner noise、policy action seed、transition 数严格匹配 | 四 run 的 `noise_seeds`（16 per-slot）与 `policy_action_seeds`（16）逐值相等；update-0 rollout 轨迹层指标逐位一致（distance 136.47 m、mean speed 10.66 m/s、route delta 0.850、action mean/std、fuel proxy 6.52 mL 全同），仅 reward 随 λ 缩放；`total_transitions` 均 6400 |
| 不调整 PPO 参数 | 四 run resolved config 逐行 diff 仅 `energy: λ` 一行不同；lr/epochs/batch/minibatch/target_kl/entropy/value 系数全部一致 |
| 相同 checkpoint cadence | 每 run 保存 `policy-initial.pt` + `policy-update-000..049.pt`（cadence=1，各 50 个）+ `policy-final.pt` + `training-state.ckpt` |
| 同一 held-out evaluation matrix | 四 run 共用 `no_traffic_heldout_policy` 协议（S/SC 16–23、seed 760025、300 步、DDIM5、Beta mean、orthogonal guidance） |

机械健康（四 λ run 一致）：50/50 updates 完成；frozen planner hash
`6a014ec5…` 不变；policy 按 λ 更新（final hash 分别为 `ebfeb95d…` / `edd1366e…`
/ `767f1aba…` / `ce7a7871…`）；诊断全部 finite；|approx KL| ≤ 5.7e-8，clip
fraction = 0，KL early stop 0 次；Beta α,β ∈ [1.9991, 2.0009]，boundary mass 与
Beta(2,2) 基线一致，无坍塌；collision / out-of-road 累计 0 / 0；episode length
恒为 8；stopped fraction 0；entropy 恒为 1.1361；value loss 随 λ 单调缩放
（update 0：6.67 / 6.28 / 5.63 / 4.72，量级与 `(16+λ)` 归一化后的 value target
尺度一致），update 0→49 无趋势漂移。

update-0 reward 公式交叉验证（policy 更新前的纯 reward 缩放）：分量均值
ttc=1.0、progress=0.99501、comfort=0、speed=1.0、energy=0.38515，
`total_reward(λ) = 128 × (13.97505 + λ·0.38515)/(16+λ)`：

| λ | energy 占比 | update-0 total_reward | 公式预测 |
| --- | --- | --- | --- |
| 1 | 5.9% | 108.1238 | 108.124 |
| 2 | 11.1% | 104.8558 | 104.856 |
| 4 | 20.0% | 99.3001 | 99.300 |
| 8 | 33.3% | 90.9666 | 90.967 |

probe（16 slot 前后对照）：`probe_before` 四 run 逐位相同（α=β=2.0，
guidance_mean=0）；`probe_after` guidance_mean 位移 ~4.2e-4（lateral）/
~2.7e-4（longitudinal），λ 间差异在 1e-6 量级（λ=1：-4.163e-4 / -2.693e-4；
λ=8：-4.174e-4 / -2.655e-4）——方向一致、幅度可忽略。

**held-out evaluation（final checkpoint，16 episodes × 4 λ）**：

| 指标 | A0 frozen（E-029） | A1 seed 0（E-031，100 updates） | λ=1 | λ=2 | λ=4 | λ=8 |
| --- | --- | --- | --- | --- | --- | --- |
| arrive_dest / time_truncation | 13 / 3 | 13 / 3 | 13 / 3 | 13 / 3 | 13 / 3 | 13 / 3 |
| collision / out-of-road / wrong-direction | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| distance（均） | 169.4 m | 169.5 m | 169.508 m | 169.508 m | 169.508 m | 169.508 m |
| route completion（均） | 0.946 | 0.947 | 0.9467 | 0.9467 | 0.9467 | 0.9467 |
| mean speed（均） | 9.99 m/s | 10.00 m/s | 9.9998 | 9.9998 | 9.9998 | 9.9998 |
| stopped fraction（均） | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| proxy energy（均） | 7.839 mL | 7.845 mL | 7.84522 | 7.84522 | 7.84523 | 7.84523 |
| energy intensity（均） | 47.05 mL/km | 47.05 | 47.0493 | 47.0493 | 47.0493 | 47.0493 |

per-scenario 高精度核验确认评测真实执行了各自 policy：四个 job summary 的
`policy_checkpoint.policy_hash` 与对应训练 run 的 `final_policy_hash` 一致且互异；
per-scenario energy 在 1e-6~1e-5 mL 量级上互异（如 `|E_λ8 − E_λ1|` 最大
8e-6 mL），但聚合偏离 ≤ 0.01%。未到达场景（sc16/sc18/sc23）与终止结构与
A0/A1 相同。

**判定**：在 50 updates / E-028 config-0001 / lr 线性衰减至 0 的预算下，λ ∈
{1, 2, 4, 8} 的 sweep **机械有效但 held-out 行为上与 R0 及彼此均不可区分**
（`no measurable energy-term effect` 分支）——到达率、终止结构、安全指标完全
一致，速度/进度/能耗聚合差异 ≤ 0.01%，无任何随 λ 单调或局部改善的 energy
信号；训练侧唯一随 λ 变化的是 reward/value 的尺度（归一化分母所致），policy
位移（probe guidance_mean ~4e-4、KL ~1e-8、clip frac=0）与 E-031 anchor 的
极小步长签名一致。这印证了 E-031 对 Task B 的风险预判：该 conservative 配置下
per-update 位移量级 ~1e-3，energy term 主效应低于可测阈值。

**结论边界**：本记录支持：λ sweep artifact 成立——四 λ run 匹配协议逐项满足
（同 initial policy、同 seeds/noise/action streams、同 PPO 超参、同 checkpoint
cadence、同 held-out 矩阵）、机械健康、可复现；在该预算与配置下所有 λ 与 R0
基本不可区分，按 Issue #82 候选筛选规则应记录为 `no measurable energy-term
effect`。不支持：任何 λ 排序或"最佳 λ"；超过 50 updates、更高 lr 或不同 PPO
配置下的 λ 主效应（是否提高预算/lr 属协议决策，须回到 Issue #82/#80）；真实
车辆能耗结论（energy 仍为 fuel proxy）；λ=1 与 E-030 transfer gate run 的逐值
可比性（后者 20 updates）。Day 3 候选 λ 的正式筛选与统一 trade-off summary 由
Task C 基于本 artifact 完成。

**产物**：`outputs/studies/scalar-reward/e-032-issue82-task-b-lam-sweep/`（git
忽略），含 `a2-rlam{1,2,4,8}-seed-0/`：每 run 含 `resolved_config.yaml`、
`summary.json`、`runtime_metadata.json`、`tracked_diff.patch`（空）、
`policy-initial.pt` / `policy-final.pt`、50 个 `policy-update-NNN.pt`、
`training-state.ckpt` 与 `updates/update-000..049/` 的 16 slot episode audit
NPZ；`evaluation/final/` 含 16 个场景目录的 `trace.npz` + `summary.json` 与
job 级 `summary.json` / `resolved_config.yaml` / `runtime_metadata.json`。

**验证**：本次无代码改动（tracked diff 为空、工作树干净），不涉及代码测试；
验证限于各 run 的 runner 协议校验放行、上述匹配判据逐项核验与 update-0 reward
公式逐位交叉验证。
