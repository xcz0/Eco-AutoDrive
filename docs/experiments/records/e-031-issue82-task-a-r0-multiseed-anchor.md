# E-031 Issue #82 Task A：R0 正式多 seed anchor（A1 = PPO + PlannerRFT R0）

[返回实验索引](../README.md)

**日期 / 类型 / 目的**：2026-09-07 / 正式运行（多 seed 训练 anchor）/ 完成 Issue #82
Task A：用 E-028 稳定 PPO 配置（config-0001）在 matched 协议下训练
`R0`（`plannerrft_no_energy_v1`）3 seeds × 100 updates，建立 A1 作为后续 energy term
因果比较的 no-energy PPO anchor，并对 final checkpoint 做统一 reward-independent
held-out evaluation。本任务不证明 R0 更"好"，也不以 training return 选择 reward。

**代码**：`b1c8079255d5b71c7df674870d25147fbf7fb38a`（完成 Issue #81 Task D，
b1c8079）+ 运行时未提交 diff（本次为 Task A 所做的 seed namespace 扩展：
`protocol.yaml` 的 `training.seed: 0` → `training.seeds: [0, 1, 2]`、runner/CLI 增加
显式 `--training-seed` 选择并拒绝 `runtime.seed` override；完整内容见各 run 的
`tracked_diff.patch`，与随后登记本记录的 commit 一致）。上游源码与模型 checkpoint
见[共同资产](../README.md#共同资产)。

**环境 / 模型**：Windows 10（10.0.26200）；Python 3.10.20；PyTorch 2.12.1+cu126；
Lightning 2.6.5；MetaDrive 0.4.3；resource profile `rtx_a4000`，单卡 `cuda:0`、
`bf16-mixed`，vector rollout 4 workers。checkpoint 为 `checkpoints/DP-Origin/model.pth`
（EMA 276 tensors / 6,042,628 parameters，`frozen_planner_hash` 全程
`6a014ec5…` 不变）。

**配置**：protocol manifest `configs/experiments/scalar_reward/protocol.yaml` 组合
`jobs/training/ppo_conservative` + `components/reward=plannerrft_no_energy_v1`；
训练场景池 S/SC map seeds 0–7（16 scenarios，与 held-out 池 16–23 不相交）；DDIM5
（`ddim_stochasticity=0`）；training seeds `{0, 1, 2}`（每 run 经 `--training-seed`
显式选择，`training.replay_id=0`）；E-028 config-0001 超参经 override 透传：
`ppo.learning_rate=1.6301e-05`、`ppo.target_kl=0.006`、`ppo.batch_size=128`、
`training.transitions_per_environment=8`、`training.update_count=100`、
`ppo.scheduler_total_optimizer_steps=100`；其余为 conservative 默认（`epochs=1`、
`minibatch_size=128`、`gamma=0.99`、`gae_lambda=0.95`、`clip_epsilon=0.2`、
`entropy_coefficient=0.01`、`value_coefficient=0.5`）。batch 组成契约为
`16 scenarios × 8 transitions = 128`（与 E-030 相同、与 E-028 原搜索的 8×16 组成
不同，为已知差异点）。held-out evaluation 复用协议 `policy_job`
（`jobs/evaluation/no_traffic_heldout_policy`）：S/SC seeds 16–23（16 episodes）、
no-traffic、300 步 horizon、0 warmup、DDIM5、runtime seed 760025、
`env.num_scenarios=24`、guidance `orthogonal_policy`、policy action 固定为 Beta
mean——与 E-029 A0 逐项 matched，不读取训练 reward。

**命令**：

```powershell
just scalar-reward train --arm a1 --training-seed <0|1|2> `
  --output-dir outputs/studies/scalar-reward/e-031-issue82-task-a-r0-anchor/a1-r0-seed-<s> `
  --override ppo.learning_rate=1.6301e-05 --override ppo.target_kl=0.006 `
  --override ppo.batch_size=128 --override training.transitions_per_environment=8 `
  --override training.update_count=100 --override ppo.scheduler_total_optimizer_steps=100
# 评测：--arm a1 --checkpoint final --checkpoint-path <run>/policy-final.pt，
#      --output-dir <run>/evaluation/final
```

训练每 seed 约 722–728 s，评测每 seed 约 84–102 s；三个训练 run 与三个评测 run 均
`status: completed`，runner 协议校验（seed ∈ namespace、replay、reward profile、
sampler、场景池、`num_scenarios` 覆盖）全部放行。

**结果**（3 seeds × 100 updates × 128 transitions = 12800，每 update 1 个 optimizer
step；LR schedule 100 步线性走完）：

| Task A 判据 | seed 0 | seed 1 | seed 2 |
| --- | --- | --- | --- |
| 全部 configured updates 完成 | 100/100，12800 transitions | 同左 | 同左 |
| frozen planner hash 不变 | `6a014ec5…` == after | 同左 | 同左 |
| Exploration Policy 更新 | `0496977…` → `42eb258b…` | `feee6d85…` → `a593ae4c…` | `0cd7d936…` → `cbe1fec…` |
| PPO 诊断 finite | 全部 finite（NaN/Inf 为硬失败校验，未触发） | 同左 | 同左 |
| approx KL / clip fraction | \|KL\| ≤ 6.6e-8；clip frac = 0 | ≤ 7.4e-8；0 | ≤ 5.6e-8；0 |
| KL early stop | 0 次 | 0 次 | 0 次 |
| Beta boundary / invalid action | α,β ∈ [1.9983, 2.0013]；boundary mass 与 Beta(2,2) 基线逐项一致，无坍塌 | 同左 | 同左 |
| collision / out-of-road（累计） | 0 / 0 | 0 / 0 | 0 / 0 |
| episode length | 恒为 8（= transitions_per_environment，无 collapse） | 同左 | 同左 |
| stopped fraction | 0.0000 | 0.0000 | 0.0000 |

checkpoint 产物：每 run 保存 `policy-initial.pt`、`policy-update-000..099.pt`
（cadence=1）与 `policy-final.pt`，update 0 / 50 / 100（final）三点可直接比较。
update 0→50→99 训练 rollout 趋势（三 seeds 一致）：mean speed 10.53–10.72 m/s、
distance 134.8–137.2 m、route progress delta 0.839–0.853、comfort 分量 0–0.008，
全程基本平坦；entropy 恒为 1.1361；value loss 缓降（seed 0：7.133→7.076）；probe
`guidance_mean` 位移 ≤ ~1e-3（action 空间 ±1）。probe before 的 α=β=2.0 逐项相同
（seed 化的 init 体现为 initial policy hash 不同）。

**held-out evaluation（final checkpoint，16 episodes × 3 seeds）与 A0（E-029）
对比**：

| 指标 | A0 frozen（E-029） | A1 seed 0 | A1 seed 1 | A1 seed 2 |
| --- | --- | --- | --- | --- |
| arrive_dest | 13/16 | 13/16 | 13/16 | 13/16 |
| max_step 终止 | 3 | 3 | 3 | 3 |
| collision / out-of-road / wrong-direction | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| distance（均） | 169.4 m | 169.5 m | 169.5 m | 169.5 m |
| route completion（均） | 0.946 | 0.947 | 0.947 | 0.947 |
| mean speed（均） | 9.99 m/s | 10.00 m/s | 10.00 m/s | 10.00 m/s |
| stopped fraction（均） | 0.000 | 0.000 | 0.000 | 0.000 |
| proxy energy（均） | 7.839 mL | 7.845 mL | 7.845 mL | 7.845 mL |
| energy intensity（均） | 47.05 mL/km | 47.05 mL/km | 47.05 mL/km | 47.05 mL/km |

per-scenario 高精度核验确认评测真实执行了各自 policy：三 seed 的 final policy
hash 不同（`42eb258b…` / `a593ae4c…` / `cbe1fec…`，job summary 的
`policy_checkpoint.policy_hash` 与训练 summary 一致），per-scenario energy 在
1e-4 mL 量级上互异且与 A0 不同（如 `s16`：A0 4.69183，A1 4.69805 / 4.69839 /
4.69857 mL），但偏离幅度 ≤ ~0.2%/scenario，聚合后差 ≤ 0.1%。

**A1 vs A0 判定**：在 100 updates / E-028 config-0001 / lr 线性衰减至 0 的预算下，
R0 PPO 产生**机械有效但 held-out 行为上不可测（negligible）**的效果——到达率、
终止结构、安全指标与 A0 完全一致，速度/进度/能耗聚合差异 ≤ 0.1%，方向不定
（部分 scenario 微升、部分微降）。这与 E-030 的"clip frac=0、ratio≈1、per-update
KL ~1e-8"极小步长签名一致：anchor 的行为等价于 frozen planner + 近似零 guidance。

**结论边界**：本记录支持：A1 anchor 成立——3 seeds 全部机械健康完成 100 updates
（updates 完成、planner hash 不变、policy 按 seed 更新、诊断 finite、无 Beta
boundary collapse / NaN / KL 异常 / episode-length collapse），final checkpoint 的
matched held-out evaluation 可复现且与 A0 同协议可比；A1 vs A0 在该预算与配置下
无 measurable learned behavioral effect（这是 anchor 的事实属性，不是缺陷）。
不支持：任何"R0 更好/更差"结论；超过 100 updates 或更强学习率下的行为效应；
真实车辆能耗结论（energy 仍为 fuel proxy）；以及与 batch 组成不同（8×16）的
E-028 Stage C 运行的逐值可比性。**对 Issue #82 Task B 的直接含义**：λ sweep 若
沿用同一 conservative PPO 配置与 30–50 updates 预算，energy term 的主效应大概率
同样低于可测阈值（per-update 位移量级 ~1e-3），"所有 λ 与 R0 基本不可区分"是
需要预设的真实风险分支；是否提高 update 预算 / 学习率属于协议决策，须回到
Issue #82 / #80 明确，不在本记录内自行更改。

**产物**：`outputs/studies/scalar-reward/e-031-issue82-task-a-r0-anchor/`（git
忽略），含 `a1-r0-seed-{0,1,2}/`：每 run 含 `resolved_config.yaml`、
`summary.json`、`runtime_metadata.json`、`tracked_diff.patch`、
`policy-initial.pt` / `policy-final.pt`、100 个 `policy-update-NNN.pt`、
`training-state.ckpt` 与 `updates/update-000..099/` 的 16 slot episode audit NPZ；
`evaluation/final/` 含 16 个场景目录的 `trace.npz` + `summary.json` 与 job 级
`summary.json` / `resolved_config.yaml` / `runtime_metadata.json`。

**验证**：代码侧改动（seed namespace 扩展）由
`tests/configuration/test_scalar_reward.py` 覆盖（10 passed，含 namespace 外 seed
拒绝与 `runtime.seed` override 拒绝回归）；`just lint` / `just typecheck` 通过。
