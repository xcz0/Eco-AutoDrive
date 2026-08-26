# E-024 `plannerrft_energy_v1` 单次真实 PPO smoke

[返回实验索引](../README.md)

**日期 / 类型 / 目的**：2026-08-26 / 本机实现验收 / 验证 Issue #59 新 reward profile 从
MetaDrive execution、TorchRL vector rollout、GAE/PPO 到 typed audit、NPZ 和 summary 的一次真实
更新链路。按任务边界不运行 builtin/energy A/B。

**状态**：单次 update、双能耗审计、artifact schema 和有限值门槛通过；Issue #59 保持打开。

## 代码、环境与配置

- 基础 commit：`f4dfe07e40c3bfb47682334d58643441b5df7faa`；运行包含 Issue #59 当前未提交
  实现，完整 dirty state 与 patch 保存在运行目录的 `runtime_metadata.json` 和
  `tracked_diff.patch`。
- Windows 10 build 26200；Python 3.10.20；PyTorch 2.12.1+cu126；Lightning 2.6.5；
  MetaDrive 0.4.3；CUDA `cuda:0`；BF16 mixed precision。
- `ppo_energy_smoke`：no-traffic、history warmup 0、场景 `S` / `SC` seed 0、training seed 0、
  replay 0、DDIM5、32 transitions、2 episodes、1 update。planner checkpoint 为 276 EMA tensors / 
  6,042,628 parameters。
- reward profile 为 `plannerrft_energy_v1`；权重 `5/5/2/4/1`，energy reference
  50 mL/km，minimum step distance 0.01 m。完整阈值以运行目录 `resolved_config.yaml` 为准。
- 本机忽略产物目录：
  `outputs/training/ppo-energy/2026-08-26/16-44-59-seed-0-replay-0`。

## 命令与验证

```powershell
just train ppo_energy_smoke 0 0
```

训练成功返回 `status=completed`。随后只读检查两个 rollout NPZ：每个严格包含 63 个 energy
profile 字段，`reward_profile` 均正确，全部浮点数组有限；每个 transition 的 scalar reward 与
`gate * (5*TTC + 5*Progress + 2*Comfort + 4*Speed + Energy) / 17` 在
`rtol=1e-6, atol=1e-6` 下逐值一致。两个文件的 fuel proxy 合计与 summary 精确一致；按文件分别
求和再与 batched summary reward 比较只有 float32 reduction order 的约 `9.5e-7` 差异。

实现回归同时通过：

```text
just test      -> 276 passed, 41 deselected
just test-sim  -> 29 passed, 288 deselected
just lint      -> All checks passed
just typecheck -> 0 errors, 0 warnings
```

## 单次 update 结果

| 字段 | 值 |
| --- | ---: |
| transitions / episodes / updates | 32 / 2 / 1 |
| total reward | 27.080097 |
| actual execution distance | 33.838562 m |
| native MetaDrive step energy total | 0.0 mL |
| execution-recomputed fuel proxy | 1.609623 mL |
| execution-recomputed intensity | 47.567704 mL/km |
| mean policy / value / total loss | 0.181898 / 16.485199 / 16.655722 |
| mean approximate KL / entropy | 0.000869 / 1.137444 |
| maximum pre-clip gradient norm | 14.958571 |
| final learning rate | 0.000213388 |

PPO update 的 loss、KL、entropy、gradient 和 summary reward/energy 均有限。完整 batch advantage
normalization 会拒绝非有限 GAE 统计，PPO 每步也会拒绝非有限 loss/diagnostic；本次 update 通过了
这些运行时门槛。

初始 policy hash 为
`049697739ab5d6e7bd212938bbac82c35215eb9ed14c02557952098a9718d05d`，最终为
`bbc39fa47f845dc0c41bf26c879f46294b414d3f0eb9ae3838fe9e3f27e4a0f5`，证明 policy 已更新。
frozen planner hash 前后均为
`146f582dd5994f6490aa1b4293d6d972b611362151b54d84022f09f13e22ced6`。

component means 为：gate/collision/drivable/wrong-direction/TTC/progress/speed 均为 1.0，comfort
为 0.0，energy 为 0.386302；无 collision 或 out-of-road。comfort 为 0 符合当前运动学 waypoint
执行与真实车辆阈值组合下的已接受预期，不能解释为真实车辆舒适性测量。

## 结论边界

本记录只证明该机器、上述两个 no-traffic 场景和一次 32-transition update 上，新 reward、双能耗
流、GAE/PPO 与 artifact 链路能够真实运行且可审计。native energy 为零再次符合 E-019 的 phase
ordering 结论；reward 使用 execution-recomputed proxy。

本记录不比较 `metadrive_builtin_v1`，不证明 PlannerRFT/nuPlan scorer parity，不证明真实车辆
动力学或舒适性，也不支持任何节能改善、收敛、稳定性或跨 seed 结论。因此不关闭 Issue #59。
