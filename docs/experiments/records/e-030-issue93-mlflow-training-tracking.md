# E-030 Issue #93：PPO / MLflow 本机验收

**日期 / 类型 / 目的**：2026-09-08；本机实现验收；验证真实 PPO update 的 MLflow 指标、
artifacts、不同 seed 的独立 Run，以及 checkpoint 恢复时继续同一 Run。

## 代码与环境

- Git HEAD：`8ba897fe7a9a0fef2b2a4ea59e13bd8453f57ac3`，分支 `scalar-energy-reward`，
  加本次 Issue #93 未提交实现。每份 runtime metadata 保存当时的 Git status，
  `tracked_diff.patch` 保存对应 tracked diff；新增未跟踪文件列在 Git status 中。
- Windows，Python 3.10.20，CUDA 设备 `cuda:0`，`bf16-mixed`；资源配置
  `rtx3050_laptop`，rollout worker count 4、每 worker 8 threads。
- MLflow 3.16.0、TorchMetrics 1.9.0、Lightning 2.6.5、Torch 2.12.1+cu126、
  TorchRL 0.13.3、TensorDict 0.13.0。新增依赖已安装，本地 `uv.lock` 已更新且仍被 Git 忽略。
- 使用现有 `checkpoints/DP-Origin`，EMA 276 tensors、6,042,628 parameters。
  `jobs/training/ppo_energy_smoke`、DDIM5、`plannerrft_energy_v1`、no-traffic、
  S/SC map seed 0，每个 scenario 16 transitions，每 update 共 32 transitions。
  保持原 PPO 参数，包括 4 epochs、minibatch size 16、scheduler horizon 32 optimizer steps。

## 运行命令与产物

以下命令在仓库根目录运行，均使用 `training.replay_id=0`。成功运行的标准输出与错误输出
分别保存到 `outputs/issue93-seed0.log`、`outputs/issue93-seed1.log`、
`outputs/issue93-resume-fixed.log`。

```powershell
just training run --config-name jobs/training/ppo_energy_smoke components/resources=rtx3050_laptop runtime.seed=0 training.replay_id=0 tracking.run_name=issue93-seed-0 +tracking.tags.study=issue93-acceptance hydra.run.dir=outputs/training/issue93-mlflow/seed-0

just training run --config-name jobs/training/ppo_energy_smoke components/resources=rtx3050_laptop runtime.seed=1 training.replay_id=0 tracking.run_name=issue93-seed-1 +tracking.tags.study=issue93-acceptance hydra.run.dir=outputs/training/issue93-mlflow/seed-1

just training run --config-name jobs/training/ppo_energy_smoke components/resources=rtx3050_laptop runtime.seed=0 training.replay_id=0 training.update_count=2 training.resume_checkpoint_path=outputs/training/issue93-mlflow/seed-0/training-state.ckpt tracking.run_name=issue93-seed-0 +tracking.tags.study=issue93-acceptance hydra.run.dir=outputs/training/issue93-mlflow/seed-0-resumed-fixed
```

每个输出目录包含 resolved config、Hydra overrides、runtime metadata、tracked diff、
严格 summary、update NPZ 和 checkpoints。MLflow 数据库为 `outputs/mlflow/mlflow.db`，
artifact root 为 `outputs/mlflow/artifacts`，experiment `eco-autodrive-ppo`（ID `1`）。

| 运行 | training seed | Run ID | 本地产物中的 update indices | 累计 transitions | 状态 |
| --- | --- | --- | --- | --- | --- |
| seed-0 | 0 | `ca2aafeb3bef4810b191c62bde9c345d` | 0 | 32 | FINISHED |
| seed-1 | 1 | `24cd409f40fa45a6b95bbb788053a674` | 0 | 32 | FINISHED |
| seed-0-resumed-fixed | 0 | `ca2aafeb3bef4810b191c62bde9c345d` | 0, 1 | 64 | FINISHED |

seed 0 的 diffusion noise seeds 为 `2140815137, 4177475342`，policy action seeds 为
`2920704928, 3069229606`；seed 1 分别为 `468171111, 1065024745` 和
`1394698924, 67319374`。恢复运行使用原 seed 0 命名空间。

三个成功调用的 MLflow invocation artifact 子目录依次为：

- seed-0：`invocations/b437c80388c74579b3f5f6f846ab293c`
- seed-1：`invocations/6d4291deb2c340289edb84c8e93e9072`
- 恢复：`invocations/01627ab253cc45c38587aa5724242463`

两个初始调用都关联 initial/update-000/final policy、training-state、resolved config、
runtime metadata、tracked diff 和 summary；恢复调用独立关联 update-001/final policy、
training-state 及自己的配置、metadata、diff 和 summary。

## 结果与失败记录

| 最后一个 update | mean total loss | total reward | mean speed (m/s) | executed proxy (mL/km) | collision / out-of-road transitions |
| --- | --- | --- | --- | --- | --- |
| seed-0 / update 0 | 16.4735643864 | 27.0800971985 | 10.5745515823 | 47.5677044535 | 0 / 0 |
| seed-1 / update 0 | 16.1989252567 | 27.0738964081 | 10.7923450470 | 48.0011600221 | 0 / 0 |
| seed-0 / update 1（恢复） | 14.2876173258 | 27.0800380707 | 10.5769071579 | 47.5717203461 | 0 / 0 |

三次成功调用的 policy 均发生参数更新，冻结 planner 的前后 hash 均相同。
逐项读取 MLflow metric history，与对应持久化 summary 经 adapter 得到的每个训练指标比较：
数值精确一致，每个 metric/step 恰好一个点。seed 0 Run 最终包含 steps `[0, 1]`，
seed 1 Run 仅有 step `[0]`；恢复没有新建 Run 或重复记录 step 0。

首次恢复曾失败，输出目录为 `outputs/training/issue93-mlflow/seed-0-resumed`，
日志为 `outputs/issue93-resume.log`。原因是原训练 checkpoint 使用 JSON-mode dump 保存
summary/probe，tuple 因而成为 list；原恢复逻辑调用 strict Python validation，拒绝这些
list。修复为使用严格 JSON validation 读取同一持久化表示，不改变 checkpoint 格式或 PPO
数学。故障发生在 MLflow attach 和新 rollout 之前，原 Run 未被修改。增加真实 Fabric
保存/恢复回归测试后，使用同一 checkpoint 重跑成功；保留首次失败记录。

## 验证与结论边界

针对本次变更，59 个不同的 CPU 测试用例通过，另有 1 个真实 MetaDrive/CUDA 训练测试通过。
相关入口包括 `tests/training/test_tracking.py`（15 项）、`test_ppo.py`（7 项），以及
configuration 下的 jobs（13）、workflows（7）、scalar_reward（10）、experiments（7）。
运行入口为 `just test-target`；simulator 验证命令为：

```powershell
just test-target tests/simulation/test_training_tracking.py -m simulator
```

额外完成 `just lint`、`just typecheck`。测试覆盖真实 SQLite 参数和 artifacts、独立 Run、
状态转换、缺失指标补录、冲突/陈旧 checkpoint 拒绝、旧 checkpoint provenance、关闭跟踪、
非等长 episode 分母、零距离和 PPO / RNG / checkpoint 数值回归。

本记录支持训练跟踪、指标映射和同 Run 恢复链路的正确性；不构成 A1/A2 reward 对照、
held-out evaluation 或节能结论。恢复沿用现有重新建立 collector 和随机 generator 的行为，
没有扩展到恢复 MetaDrive 中间状态，也不声称与不中断的仿真逐位等价。
