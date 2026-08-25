# E-023 TorchRL ParallelEnv migration correctness check

**日期 / 类型 / 目的**：2026-08-25 / 本机迁移验证 / 在删除自定义 MetaDrive IPC 前，对真实
MetaDrive 的 legacy 与 TorchRL vector backend 做一次行为差分。用户明确将本次验收收敛为单次
正确性运行，不要求正式性能对照。

**状态**：正确性门槛通过；没有形成性能结论。

## 代码、环境与配置

- 基础 commit：`39f1823`，运行包含本次未提交 TorchRL façade、evaluation/rollout 迁移和差分测试。
- 机器：Windows 10 build 26200；AMD64 Family 25 Model 80（16 logical cores）；NVIDIA GeForce
  RTX 3050 Laptop GPU；Python 3.10.20；TorchRL 0.13.3；TensorDict 0.13.0；MetaDrive 0.4.3。
- 场景：2 个 no-traffic 物理 slots；`S` seed 0 full reset/step，随后 slot 0 换为 `SC` seed 1 并做
  partial reset/step；trajectory 为有限 `float32 [80,4]` stationary trajectory。
- 差分：slot/scenario 顺序和 discrete termination 字段精确比较；observation、reward、initial
  state 与 execution 连续值使用 `rtol=1e-6, atol=1e-6`。

## 命令与结果

```powershell
just test-target tests/integration/test_torchrl_parallel_env_poc.py::test_torchrl_backend_matches_legacy_full_and_partial_results
```

结果为 `1 passed in 17.50s`。full 与 partial 操作的结果顺序、scenario identity、observation、reward、
termination、initial state 和 execution state 全部通过。随后删除 legacy implementation、迁移期
adapter/config 与该差分测试；最终生产回归由当前 `tests/integration/test_torchrl_parallel_env.py`
和 `tests/integration/test_vector_metadrive_env.py` 维护。

删除完成后，以一次命令运行 façade、真实 MetaDrive、evaluation refill、rollout wave/RNG/timing
相关集合，结果为 `25 passed in 65.34s`；Ruff 对 `src scripts tests` 检查通过。

## 结论边界

本记录支持在该机器、固定依赖和上述两个 no-traffic slots 上切换到 TorchRL `ParallelEnv`，并支持
删除项目自定义 IPC。它不比较吞吐或墙钟，不声称满足原计划的 `1.10x` 性能门槛，也不把结论外推
到其他硬件、worker 数或 traffic 长程矩阵。迁移期间曾运行的短性能 smoke 不构成本记录的结论。
