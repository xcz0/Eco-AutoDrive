# E-027 Issue #74 rollout hot-path optimization（RTX 3050 Laptop）

**日期 / 类型 / 目的**：2026-08-28 / 本机优化验证 / 按局部 DiT 编译、同步清理、
DDIM allocation 的顺序验证 Issue #74 候选优化，并只保留契约和证据允许的改动。

## 代码与环境

- Git HEAD：`38f525148e5831f1580a8506479de8b6e6408242`；每个成功 benchmark artifact
  保存运行时 tracked diff，最终实现另包含本记录与 system contract 更新。
- Windows 10 build 26200；Python 3.10.20；PyTorch 2.12.1+cu126；Lightning 2.6.5；
  MetaDrive 0.4.3；NVIDIA GeForce RTX 3050 Laptop GPU（4,294,508,544 bytes），
  CUDA 12.6、BF16 mixed precision。
- 官方 checkpoint：`checkpoints/DP-Origin/model.pth` 与 `args.json`；DDIM-5、
  `ddim_stochasticity=0`、orthogonal policy guidance。

## 配置与顺序

B=2/B=4 均使用 vector collector、每 slot 16 transitions、4 PPO epochs、minibatch 16、
1 warmup update、3 measured updates、3 repeats；scenario/noise/policy-action seed bases 分别为
`0`、`0`、`10000`。所有成功 artifact 保存 resolved config、Hydra overrides、tracked diff、
planner phase、cold start 和峰值 CUDA allocated memory；benchmark provenance 另保存实际
`planner_compile_mode`。

```powershell
just benchmark rollout `
  "benchmark.batch_sizes=[2,4]" `
  "benchmark.collector_modes=[vector]" `
  "benchmark.transitions_per_slot=16" `
  "benchmark.ppo_epochs=4" `
  "benchmark.ppo_minibatch_size=16" `
  "benchmark.warmup_updates=1" `
  "benchmark.measured_updates=3" `
  "benchmark.repeats=3" `
  "training.planner_compile_mode=eager" `
  "hydra.run.dir=outputs/benchmarks/issue74-2026-08-28/optimization/eager"
```

编译 smoke 将 batch sizes 改为 `[2]`、warmup/measured/repeats 改为 `1/1/1`，并设置
`training.planner_compile_mode=dit_reduce_overhead`；其余输入与 seed 不变。最终 safe cleanup
使用与 eager baseline 相同的完整 B=2/B=4 矩阵。

## 阶段结果

### 1. eager baseline

下表为每个 batch 的 9 个 steady-state samples 中位数；括号内为三个 repeat 各自的
collection 中位数。

| B | Collection (s) | Repeat medians (s) | End-to-end (s) | transitions/s | Decision (s) | Cold E2E (s) | Peak bytes |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 2 | 4.712 | 4.638, 4.712, 4.939 | 5.042 | 6.79 | 4.398 | 13.401 | 102,895,104 |
| 4 | 5.363 | 5.268, 5.363, 5.382 | 6.023 | 11.93 | 4.908 | 16.864 | 110,843,904 |

prepare/reference 与 guided DDIM accelerator 中位数分别为 B=2 `1.504/2.668 s`、
B=4 `1.772/2.868 s`；production guidance action equality check 分别为
`0.0173/0.0167 s`。

### 2. compile-only

`dit_reduce_overhead` 按计划只编译 `decoder.dit` 的 bound `forward`，没有替换注册模块；
模式为 `reduce-overhead`、`fullgraph=True`、`dynamic=False`。本机在第一次 graph compilation
时明确失败：`torch._inductor.exc.TritonMissing: Cannot find a working triton installation`。
当前 lock 只在 `sys_platform == 'linux'` 安装 Triton，因此没有成功 compile graph、没有可报告的
recompile、graph break 或 steady-state 样本。运行未 fallback eager，也没有形成 compile-only
B=2/B=4 artifact。编译默认门槛未进入性能判断，PPO base profile 保持 `eager`。

### 3. sync/allocation cleanup

- 删除 production `torch.testing.assert_close`；planner action identity 改由单元测试覆盖。
- fixed/reference 与 learned-policy 的精确零 action fast path保持不变。曾尝试以 device-side
  tensor selection 去除 learned action 的 `count_nonzero(...).item()`，完整矩阵为 B=2
  `5.293 s`、B=4 `5.761 s`，未证明安全收益，已全部撤销。
- `ddim_stochasticity=0` 向 diffusers 传 `variance_noise=None`，不再创建
  `zeros_like(sample)`；单元测试证明输出与显式零 tensor 逐位相等，且 generator state 逐位不变。

最终 safe cleanup 的三次对照如下：

| B | Collection (s) | Repeat medians (s) | End-to-end (s) | transitions/s | Decision (s) | Cold E2E (s) | Peak bytes |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 2 | 5.246 | 5.054, 5.246, 5.440 | 5.643 | 6.10 | 4.901 | 15.149 | 102,895,104 |
| 4 | 5.690 | 5.732, 5.628, 5.647 | 6.391 | 11.25 | 5.223 | 17.851 | 110,843,904 |

该顺序测量不能把全局差值归因于两个微优化：未修改的 prepare/guided DDIM 同时比首轮慢
约 13%--18%。随后临时恢复旧 assert 与零 tensor 的 control 又得到 B=2 `3.754 s`、
B=4 `5.372 s`，其中 B=4 三个 repeat 中位数为 `4.243/7.998/5.372 s`，进一步证明本机
温度/功耗状态漂移大于候选微优化。最终只依据逐值/RNG 契约测试保留两项清理，不宣称
collection、end-to-end 或 throughput 获得改善；显存峰值也未变化。

## 验收结论与边界

- compile 模式已成为严格必填配置、CUDA-only 且无 fallback；但本机 compile smoke 失败，
  因而没有达到 B=2/B=4 repeat-level 与总中位数 `>=3%` 门槛，默认值保持 eager。
- learned zero-action `.item()` 保留；未实施 bootstrap、critic、TensorDict clone、constraint
  in-place 或 per-slot RNG 改写。
- eager 与最终 safe cleanup 都完成固定 seed 的三次 B=2/B=4 正式矩阵，无 OOM、非有限值或
  simulator failure。compile graph/recompile 稳态结论不可用，因为后端初始化前已失败。
- 本实验支持配置/失败语义和两项局部等价性；不支持在当前 Windows 机器上启用 compile，
  也不支持声称端到端性能提升。要改变默认值，仍需在具备可用 Triton 的 CUDA 环境重新执行
  相同门槛。

## 验证

- targeted CPU：`20 passed, 1 deselected`；新增严格 compile mode、zero guidance action
  identity/diagnostics、DDIM eta=0 输出与 RNG 测试均通过；配置严格拒绝未知 compile mode
  的单文件复跑为 `10 passed`。
- `just test-gpu`：`1 passed, 57 deselected`。compiled numerical/hash/repeat execution
  未运行，原因是同一环境的 compile smoke 在首次 backend 初始化即明确失败。
- `just test-sim`：`5 passed, 53 deselected`。
- slow 真实 checkpoint 两-transition PPO 节点：`1 passed`，policy 更新且 frozen planner hash
  不变。
- `just lint`：通过；`just typecheck`：0 errors、0 warnings。

## 产物

- eager：`outputs/benchmarks/issue74-2026-08-28/optimization/eager/`
- compile smoke：`outputs/benchmarks/issue74-2026-08-28/optimization/compile-smoke/`
- rejected learned zero-action variant：
  `outputs/benchmarks/issue74-2026-08-28/optimization/final-eager/`
- final safe cleanup：`outputs/benchmarks/issue74-2026-08-28/optimization/final-safe/`
- thermal control：`outputs/benchmarks/issue74-2026-08-28/optimization/control-hot/`
