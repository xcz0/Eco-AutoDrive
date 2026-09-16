# E-033 Issue #94 Task A：λ-gradient identifiability gate

[返回实验索引](../README.md)

**日期 / 类型 / 目的**：2026-09-09 / 正式固定批次诊断 / 完成 Issue #94 Task A。
在同一 initial Exploration Policy、同一 update-0 rollout、同一 GAE/PPO actor objective
下离线比较 `λ ∈ {0,1,2,4,8,16}`。本实验只执行 actor backward，不执行 optimizer
step，不选择 Task B/C 分支，也不证明 closed-loop learned behavioral effect。

**代码**：基线 `1257c399625ac02e28b2445f6f291616c4d1a891`；运行时
`git_status_short` 记录本 Task 的未提交文件，`tracked_diff.patch` 捕获当时已跟踪的
`justfile` 修改。由于新增文件不进入 `git diff`，产物 `source/` 额外保存实际执行的
实验模块与 CLI，`diagnostic_config.yaml` 保存实际诊断配置；后续实验记录与 system
contract 文档不冒充运行时已存在的 provenance。

**批次来源与匹配协议**：本机没有 E-030/E-031/E-032 的原始 rollout 产物，因此按本
Task 执行前确定的方案重新采集一批 update-0，而不是声称复用 E-032 历史批次。采集使用
scalar-reward protocol 的 R0 arm、training seed 0、replay 0、S/SC maps × map seeds
0–7、每场景 8 transitions，共 16 episodes / 128 transitions。DDIM5、
`ddim_stochasticity=0`、no-traffic、10 Hz rollout、4 个 vector workers。全部 λ 从这
一批固定的 component scores、safety gate、policy context、guidance action、old
log-prob、current/next value 和 episode boundary 重算；不再访问 simulator。

**环境 / 模型**：Windows 10；Python 3.10.20；PyTorch 2.12.1+cu126；Lightning
2.6.5；MetaDrive 0.4.3；NVIDIA GeForce RTX 3050 Laptop GPU，CUDA BF16 mixed
precision。官方 planner checkpoint 为 `checkpoints/DP-Origin/model.pth`，EMA 276
tensors / 6,042,628 parameters；frozen planner hash `6a014ec5…`。initial policy
hash `04969773…` 与 E-031/E-032 记录中的 seed-0 initial policy 一致。

**计算语义**：`λ=0` 使用 `plannerrft_no_energy_v1`；正 λ 使用
`plannerrft_energy_v1`，共享权重为 TTC/Progress/Comfort/Speed = 5/5/2/4，energy
权重为 λ，分母为 `16+λ`。每个 episode 复用训练的 TorchRL GAE，再在完整 128 样本
batch 上使用 sample std 标准化一次 advantage。随后用同一 initial policy、同一
`ClipPPOLoss` 和同一 batch action/old log-prob 只反传 `loss_objective`。不包含
critic/entropy loss，不做 gradient clipping，不调用 optimizer 或 scheduler。

**命令**：

```powershell
just lambda-identifiability `
  --output-dir outputs/studies/scalar-reward/e-033-issue94-task-a-identifiability
```

正式运行约 32.9 s，状态 `completed`。产物核验确认 16 个 episode NPZ、128 个 sample
index 与 training TensorDict 严格对应，全部数组有限，episode 内 next-value linking
正确，所有 normalized advantage 的 sample std 在 `1e-6` 内等于 1。

## Reward component dynamic range

| Component | Mean | Std | p0 / p50 / p95 / p100 | 饱和事实 |
| --- | ---: | ---: | --- | --- |
| TTC | 1.000000 | 0 | 1 / 1 / 1 / 1 | 100% 为 1 |
| Progress | 0.995046 | 0.017057 | 0.872080 / 1 / 1 / 1 | 86.72% 为 1 |
| Comfort | 0 | 0 | 0 / 0 / 0 / 0 | 100% 为 0 |
| Speed | 1.000000 | 0 | 1 / 1 / 1 / 1 | 100% 为 1 |
| Energy | 0.385150 | 0.007761 | 0.361656 / 0.384775 / 0.399078 / 0.410754 | 有限但窄的 transition variance |
| Safety gate | 1.000000 | 0 | 1 / 1 / 1 / 1 | 100% 为 1 |

除 Energy 与少量 Progress 外，当前 batch 的 reward components 没有 transition 动态
范围。R0 reward mean/std 为 `0.873452 / 0.005330`；λ16 为
`0.629301 / 0.003013`。raw advantage 因 reward scale/offset 明显变化：均值从
`3.431012` 降至 `2.473651`，sample std 从 `1.585904` 降至 `1.141464`。

## Normalized advantage 与 actor gradient

| 对比 | Pearson | Spearman | sign flip | actor-head cosine | head norm ratio | lateral cosine / ratio | longitudinal cosine / ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| λ0 → λ8 | 0.999983 | 0.987117 | 0 | 0.999991 | 1.004171 | 0.999996 / 0.993754 | 1.000000 / 1.005740 |
| λ0 → λ16 | 0.999952 | 0.978268 | 0 | 0.999975 | 1.007059 | 0.999987 / 0.989401 | 1.000000 / 1.009708 |

λ0→λ16 normalized-advantage matched delta 的 mean 约 `0`、std `0.009820`、RMSE
`0.009782`、最大绝对值 `0.020767`；16 个 scenario 的 mean delta 范围为
`[-0.015796, 0.011922]`。Spearman 的下降说明 Energy 的有限 variance 确实改变了部分
样本秩次，但 Pearson、零 sign flip 和 actor-gradient 方向表明这种改变没有形成明显的
actor update 方向分离。完整 15 个 λ pair、各 λ reward/advantage quantiles、逐 scenario
和逐 transition matched differences 见结构化产物。

actor-head norm 从 λ0 的 `0.0860502` 增至 λ16 的 `0.0866577`。lateral head norm
从 `0.0312129` 降至 `0.0308821`，longitudinal head norm 从 `0.0801898` 增至
`0.0809682`。所有 λ 的 shared-trunk actor gradient norm 均严格为 0，因为 initial
actor-head weight 为零；因此 trunk cosine 和 norm ratio 记为 `null`，不能当作方向相同。

**连续诊断结论**：按执行前约定，本记录不建立额外数值阈值，也不自动决定 Task B/C。
证据支持以下精确表述：在该 initial policy 与新采集的 128-transition no-traffic batch
上，λ 明显改变 reward 与 raw advantage 的尺度，Energy 的有限 variance 也改变部分
normalized-advantage 秩次；但 λ0→λ16 没有 advantage sign flip，normalized advantage
Pearson 为 0.999952，actor-head gradient cosine 为 0.999975，norm 仅变化 0.706%。
因此当前 reward/batch 下可测差异很小，整体与 Issue #94 描述的“不易辨识”模式一致。
是否据此进入 Task B 仍由后续 gate 裁定。

**结论边界**：本实验只支持该新 batch、seed-0 initial policy 与当前 no-traffic reward
calibration；它不复现 E-032 的逐值 rollout，不证明 optimizer 问题，不证明训练后的行为
分离，也不选择最终 λ。TTC/Comfort/Speed/Safety 的完全饱和以及 Progress 的高饱和是
Task B dynamic-range audit 的直接输入；Energy 本身并非常数，因此不支持优先修改 Energy
normalization。

**产物**：`outputs/studies/scalar-reward/e-033-issue94-task-a-identifiability/`（git
忽略）。核心文件为 `report.md`、`summary.json`、`diagnostics.npz`、
`sample_index.json`、`training-batch.pt`、`policy-initial.pt`、`resolved_config.yaml`、
`diagnostic_config.yaml`、`runtime_metadata.json`、`tracked_diff.patch`、`source/` 与
`updates/update-000/` 的 16 个 episode audit NPZ。

**验证**：`just test-target tests/training/test_lambda_identifiability.py
tests/training/test_ppo.py tests/training/test_reward.py`：36 项中首轮 35 项通过，唯一失败
是测试内含 ties 的 Spearman 期望值误写；修正为实际平均秩结果 `8/9` 后新增诊断测试
15/15 通过。PPO/reward 既有直接相关测试 21/21 通过。新增模块 Pyright 为 0 errors，
Ruff 全部通过；正式产物另做 128 样本、有限性、dtype、next-value、normalized std、
零 trunk gradient、policy/planner hash 与 optimizer step 的只读一致性核验。
