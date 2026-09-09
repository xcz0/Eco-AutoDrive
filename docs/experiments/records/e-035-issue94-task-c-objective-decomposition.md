# E-035 Issue #94 Task C：Objective decomposition 与 normalization attribution

[返回实验索引](../README.md) · [源实验 E-033](e-033-issue94-task-a-lambda-identifiability.md) ·
[源实验 E-034](e-034-issue94-task-b-reward-calibration.md)

**日期 / 类型 / 目的**：2026-09-09 / 正式离线固定批次诊断 / 完成 Issue #94 Task C：在固定
update-0 batch 上比较 R0-only 与 Energy-only 两个 objective endpoint 的 actor 优化方向，
归因 advantage normalization 对差异的压缩，并检查 `λ={16,64,256}` 的 gradient-space stress
轨迹。本实验只执行 actor backward，不执行 optimizer step，不选择训练 reward。

**代码 / 环境**：基线 `1a89b3c` 加本次未提交改动；`runtime_metadata.json` 和
`tracked_diff.patch` 记录运行时仓库状态，`source/` 保存实际执行的模块与 CLI。Windows、
Python 3.10.20、PyTorch 2.12.1+cu126、CUDA:0；actor backward 为 float32，无 rollout
autocast。

**批次来源**：本机已无 E-033/E-034 原始产物目录，因此按 E-033 的同一协议重新采集一批
update-0 作为源 batch（`e-035-issue94-task-c-source-batch/`）：R0 arm、training seed 0、
replay 0、S/SC × map seeds 0–7、每场景 8 transitions，共 16 episodes / 128 transitions。
源 batch 的 initial policy hash `04969773…` 与 frozen planner hash `6a014ec5…` 与 E-033
记录一致；component 统计与 λ0→λ16 Task A 诊断在 `1e-4 ~ 1e-7` 内复现 E-033 记录值
（非逐位一致，源自 BF16 AMP rollout 的浮点噪声经差分量放大：原始运动学中位数漂移
≤0.5%）。因此本记录不声称逐位复用 E-033 batch，只声称同协议、同 seed、同 hash 的统计
复现。

**校准**：Task B 的校准规则（目标分数 0.6/0.6）在本 batch 上重新执行，实际校准值为
`full_score_delta_m=1.7807237`、`longitudinal=4.1442301`、`jerk=112.0860999`
（E-034 冻结值 `1.7813476/4.1575485/111.7048700`，相对差 ≤0.4%）；lateral/yaw 保持
3.0/0.5。配置中的 `expected_calibration` 以 `rtol=0.05` 作为源 batch 溯源守卫，防止误用
其他 batch 或协议漂移；校准本身由冻结规则在本 batch 数据上确定。校准后本 batch 动态
范围：Progress `0.598376 ± 0.033143`、Comfort `0.290963 ± 0.392637`、Energy
`0.385149 ± 0.007764`，与 E-034 报告一致。

**配置 / 命令**：

```powershell
just lambda-identifiability `
  --output-dir outputs/studies/scalar-reward/e-035-issue94-task-c-source-batch
just objective-decomposition `
  --source-dir outputs/studies/scalar-reward/e-035-issue94-task-c-source-batch `
  --output-dir outputs/studies/scalar-reward/e-035-issue94-task-c-objective-decomposition
```

`configs/experiments/scalar_reward/objective_decomposition.yaml` 显式指定 stress λ、量化
分位、校准目标分数、E-034 冻结值、溯源容差与 Gate C 阈值。

**计算语义**：arms 为校准后 R0（共享权重 5/5/2/4，分母 16）、`λ={16,64,256}`（分母
16+λ）与 Energy-only endpoint（`reward = safety_gate × reward_component_energy`，仅作
objective endpoint，不是训练 profile）。全部 arm 复用同一批 component scores、safety
gate、policy context、动作、old log-prob、critic current/next value 和 episode boundary。
每个 arm 在三种纯诊断 advantage 形式下各做一次 full-batch `loss_objective` backward：

```text
raw          GAE 输出原样
center-only  减去 full-batch 均值
z            当前训练路径的 full-batch sample-std 标准化
```

不包含 critic/entropy loss，不做 clipping、optimizer 或 scheduler step；backward 后
policy hash 不变，optimizer steps 为 0。

## C1 Endpoint：R0 vs Energy-only

| Arm | Reward mean ± std | Raw advantage mean ± std | Head grad norm raw/center/z |
| --- | ---: | ---: | --- |
| R0（校准） | 0.785863 ± 0.050993 | 3.108009 ± 1.437447 | 0.18313 / 0.13939 / 0.09697 |
| λ=16 | 0.585506 ± 0.024693 | 2.312149 ± 1.066794 | 0.13643 / 0.10037 / 0.09409 |
| λ=64 | 0.465291 ± 0.010351 | 1.834633 ± 0.844852 | 0.10848 / 0.07697 / 0.09110 |
| λ=256 | 0.408720 ± 0.007080 | 1.609919 ± 0.740635 | 0.09536 / 0.06596 / 0.08905 |
| Energy-only | 0.385149 ± 0.007764 | 1.516289 ± 0.697276 | 0.08991 / 0.06137 / 0.08801 |

| Endpoint 形式 | Pearson | Spearman | Sign flip | Advantage RMSE | Head cosine | Head norm ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| z（当前预处理） | 0.997583 | 0.980109 | 0 | 0.069255 | 0.999713 | 0.907619 |
| raw | 0.997583 | 0.980109 | 0 | 1.755549 | 0.997131 | 0.490962 |
| center-only | 0.997583 | 0.980109 | 0 | 0.740527 | 0.999713 | 0.440267 |

lateral/longitudinal head cosine（z 形式）分别为 0.999639/0.999754，与整 head 一致。
shared-trunk 梯度仍因 actor-head 零初始化严格为 0，cosine/norm ratio 记 `null`，不作为
证据。

## C2 Normalization attribution

数学恒等式在产物中经验证实：Pearson/Spearman 在三种形式下逐位一致（差异 <3e-7），
center-only 与 z 的 head cosine 相等（`0.9997127264` vs `0.9997127272`，差 9e-10），
sign 结构相同（sign flip 均为 0）。raw 形式的 head cosine（0.997131）低于 z 形式
（0.999713）：均值中心化确实进一步压缩了本已很小的方向差，但 **raw 形式本身也未达到
0.99 的可分阈值**。raw RMSE 1.755549 的主要成分是 reward 均值差（3.108 vs 1.516），
centering 后降为 0.740527，标准化后为 0.069255。

## C3 λ stress 轨迹

| λ | Head cosine vs R0 | Angular separation (rad) | 占 endpoint 分离比例 | Head norm ratio |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 0.999973 | 0.007327 | 30.57% | 0.970282 |
| 64 | 0.999883 | 0.015289 | 63.78% | 0.939485 |
| 256 | 0.999780 | 0.020991 | 87.57% | 0.918359 |

λ 增大时 angular separation 单调增加，λ=64/256 分别达到 endpoint 分离的 63.78%/87.57%
（≥50%），方向与“reward 线性插值 → gradient 空间正组合”的解释一致。

## Gate C 裁定

```text
Verdict: FAILED
Attribution: objective_batch_collinearity
```

失败项：endpoint actor-head cosine `0.999713 > 0.99`（raw 形式 `0.997131` 同样 > 0.99）；
normalized-advantage RMSE `0.069255 < 0.10` 且 sign-flip fraction `0 < 0.05`。按 Task C
归因规则，三种形式下 R0 与 Energy-only 都近乎共线，标记为 `objective/batch
collinearity`，而不是 `normalization-suppressed identifiability`。

**解释**：当前 batch 上 R0 与 Energy-only 的 actor 优化方向差异不是被
`GAE → full-batch z-normalization` 消掉的——raw 形式已经共线。共线的主要来源是全部
arm 共享的 critic value/GAE 结构（相同 V/V'、episode boundary），其贡献的 raw advantage
尺度（std 0.70–1.44）远大于 reward 组合差异本身（R0 的 reward std 0.051 主要来自校准后
Comfort，Energy-only 仅 0.0078）。λ stress 轨迹单调且有限 λ 能达到 endpoint（微小）分离
的 ≥50%，说明 relative-scale/λ parameterization 不是当前的主要瓶颈；瓶颈是 endpoint
本身在该 batch 上不可辨识。

## 与 E-034 校准组 0→16 的交叉核验

本实验 R0→λ16（校准、z 形式）：Pearson `0.999742`、Spearman `0.999193`、head cosine
`0.999973`、norm ratio `0.970282`、RMSE `0.022621`；E-034 记录值分别为 `0.999739`、
`0.999239`、`0.999973`、`0.970824`、约 `0.022769`。lateral/longitudinal cosine
`0.999965/0.999977` 亦与 E-034 一致。整条计算链在 batch 重采噪声内复现 E-034。

**结论边界**：以上结论只支持该 initial policy、该 no-traffic batch 与该校准配置；不证明
optimizer 问题，不证明训练后行为分离，不选择训练 reward 或 λ。Energy-only 仅是
objective endpoint。按 Issue #94 顺序，下一步是 Task D（guidance control-authority
intervention）；Task E 的进入条件之一（Gate C failed）在本 batch 上已成立，但是否执行
仍取决于 Gate D。

## 验证与产物

新增测试 `tests/training/test_objective_decomposition.py` 7/7 通过，覆盖 Energy-only
重组、三种 advantage 形式关系（center=raw−mean、z=center/σ、符号结构）、梯度线性恒等式
（`g_z = g_center/σ`、λ 插值下 `g_λ(16+λ)/σ_λ = 16·σ_R0·g_R0 + λ·σ_E·g_E` 的逐元素
核验）、Gate C 标签/裁定、E-034 冻结值守卫与配置校验。直接相关既有测试
（Task A/B、PPO、reward）49/49 通过；Ruff 通过；新增/修改模块 Pyright 0 errors。

正式产物只读核验：138 个数组全部有限、128 样本、每 arm normalized advantage sample
std 在 `1e-6` 内等于 1、center/z 符号结构相同、policy hash 前后一致、optimizer steps
为 0。源 batch 产物（`e-035-issue94-task-c-source-batch/`）为完整 Task A 结构，其诊断
兼作 E-033 复现证据。

产物目录：`outputs/studies/scalar-reward/e-035-issue94-task-c-objective-decomposition/`
（git ignored）：

- `report.md` / `summary.json`：arm/pair/gate 全量统计与裁定。
- `diagnostics.npz` / `sample_index.json`：逐 transition reward/advantage（三种形式）、
  逐 form 逐参数组梯度向量与配对差异。
- `calibration_verification.json`：本 batch 校准值与 E-034 冻结值对照。
- `resolved_config.yaml` / `diagnostic_config.yaml` / `runtime_metadata.json` /
  `tracked_diff.patch` / `source/`：运行来源与实际执行的源码快照。

Task C 的实现、离线诊断与 Gate C 裁定已完成；未执行 Task D–H，未修改全局 reward 默认
配置，未关闭 GitHub Issue #94。
