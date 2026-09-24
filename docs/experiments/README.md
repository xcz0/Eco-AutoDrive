# 实验记录

本目录登记实际运行：用了什么代码、配置和随机条件，观察到什么，以及证据能支持到哪里。
每次运行的事实保存在 `records/` 与对应 artifact；当前研究结论见 [Findings](../research/findings.md)。
本 README 只拥有登记流程、最低 provenance、检索方式与模板。

## 哪些运行怎样登记

| 类型 | 登记与解释边界 |
| --- | --- |
| 正式实验 / 基线 | 从 clean commit 运行；冻结研究条件，完整登记结果与失败。完成正式运行也不自动证明节能、安全或 parity。 |
| 诊断 | 标明干预、控制变量、测量信号与归因范围；开发诊断允许 dirty 工作区，但不得据此登记正式结果。 |
| 本机验证 | 标明软件契约、机械链路或本机性能的验证范围；不升级为普遍科研结论。 |

只为真实运行建 record。已接受但未运行的工作由对应 GitHub Issue 跟踪；未接受的研究问题归[Hypotheses](../research/hypotheses/ablation-plan.md)。软件测试不代替研究实验。

## 最低 provenance 与登记流程

1. 记录 ID、日期、类型、目的和实际完成/失败/部分完成状态；文件名使用 `e-NNN-描述.md`。引用时使用完整路径，不能只靠可能重复的 E-ID。
2. 记录实际 Git commit、branch、dirty status，并链接本次使用的 resolved config、overrides、命令和运行 metadata。应能沿这些来源找到依赖/设备、上游与模型身份、场景/地图、实际 training/map/noise/action seeds、sampler 与 cadence；不要用当前默认配置代填历史条件。
3. 记录主要结果、失败/缺失情况、统计口径和结论边界，链接 supporting artifact 与相关 Issue。若结果改变当前结论，再更新 Findings；不在 README 追加历史摘要或完整索引。
4. 历史 evidence 原则上 append-only，保留当时真实保存的来源；没有记录的事实不得补造。历史命令、旧入口与旧锚点按当时上下文解释，不为链接清理重写 records。

这里规定研究者怎样登记；writer/reader 必须保存、校验什么以及 missing/partial/undefined 的含义归 [Artifacts contract](../contracts/artifacts.md)，不在模板另建 artifact 字段规范。
正式结果要求 clean commit，不新增自动 preflight、source copy、tracked diff 或补偿性 manifest。

## Artifact 保存与历史来源

运行目录由 Hydra 独立创建，不覆盖旧结果。大型 artifact、raw tensors、GIF 与原始日志保存于 Git 忽略的 `outputs/` 或外部存储；record 写明可定位路径，不把大型文件嵌入 Git 文档。
实验记录与 artifact 保留真实失败，不用缺失值或新运行替换旧证据。

若历史 record 依赖当时的共享资产说明，应按其历史 Git revision 查阅原 README。
Findings 只在实际引用该证据时给出所需的固定 revision 来源；不把历史资产提升为现行规范，也不另外维护共同资产档案或旧锚点兼容页。

## 检索

按文件名、研究问题、Issue 或关键字查找，再读取所需 record：

```powershell
rg --files docs/experiments/records -g '*issue98*'
rg -n '结论边界|cadence|critic' docs/experiments/records -g '*.md'
```

只需当前结论时先读 [Findings](../research/findings.md)，无需遍历全部历史。

## Record 模板

```markdown
# E-NNN 标题

**日期 / 类型 / 目的 / 状态**：
**代码与运行身份**：commit、branch、dirty status；runtime metadata 路径
**环境与模型**：依赖/设备、上游与 checkpoint 身份的实际来源
**研究条件**：场景/地图、实际 seeds、sampler/cadence；相关 Protocol / Issue
**配置与命令**：resolved config、Hydra overrides、实际命令
**结果**：主要指标、统计口径、失败/缺失/部分完成情况
**产物**：summary、trace、训练产物或外部归档的实际路径
**结论边界**：支持什么；不支持什么；与历史条件的差异
```
