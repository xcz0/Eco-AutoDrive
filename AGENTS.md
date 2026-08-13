# Repository Guidelines

## 上下文与事实来源

按任务最小读取资料；需要更新事实时写回对应权威位置，不要无差别加载或在多处重复维护：

- 领域概念与术语：`CONTEXT.md`
- 数据流、执行逻辑、训练/推理/评测语义和系统不变量：`docs/agents/system-contract.md`
- 长期设计理由与变更：相关 `docs/adr/`；若方案与现有 ADR 冲突，显式指出
- active work：按 `docs/agents/issue-tracker.md` 使用 GitHub Issues
- 未定研究方向：`docs/research/README.md`
- 实验修改、运行或解释：`docs/experiments/README.md`、对应实验记录及实际 config/artifact
- 当前入口、依赖与实现行为：`README.md`、`pyproject.toml`、`uv.lock`、相关代码和测试

事实优先级：**已实现代码和测试 > 机器可读配置与依赖文件 > authoritative docs > 其他说明**。发现冲突时明确指出；若本次修改使对应实现或 authoritative docs 失效，同步更新受影响内容。

## 工作原则

本项目是个人科研代码库，以逻辑正确性优先，不按生产系统的防御性工程标准设计。

- 显式定义必需输入、配置、依赖、张量形状、单位、时间频率和坐标基准；缺少必需条件时立即失败并给出可定位错误。
- 不添加静默默认、自动降级、宽泛异常捕获或坏样本跳过。
- 不用平滑、裁剪、中心线投影、回退控制器或零轨迹掩盖模型失效。
- 优先使用成熟第三方库；没有明确复用需求时，不建立通用框架。
- 未完成的 issue 或 research 设想不得描述为可用，也不得用假实现填充占位入口。

除非任务明确要求，或已有失败证据表明必要，否则不要新增：

- 文件、checkpoint、artifact 或依赖的 hash/checksum；
- baseline/reference/A-B/重复实验作为常规验证；
- 额外环境、硬件、依赖或文件完整性 preflight；
- manifest、版本握手、兼容/迁移层、冗余状态记录或仅针对理论故障的防御性检查。

验证只确认本次修改是否正确。若额外工作只是提高鲁棒性、完整性、可追溯性或复现保证，而非当前任务正确性所必需，则不实现、不执行。

## 代码与数据边界

- `configs/` 保存环境、模型、训练和实验配置；实验参数不得硬编码进 Python。
- `third_party/metadrive/` 是运行时本地 editable 源码；`ref/` 是只读上游快照，业务代码不得导入。
- `checkpoints/`、`outputs/`、`.venv/`、`.env`、`ref/` 及被忽略的上游源码不得提交。

## 实现约束

- 公共 API 必须有类型标注；只为非显然逻辑写简短 docstring。
- 使用 Python 3.10；遵循 Ruff 100 字符行宽及 `E/F/I/UP` 规则。
- 文件路径使用 `pathlib.Path`，不得硬编码平台绝对路径。
- Hydra 必需字段必须显式提供，不得用随意默认值隐藏缺项。
- 在系统边界验证维度、dtype、设备、单位、范围和有限性。
- 随机性由显式 seed 控制；实验记录保存实际 seed。
- 依赖变更同时更新 `pyproject.toml` 和 `uv.lock`。

## 验证

采用覆盖本次修改的最小验证集合：

- 文档修改：不运行代码测试。
- 局部实现修改：优先运行直接相关测试。
- bug 修复：必须包含并运行能复现问题的回归测试。
- 跨模块或公共行为修改：补充必要的快速测试。
- MetaDrive 交互、坐标转换、轨迹执行或仿真语义修改：运行直接覆盖该行为的 simulator 测试。
- 实验方法或指标修改：验证对应实验路径；仅当结论依赖 baseline 对照时运行 baseline。
- 依赖修改：检查 `pyproject.toml` 和 `uv.lock`。

依赖硬件或仿真器的测试应显式标记；缺少依赖时由命令排除 marker，测试本身不得静默跳过。

编码智能体运行在已准备好的 Windows 沙箱中。假定 `.venv` 可用；不要把 `uv sync`、Python 版本探测或其他环境预检作为常规步骤，也不要使用 `uv run`。使用 `.venv\Scripts\` 下的工具：

```powershell
.venv\Scripts\pytest.exe -m "not gpu and not simulator and not slow"
.venv\Scripts\ruff.exe check .
.venv\Scripts\ruff.exe format --check .
```

针对性验证通过且没有其他失败证据时，停止验证。

## Git 与实验记录

提交信息使用简短、祈使式主题；一次提交只完成一个逻辑变更。

涉及轨迹、地图、能耗或评测行为的实验，在 `docs/experiments/` 记录代码状态、未提交差异、上游版本、数据或地图、resolved config、Hydra overrides、随机种子、验证命令、结果和产物路径，并更新实验索引。大型产物和原始日志保留在 `outputs/` 或外部存储；缺失元数据标记为“未记录”，不得补猜。
