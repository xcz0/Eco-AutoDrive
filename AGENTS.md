# Repository Guidelines

## 项目原则

Eco-AutoDrive 是个人科研代码库。优先保证**逻辑正确、实验语义明确、结论边界可追溯**；不要把生产系统的防御性工程规范默认带入本项目。

实现当前任务所需的最小正确改动。若额外工作只提高鲁棒性、兼容性、完整性或可追溯性，而不是当前任务正确性所必需，则不实现、不验证。

## 先确定事实来源

不要无差别加载文档。先按任务定位权威来源，只读取解决当前问题所需的最小集合：

| 需要确认的内容 | 首选来源 |
| --- | --- |
| 领域术语、容易混淆的概念 | `docs/agents/domain.md`；需要精确定义时再读 `CONTEXT.md` |
| 当前已实现的数据流、执行逻辑、shape、单位、时间语义、训练/评测契约 | `docs/agents/system-contract.md` 的相关章节 + 对应代码和测试 |
| 当前任务、待完成工作、验收标准 | 按 `docs/agents/issue-tracker.md` 读取对应 GitHub Issue |
| 长期设计选择及理由 | 相关 `docs/adr/` |
| 尚未确定的方法、假设和研究问题 | `docs/research/` |
| 已实际运行的实验、配置、结果和产物来源 | `docs/experiments/README.md`、对应 `records/`、实际 resolved config/artifact |
| 使用入口、依赖和当前运行方式 | `justfile`、`README.md`、`pyproject.toml`、`uv.lock`、相关代码和测试 |

区分“当前事实”和“目标要求”：

- 当前实现事实：**代码和测试 > 机器可读配置与依赖文件 > authoritative docs > 其他说明**。
- 目标行为：当前用户请求和对应 Issue 的明确验收标准决定本次工作范围。
- 已接受设计理由：以未被后续 ADR 取代的相关 ADR 为准。

发现冲突时不要自行融合成第三种解释。明确指出冲突属于当前实现、目标要求还是历史/研究描述；若本次修改使 authoritative docs 失效，同步更新对应权威位置。

## 行动与权限边界

- 回答、解释、评审或规划任务：检查相关材料并给出结论，不实现未请求的修改。
- 修改、修复或实现任务：直接完成范围内的本地修改，并运行必要的非破坏性验证；不为普通读取、编辑或测试反复请求确认。
- GitHub 写入、外部发布、破坏性操作、购买或明显扩大任务范围：只有当前任务或明确工作流要求时才执行。
- 缺少真正必需的事实、文件或配置时，在最小阻塞点停止并说明缺失项；不要用默认值或猜测继续。

完成当前请求且针对性验证通过后停止，不继续寻找理论上的改进机会。

## 科研实现原则

- 显式定义必需输入、配置、依赖、张量形状、dtype、设备、单位、时间频率和坐标基准；在系统边界校验并尽早失败。
- 不添加静默默认、自动降级、宽泛异常捕获、坏样本跳过或兼容回退来隐藏缺项。
- 不用平滑、裁剪、中心线投影、回退控制器、选择“更好”的随机 seed 或零轨迹掩盖模型失效。
- 优先使用成熟第三方库；没有明确复用需求时，不建立通用框架或额外抽象层。
- 未完成的 Issue 或 research 设想不得描述为已可用，也不得用假实现或占位入口伪造完成状态。

除非任务明确要求，或已有失败证据表明必要，否则不要新增：

- 文件、checkpoint、artifact 或依赖的 hash/checksum；
- baseline/reference/A-B/重复实验作为常规验证；
- 额外环境、硬件、依赖或文件完整性 preflight；
- manifest、版本握手、兼容/迁移层、冗余状态记录或仅针对理论故障的防御性检查。

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

验证只覆盖本次修改可能影响的行为：

- 文档修改：不运行代码测试。
- 局部实现修改：优先运行直接相关测试。
- bug 修复：必须包含并运行能复现问题的回归测试。
- 跨模块或公共行为修改：补充必要的快速测试。
- MetaDrive 交互、坐标转换、轨迹执行或仿真语义修改：运行直接覆盖该行为的 simulator 测试。
- 实验方法或指标修改：验证对应实验路径；只有结论依赖 baseline 对照时才运行 baseline。
- 依赖修改：检查 `pyproject.toml` 和 `uv.lock`。

依赖硬件或仿真器的测试应显式标记；缺少依赖时由命令排除 marker，测试本身不得静默跳过。

编码智能体运行在已准备好的 Windows 沙箱中。假定 `.venv` 可用；不要把 `just setup`、Python 版本探测或其他环境预检作为常规步骤，也不要使用 `uv run`。使用 `justfile` 提供的入口：

```powershell
just test
just lint
just format
```

运行 `just test`、`just test-target`、`just test-sim` 或 `just test-gpu` 时应请求沙箱外执行，不要重建环境或改用 `uv run`。

针对性验证通过且没有其他失败证据时停止。

## 文档写回

只在一个权威位置维护同一事实：

- 领域术语定义变化 → `CONTEXT.md`；只有高风险语义区分本身变化时才同步 `docs/agents/domain.md`。
- 已实现的数据或执行契约变化 → `docs/agents/system-contract.md`。
- 新的长期设计选择或已有 ADR 被取代 → 新建/更新对应 ADR。
- 尚未确定的研究假设 → `docs/research/`。
- 实际完成的实验 → `docs/experiments/records/` 并更新实验索引。
- 可执行但尚未完成的工作 → GitHub Issues。

不要为了“保持同步”在多个入口重复复制同一段实现细节。

## Git 与 Issue

提交信息使用简短、祈使式主题；一次提交只完成一个逻辑变更。

GitHub Issue 是 active work 的持久化规范入口；创建、读取、更新和关闭约定见 `docs/agents/issue-tracker.md`。Pull Request 可以引用 Issue，但不替代 Issue 作为任务规范。
