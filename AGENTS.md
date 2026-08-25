# Repository Guidelines

## 目标

Eco-AutoDrive 是个人科研代码库。优先保证**逻辑正确、实验语义明确、结论边界可追溯**，而不是生产系统式的通用性、防御性和兼容性。

完成当前任务所需的最小正确改动，并做足以证明该改动正确的验证。达到任务目标后停止；不要顺手扩展范围或继续寻找理论上的改进机会。

## 先定位权威来源

不要无差别加载文档。按任务只读取最小必要集合：

| 需要确认的内容 | 权威来源 |
| --- | --- |
| 高风险领域语义、易混淆概念 | `docs/agents/domain.md`；需要精确定义时再读 `CONTEXT.md` |
| 当前已实现的数据流、shape、单位、时间语义、训练/评测契约 | `docs/agents/system-contract.md` 的相关章节 + 对应代码和测试 |
| 当前任务、待完成工作、验收标准 | 当前用户请求 + 对应 GitHub Issue |
| 已接受的重要设计选择及理由 | 相关 `docs/adr/` |
| 尚未确定的方法、假设和研究问题 | `docs/research/` |
| 已运行实验、配置、结果和产物来源 | `docs/experiments/README.md`、对应 `records/`、实际 resolved config/artifact |
| 使用入口、依赖和可执行命令 | `justfile`、`README.md`、`pyproject.toml`、`uv.lock` |

判断当前实现事实时，优先级为：**代码和测试 > 机器可读配置/依赖 > authoritative docs > 其他说明**。Issue 和 research 文档描述目标或假设，不能代替当前实现事实。

发现冲突时不要自行融合成第三种解释。明确指出冲突，并按本次任务决定是否同步修正对应权威位置。

## 执行边界

- 回答、解释、评审或规划：检查相关材料并给出结论，不实现未请求的修改。
- 修改、修复或实现：直接完成范围内修改，并运行必要的非破坏性验证；普通读取、编辑和测试不需要反复请求确认。
- GitHub 写入、外部发布、破坏性操作或明显扩大范围：只有当前请求或明确工作流授权时才执行。
- 缺少真正必需的事实、文件或配置时，在最小阻塞点停止并说明缺失项；不要用猜测或静默默认继续。

## 科研实现原则

- 在真正的系统边界明确输入、配置、shape、dtype、device、单位、时间频率和坐标基准；内部受控数据流不要重复做同一层 defensive validation。
- 不添加静默默认、自动降级、宽泛异常捕获、坏样本跳过或兼容回退来掩盖缺项和失败。
- 不用平滑、裁剪、中心线投影、回退控制器、选择“更好”的 seed 或零轨迹掩盖模型失效。
- 优先使用成熟第三方库。第二个真实实现出现前，不为未来可能的扩展建立通用框架、factory、registry 或额外抽象层。
- 未完成的 Issue 或 research 设想不得描述为已可用，也不得用占位实现伪造完成状态。
- 除非任务明确要求或已有失败证据表明必要，不新增 checksum/hash、额外 preflight、manifest、版本握手、迁移/兼容层、冗余审计状态或常规 A/B/重复实验。

## 项目约定与 gotchas

- `configs/` 保存环境、模型、训练和实验配置；具有实验意义的参数不要硬编码进 Python。
- `third_party/metadrive/` 是运行时本地 editable 源码；`ref/` 是只读上游快照，业务代码不得从 `ref/` 导入。
- `checkpoints/`、`outputs/`、`.venv/`、`.env`、`ref/` 及被忽略的上游源码不得提交。
- 使用 Python 3.10；公共 API 有类型标注；只为非显然逻辑写简短 docstring。
- 遵循 Ruff 100 字符行宽及 `E/F/I/UP` 规则；路径使用 `pathlib.Path`，不得硬编码平台绝对路径。
- Hydra 必需字段必须显式提供，不用随意默认值隐藏缺项；依赖变更同时更新 `pyproject.toml` 和 `uv.lock`。
- 随机性由显式 seed 控制；实际研究运行记录真实使用的 seed。

编码智能体运行在已准备好的 Windows 沙箱中。假定 `.venv` 已可用；不要把 `just setup`、Python/依赖探测或其他环境 preflight 当作常规步骤，也不要改用 `uv run`。优先使用 `justfile` 的现有入口。

## 验证

验证只覆盖本次修改可能影响的行为：

- 文档修改：不运行代码测试；检查链接、职责边界和事实是否与权威来源一致。
- 局部实现修改：运行直接相关测试。
- bug 修复：增加并运行能复现问题的回归测试。
- 跨模块或公共行为修改：运行相关快速测试；涉及 MetaDrive、坐标、轨迹执行或仿真语义时运行对应 simulator 测试。
- 实验方法或指标修改：验证对应实验路径；只有结论依赖 baseline 时才运行 baseline。
- 依赖修改：检查 `pyproject.toml` 与 `uv.lock`。

常用入口：

```powershell
just test
just test-target <path-or-node>
just test-sim
just test-gpu
just lint
just format-check
just format
```

运行 `just test`、`just test-target`、`just test-sim` 或 `just test-gpu` 时按沙箱要求申请沙箱外执行；不要重建环境或切换到其他执行入口。

针对性验证通过且没有新的失败证据时停止。

## 文档写回

同一事实只维护一个权威位置：

- 稳定领域术语定义变化 → `CONTEXT.md`；只有高风险语义区分变化时才同步 `docs/agents/domain.md`。
- 已实现的数据或执行契约变化 → `docs/agents/system-contract.md`。
- 新的长期设计选择或既有 ADR 被取代 → `docs/adr/`。
- 尚未确定的研究假设 → `docs/research/`。
- 实际完成的研究实验 → `docs/experiments/records/` 并更新实验索引。
- 可执行但尚未完成的工作 → GitHub Issues。

不要为了“保持同步”在 README、Issue、ADR、system contract 和实验记录之间复制同一段实现细节。

## Git 与 GitHub Issues

提交信息使用简短、祈使式主题；一次提交只完成一个逻辑变更。

GitHub Issues 用于记录**已接受、可执行且需要跨会话跟踪的 implementation work**及其持久化验收标准，例如 bug、重构、性能和实验基础设施任务。尚未接受的研究设想留在 `docs/research/`；Issue 描述“要变成什么”，不作为当前实现事实。

当 Issue 是任务来源时，按需读取 body、comments 和 labels；动手前只保留会影响本次执行的任务契约：

- **Goal / Problem**：为什么要改；
- **Scope / Tasks**：本次允许修改什么；
- **Non-goals**：明确不做什么；
- **Acceptance criteria**：完成的可验证条件；
- **Dependencies / parent-child links**：仅在影响顺序或范围时保留。

当前用户请求与 Issue 冲突时，以当前用户请求作为本次任务边界并指出差异；不要未经请求修改 Issue 来消除冲突。

当前请求或工作流授权 GitHub 写入时：

- 创建 Issue 使用满足任务所需的最小结构：`Goal`、必要的 `Context`、`Scope`、必要的 `Non-goals`、`Acceptance criteria`；不要为了模板完整制造空章节。
- 更新 Issue 只记录任务状态、阻塞信息和新的验收信息。实现契约、长期设计理由和实验 provenance 写入各自权威位置后，从 Issue 引用，不复制全文。
- 只有验收标准已经满足，或明确决定取消/不实施时才关闭 Issue。若验收依赖真实实验，先在 `docs/experiments/` 登记运行证据，再从 Issue 引用对应记录。

优先使用当前环境提供的 GitHub 集成，不把特定 CLI 命令写成项目规范。只有没有可用集成而回退到 Windows `gh` CLI 时，才按沙箱要求在沙箱外执行认证相关请求；沙箱内 HTTP 401 可能只是 Windows credential manager 不可见。只有沙箱外 `gh auth status --hostname github.com` 也失败时才要求重新认证；不要输出 token，也不要使用 `--show-token`。
