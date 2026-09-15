# Repository Guidelines

## 目标与完成条件

Eco-AutoDrive 是个人科研代码库。优先保证**逻辑正确、实验语义明确、结论边界可追溯**，而不是生产系统式的通用性、防御性和兼容性。

保护实验语义。baseline、算法、sampling、reward、metric、timing、termination、randomness 等研究定义的改变必须显式，不得作为普通重构顺带发生。

完成当前任务所需的最小正确改动，并做足以证明该改动正确的验证。针对性验证通过、验收标准满足且没有新的失败证据时停止；不要顺手扩展范围或继续寻找理论上的改进机会。

## 权威来源

只读取当前任务需要的材料：

| 需要确认的内容                       | 权威来源                                                                   |
| ----------------------------- | ---------------------------------------------------------------------- |
| 高风险领域语义、易混淆概念                 | `docs/agents/domain.md`；需要精确定义时再读 `CONTEXT.md`                         |
| 已实现的数据流、shape、单位、时间语义、训练/评测契约 | `docs/agents/system-contract.md` 的相关章节 + 对应代码和测试                       |
| 当前任务、范围和验收标准                  | 当前用户请求 + 对应 GitHub Issue                                               |
| 已接受的重要设计及理由                   | 相关 `docs/adr/`                                                         |
| 尚未确定的方法、假设和研究问题               | `docs/research/`                                                       |
| 已运行实验、配置、结果和产物来源              | `docs/experiments/README.md`、对应 `records/`、实际 resolved config/artifact |
| 使用入口、依赖和可执行命令                 | `justfile`、`README.md`、`pyproject.toml`、`uv.lock`                      |

判断**当前实现事实**时，优先级为：

**代码和测试 > 机器可读配置/依赖 > authoritative docs > 其他说明**

Issue 和 research 文档描述目标、计划或假设，不能代替当前实现事实。

发现来源冲突时，不自行融合成第三种解释。明确指出冲突，并根据当前任务决定是否修正相应权威位置。

## 执行边界

* **回答、解释、评审、诊断或规划**：检查相关材料并给出结论，不实现未请求的修改。
* **修改、修复或实现**：直接完成范围内的本地修改，并运行必要的非破坏性验证。读取文件、编辑范围内代码、运行测试以及修复由本次修改造成的失败，不需要逐步请求确认。
* **GitHub 写入、外部发布、破坏性操作或明显扩大任务范围**：只有当前请求或明确工作流已授权时才执行。
* 优先从当前请求、权威文档、代码、配置和测试中解决局部歧义。只有缺失信息会实质改变任务目标、实验语义或不可逆决策，且无法从现有来源可靠确定时，才在最小阻塞点停止并说明缺失项。

不要用猜测、静默默认或兼容回退替代真正缺失的任务契约。

## 科研实现原则

### 保持科学语义显式

* 输入、配置、shape、dtype、device、单位、时间频率、坐标基准、episode boundary、RNG 和 seed 等具有研究意义的约束属于实验契约。
* 在真正的系统边界验证这些契约；内部受控数据流不要重复增加同一层 defensive validation。
* 具有实验意义的参数由 `configs/` / Hydra 配置拥有，不硬编码进 Python。
* Hydra 必需字段必须显式提供，不用随意默认值隐藏缺项。
* 随机性由显式 seed 控制；实际研究运行记录真实使用的 seed。
* 实验配置与执行资源分离；换机器或执行环境不得静默改变实验定义。

### 让失败保持可见

不添加会掩盖模型、数据或配置失效的：

* 静默默认或自动降级；
* 宽泛异常捕获；
* 坏样本跳过或缺失值伪造；
* 隐式 clip、repair 或 fallback；
* 为得到更好结果而选择 seed；
* 用平滑、中心线投影、回退控制器或零轨迹掩盖模型失败。

保留原始失败和 undefined 状态，除非研究方法本身明确规定如何处理。

### 控制工程复杂度

* 优先使用成熟第三方库，不重复实现已有可靠机制。
* 第二个真实实现出现前，不为未来可能的扩展建立 factory、registry、plugin system 或额外抽象层。
* 未明确要求或没有失败证据时，不新增 checksum/hash、额外 preflight、manifest、版本握手、迁移/兼容层、冗余审计状态或常规 A/B/重复实验。
* 内部接口默认直接切换；未要求兼容时，不增加 legacy aliases、forwarders、schema migration 或兼容分支。
* 未完成的 Issue 或 research 设想不得描述为已可用，也不得用占位实现伪造完成状态。

### 按所有权组织代码

* `scripts/` 只保留 CLI 参数解析、bootstrap、展示和退出码映射。
* 稳定的 repository application logic 位于 `src/eco_planner/`，但不因此成为第三方 public API。
* 通用机制放到拥有该机制的最低稳定层；experiment 只编排研究协议，下层不得依赖具体 experiment。
* 复用机制，不合并语义：成熟库可以拥有通用基础设施，项目特有的数据、状态、audit 和科学契约由本项目自己拥有。
* 逻辑边界和 dependency direction 优先于目录层级；不要为了结构整齐预建框架。

测试验证的是软件契约，不自动证明安全、节能、parity 或其他科研结论。

## Repository gotchas

* `justfile` 是 Windows PowerShell 下的无语义 task alias；工作流参数和默认值由对应 CLI/Hydra 配置拥有。
* `third_party/metadrive/` 是运行时本地 editable 源码。
* `ref/` 是只读上游快照；业务代码不得从 `ref/` 导入。
* `checkpoints/`、`outputs/`、`.venv/`、`.env`、`ref/` 以及被忽略的上游源码不得提交。
* 使用 Python 3.10。
* 公共接口提供类型标注；只为非显然逻辑写简短 docstring。
* 遵循 Ruff 100 字符行宽及 `E/F/I/UP` 规则。
* 路径使用 `pathlib.Path`，不得硬编码平台绝对路径。
* 依赖变更同时更新 `pyproject.toml` 和 `uv.lock`。

编码智能体运行在已准备好的 Windows 沙箱中。假定 `.venv` 已可用；不要把 `just setup`、Python/依赖探测或其他环境 preflight 当作常规步骤，也不要改用 `uv run`。优先使用 `justfile` 已有入口。

## 验证

验证强度与本次修改的**语义风险和影响范围**匹配。优先运行直接覆盖修改行为的最小验证，不因“改了代码”就机械扩大测试范围。

* 文档修改：不运行代码测试；检查链接、职责边界以及事实是否与权威来源一致。
* 局部实现修改：运行直接相关测试。
* bug 修复：增加或更新能复现问题的回归测试，并运行该测试。
* 跨模块或公共行为修改：运行受影响的快速测试。
* 涉及 MetaDrive、坐标、轨迹执行或仿真语义：运行对应 simulator 测试。
* 实验方法或指标修改：验证对应实验路径；只有结论依赖 baseline 时才运行 baseline。
* 依赖修改：确认 `pyproject.toml` 与 `uv.lock` 一致。

常用入口：

```powershell
just test
just test-target <path-or-node>
just test-sim
just test-gpu
just lint
just format
just typecheck
```

运行 `just test`、`just test-target`、`just test-sim` 或 `just test-gpu` 时，按沙箱要求申请沙箱外执行；不要重建环境或切换到其他执行入口。

只有出现新的失败、受影响范围无法可靠界定、修改继续扩大或仍存在实质性未验证风险时，才扩大或重复验证。

针对性验证通过且验收标准满足后停止。

## 文档写回

只在本次任务改变了相应事实时更新权威文档。同一事实只维护一个权威位置：

* 稳定领域术语或定义变化 → `CONTEXT.md`；
* 高风险领域语义区分变化 → 同步相关 `docs/agents/domain.md`；
* 已实现的数据或执行契约变化 → `docs/agents/system-contract.md`；
* 新的长期设计选择或既有 ADR 被取代 → `docs/adr/`；
* 尚未确定的研究假设 → `docs/research/`；
* 实际完成的研究实验 → `docs/experiments/records/` 并更新实验索引；
* 可执行但尚未完成的 implementation work → GitHub Issues。

不要为了“保持同步”在 README、Issue、ADR、system contract 和实验记录之间复制同一段实现细节；从非权威位置引用权威来源。

## Git 与 GitHub Issues

提交信息使用简短、祈使式主题；一次提交只完成一个逻辑变更。

GitHub Issues 用于记录**已接受、可执行且需要跨会话跟踪的 implementation work**及其持久化验收标准，例如 bug、重构、性能和实验基础设施任务。尚未接受的研究设想留在 `docs/research/`。

Issue 描述“要变成什么”，不作为当前实现事实。

当 Issue 是任务来源时，只按需要读取 body、comments 和 labels，并从中确定：

* **Goal / Problem**：为什么要改；
* **Scope / Tasks**：本次允许修改什么；
* **Non-goals**：明确不做什么；
* **Acceptance criteria**：怎样证明完成；
* **Dependencies / parent-child links**：只在影响执行顺序或范围时使用。

当前用户请求与 Issue 冲突时，以当前用户请求作为本次任务边界并指出差异；不要未经请求修改 Issue 来消除冲突。

只有当前请求或工作流授权 GitHub 写入时：

* 创建 Issue 使用完成任务所需的最小结构，不为模板完整制造空章节。
* 更新 Issue 只记录任务状态、阻塞信息和新的验收信息；实现契约、长期设计理由和实验 provenance 写入各自权威位置后从 Issue 引用。
* 只有验收标准已满足，或明确决定取消/不实施时才关闭 Issue。
* 若验收依赖真实实验，先在 `docs/experiments/` 登记运行证据，再从 Issue 引用对应记录。

优先使用当前环境提供的 GitHub 集成。只有没有可用集成而回退到 Windows `gh` CLI 时，认证相关请求才按沙箱要求在沙箱外执行。沙箱内 HTTP 401 可能只是 Windows credential manager 不可见；只有沙箱外 `gh auth status --hostname github.com` 也失败时才要求重新认证。不要输出 token，也不要使用 `--show-token`。
