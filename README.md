# Eco-AutoDrive

Eco-AutoDrive 是一个个人科研项目：保留预训练 Diffusion Planner 的驾驶能力，在 MetaDrive
长程闭环中通过 DPPO 优化能耗，并用 FASTSim 做精细能耗复核。

当前可运行成果是官方 EMA 权重的无交通、轨迹级运动学闭环；DPPO 训练、有交通观测、
FASTSim 计量和低层控制尚未完成。准确进度与已验证结论见 [STATUS.md](docs/STATUS.md)。

## 当前成果

- checkpoint-compatible Diffusion Planner、严格 EMA 权重加载、官方归一化和 10 步
  DPM-Solver++ baseline sampler；
- MetaDrive 程序化地图到官方固定形状张量的地图适配；
- 无交通 `S`/`SC` 场景的 2 Hz 滚动规划与 10 Hz 运动学轨迹执行；
- 可复现评测产物：resolved config、`summary.json`、`trace.npz` 和闭环 GIF；
- 对 PGMap 未设置限速哨兵和轨迹执行时序错误的回归修正。

代码入口位于 `src/eco_planner/`，测试位于 `tests/`，即时运行产物位于被 Git 忽略的
`outputs/`。可提交的实验索引及结论记录在 [experiments/README.md](experiments/README.md)。

## 环境准备

项目固定使用 Python 3.10 和 uv。官方权重应放在 `checkpoints/DP-Origin/`；该目录被 Git
忽略，所需文件哈希见 [LOGIC.md](docs/LOGIC.md#checkpoint-与上游契约)。本地 MetaDrive 源码应位于
`third_party/metadrive/`，由 `pyproject.toml` 以 editable dependency 使用。

Windows 本机用于编辑、格式化和不依赖 GPU 的快速验证：

```powershell
uv sync --all-groups
uv run pytest -m "not gpu and not simulator and not slow"
uv run ruff check .
uv run ruff format --check .
```

## 运行无交通闭环

正式 CUDA 评测读取 `configs/evaluation/no_traffic.yaml`：

```bash
uv run python -m eco_planner.evaluate
```

CPU 只用于单周期或短程链路检查，例如：

```powershell
uv run python -m eco_planner.evaluate model.device=cpu env.horizon=5 `
  'scenarios=[{name:straight,map:S,seed:0}]'
```

每次运行在 `outputs/no_traffic/` 下创建独立目录。该入口会拒绝背景车辆和静态交通物体，
不能用于有交通评测。

## Docker 基准环境

正式依赖、MetaDrive/FASTSim、CUDA 和完整仿真以 Ubuntu 22.04 / CUDA 12.4 Docker 环境为准。
服务器需预装 NVIDIA 驱动、Docker Engine 和 NVIDIA Container Toolkit：

```bash
export CUDA_VISIBLE_DEVICES=0
docker compose -f docker/compose.yaml build
docker compose -f docker/compose.yaml run --rm trainer
```

进入容器后可执行：

```bash
uv sync --frozen --all-groups
uv run pytest
```

## 离线部署

在能够构建 Linux/amd64 镜像的联网机器上生成离线包：

```bash
./scripts/build_offline_docker_bundle.sh /tmp/eco-planner-offline
```

将完整目录传到服务器后安装：

```bash
cd /path/to/eco-planner-offline
./install.sh /opt/eco-planner
cd /opt/eco-planner
export CUDA_VISIBLE_DEVICES=0
docker compose -f docker/compose.yaml run --rm trainer
```

离线包包含镜像、运行源码、本地固定的 MetaDrive 源码和 SHA-256 校验和；目标服务器必须是
x86_64 Linux，安装脚本不会覆盖已有目录。

## 文档导航

| 文件 | 内容 |
| --- | --- |
| [STATUS.md](docs/STATUS.md) | 当前完成度、已验证结论和下一步 |
| [LOGIC.md](docs/LOGIC.md) | 系统必须满足的逻辑与数据契约 |
| [DECISIONS.md](docs/DECISIONS.md) | 重要方案选择及其理由 |
| [CORRECTIONS.md](docs/CORRECTIONS.md) | 已发现错误、根因、修正和证据 |
| [experiments/README.md](experiments/README.md) | 可复现实验索引与结论 |
| [AGENTS.md](AGENTS.md) | 智能体工作规则和验证要求 |
