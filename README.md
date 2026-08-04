# Eco-AutoDrive

通过强化学习优化扩散自动驾驶模型在长程运行中的能耗。

## 本地编辑环境

项目基线为 Python 3.10 和 uv。Windows 本机仅用于编辑、格式化及快速单元测试；MetaDrive、FASTSim 和 GPU 训练以 Linux Docker 环境为准。

```powershell
uv sync --all-groups
uv run pytest
uv run ruff check .
```

## 官方权重无交通闭环

`evaluate.py` 使用固定的官方 EMA checkpoint，在 MetaDrive 直道 `S` 和缓弯 `SC` 上以
2 Hz 重规划、10 Hz 运动学轨迹执行运行无交通闭环。正式 CUDA 评测命令为：

```bash
uv run python -m eco_planner.evaluate
```

每个场景在 `outputs/no_traffic/` 下生成 `summary.json`、`trace.npz` 和带规划/执行轨迹叠加的
`closed_loop.gif`。CPU 仅用于单周期链路检查，例如：

```powershell
uv run python -m eco_planner.evaluate model.device=cpu env.horizon=5 `
  'scenarios=[{name:straight,map:S,seed:0}]'
```

该入口会拒绝任何背景车辆或静态交通物体，不包含邻车历史、低密度交通、FASTSim 或 DPPO。
详见 [阶段 1 无交通闭环](docs/阶段1_无交通闭环.md)。

## Docker 训练环境

远程 Ubuntu 服务器需要 NVIDIA 驱动、Docker Engine 和 NVIDIA Container Toolkit。设置目标 GPU 后构建并进入容器：

```bash
export CUDA_VISIBLE_DEVICES=0
docker compose -f docker/compose.yaml build
docker compose -f docker/compose.yaml run --rm trainer
```

镜像使用 CUDA 12.4 / Ubuntu 22.04，并在 `/opt/venv` 安装锁定依赖。详见 [总体方案](docs/DiffusionPlanner_MetaDrive_DPPO_总体方案.md)。

## 离线部署到远程服务器

在有稳定网络、且 Docker 能构建 Linux/amd64 镜像的机器上执行（例如 Linux、WSL 或 Docker
Desktop）：

```bash
./scripts/build_offline_docker_bundle.sh /tmp/eco-planner-offline
```

将整个 `eco-planner-offline/` 目录传到服务器。该目录包含镜像、完整运行源码（包括本地固定的
MetaDrive 源码）以及 SHA-256 校验和。服务器无需在安装时访问镜像仓库或 Python 包索引：

```bash
cd /path/to/eco-planner-offline
./install.sh /opt/eco-planner
cd /opt/eco-planner
export CUDA_VISIBLE_DEVICES=0
docker compose -f docker/compose.yaml run --rm trainer
```

服务器仍须预装 NVIDIA 驱动、Docker Engine 和 NVIDIA Container Toolkit。离线包使用
`linux/amd64`；目标服务器必须是 x86_64 Linux。安装脚本不会覆盖已有目录。
