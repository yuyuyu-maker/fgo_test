# Go2 Deploy (localhost command UI)

Unitree Go2 部署控制台：浏览器打开 `http://localhost:8080`，用摇杆 / WASD 发速度指令。  
模型切换走 **`configs/deploy.yaml`**（`active_model` 或 CLI `--model`）。

## 宇树官方 SDK（开源）

| 项目 | 链接 |
|------|------|
| **Python SDK2（本仓库用）** | https://github.com/unitreerobotics/unitree_sdk2_python |
| C++ SDK2（对照） | https://github.com/unitreerobotics/unitree_sdk2 |
| 快速开始 | https://support.unitree.com/home/en/developer/Quick_start |
| 运动服务 Sport | https://support.unitree.com/home/en/developer/sports_services |

安装到本目录 `thirdparty/`：

```bash
cd /nix/plsy/fgo_test/go2_deploy
source .venv/bin/activate
bash scripts/setup_unitree_sdk2.sh
```

然后在 `configs/deploy.yaml` 里：

```yaml
backend:
  name: sdk2
  iface: eth0   # 改成连狗的网卡
```

## 两种模式（网页顶部会显示）

| 模式 | 含义 | 启动 |
|------|------|------|
| **sport** | 宇树**内置步态** `SportClient.Move` | `--mode sport --backend sdk2` |
| **fpo** | **FPO ONNX 推理**（Python 50Hz；可选 LowCmd） | `--mode fpo --model baseline` |

先导出：`bash scripts/export_fpo_onnx.sh --model baseline`  
ONNX 落在 **`go2_deploy/exported/<model>/policy.onnx`**，部署时带整个 `go2_deploy/` 即可（不需要再带 `isaaclab_experiments` 的 `.pt`）。  
参数对齐见 `configs/fpo_go2_aligned.yaml` 与 [docs/DEPLOY_CPP_ONNX.md](docs/DEPLOY_CPP_ONNX.md)。

## 模型部署

- 推理后端：`go2_deploy/backends/fpo_onnx.py`（ONNX Runtime）
- 仍可对接 [unitree_rl_lab](https://github.com/unitreerobotics/unitree_rl_lab) C++（可选 `--udp-cmd`）
- 网页命令与策略均为 **50Hz**

## 快速开始

```bash
cd /nix/plsy/fgo_test/go2_deploy
source .venv/bin/activate
pip install -r requirements.txt

# 看有哪些模型、ckpt 是否存在
python -m go2_deploy --list-models

# 默认 sport（mock）；松开按键即暂停命令
python -m go2_deploy

# FPO 模式（50Hz ONNX 推理；默认 mock 状态）
python -m go2_deploy --mode fpo --model baseline

# Sport 真机内置步态
python -m go2_deploy --mode sport --backend sdk2 --iface eth0 --host 0.0.0.0
```

浏览器：`http://127.0.0.1:8080`  
健康检查 / 模型列表：`/api/health`、`/api/models`

> `--host 0.0.0.0` 时同网段可访问，无鉴权，只在可信网络用。

## 切换模型（YAML）

编辑 `configs/deploy.yaml`：

```yaml
active_model: baseline   # 改成 reflow / reward_aware / ...

models:
  baseline:
    checkpoint: ../isaaclab_experiments/logs/.../model_2499.pt
    fpo_variant: baseline
    sampling_steps: 10
    zero_sampling: true
```

当前已登记：`baseline`、`reflow`、`reward_aware`、`adaptive_compute`、`theory`、`all_ideas`、`fpo_operator_fixed`、`reflow_coef0p1`、`reflow_coef0p3`。  
训完新 run 后把对应 `checkpoint:` 改成最新 `model_*.pt` 即可。

网页在 **sport** 下走内置步态；在 **fpo** 下只转发速度命令给 C++。YAML 的 ckpt / `fpo_variant` 给 FPO 部署用。

## 操作

| 输入 | 效果 |
|------|------|
| 摇杆 / 触屏 | `vx`（前后）、`vy`（左右） |
| `W A S D` | 前后左右 |
| `Q E` | yaw |
| 空格 / STOP | 急停 |
| Stand Up / Down | 高层站立/趴下（sdk2） |

Deadman：超过 `server.deadman_s`（默认 0.35s）无更新则自动清零。

## 目录

```
go2_deploy/
  configs/deploy.yaml          # 后端 + 模型注册表
  scripts/setup_unitree_sdk2.sh
  thirdparty/unitree_sdk2_python/   # setup 脚本克隆
  go2_deploy/
    config.py
    server.py
    backends/mock.py | sdk2_sport.py
  static/
```
