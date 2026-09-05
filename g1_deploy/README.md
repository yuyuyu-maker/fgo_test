# G1 Deploy（29DoF）

Unitree **G1-29dof** 官方 C++ 部署包（从 `unitree_rl_lab/deploy` 抽出，可独立拷走）。  
控制器：`robots/g1_29dof/build/g1_ctrl`（ONNX Runtime + unitree_sdk2）。

对照 Go2 的 Python 网页控制台见 `../go2_deploy/`；G1 走官方 C++ FSM（Passive → FixStand → Velocity / Mimic）。

## 目录

```
g1_deploy/
  include/                 # FSM / isaaclab C++ 头文件
  robots/g1_29dof/         # g1_ctrl 源码 + config + 策略
    config/config.yaml
    config/policy/velocity/v0/          # 官方速度策略 ONNX
    config/policy/velocity/v1_ours/     # 自训 PPO 导出槽（export 后自动优先）
  thirdparty/              # onnxruntime（脚本下载）+ 可选 sdk2 源码
  scripts/
  exported/                # 导出 ONNX 备份
```

策略选择：`policy_dir` 指向父目录时，会取其中**字典序最大且含 `exported/`** 的子目录。  
因此导出到 `v1_ours` 后会优先于 `v0`；要强制官方策略，把 `config.yaml` 里改成：

```yaml
Velocity:
  policy_dir: config/policy/velocity/v0
```

## 依赖

```bash
sudo apt install -y libyaml-cpp-dev libboost-all-dev libeigen3-dev libspdlog-dev libfmt-dev cmake build-essential
```

| 组件 | 说明 |
|------|------|
| **unitree_sdk2**（C++） | https://github.com/unitreerobotics/unitree_sdk2 |
| **ONNX Runtime 1.22.0** | `bash scripts/download_onnxruntime.sh` |
| **unitree_mujoco**（可选 sim2sim） | https://github.com/unitreerobotics/unitree_mujoco |

## 一键构建

```bash
cd /workspace/fgo_test/g1_deploy

bash scripts/setup_unitree_sdk2.sh      # 装到 /usr/local（可能要 sudo）
bash scripts/download_onnxruntime.sh
bash scripts/build_g1_ctrl.sh

export LD_LIBRARY_PATH=/workspace/fgo_test/g1_deploy/thirdparty/onnxruntime-linux-x64-1.22.0/lib:$LD_LIBRARY_PATH
```

## Sim2Sim（MuJoCo）

1. 安装 [unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco)，`simulate/config.yaml`：`robot=g1`，`domain_id=0`，`enable_elastic_hand=1`，`use_joystick=1`
2. 起仿真：`cd unitree_mujoco/simulate/build && ./unitree_mujoco`
3. 起控制器：

```bash
cd robots/g1_29dof/build
./g1_ctrl
```

操作：

1. `[L2 + Up]` → FixStand 站起  
2. MuJoCo 窗口按 `8` 脚落地  
3. `[R1 + X]` → 跑 Velocity 策略  
4. MuJoCo 窗口按 `9` 关掉弹性吊带  

## Sim2Real

确认板载运动程序已关闭后：

```bash
./g1_ctrl --network eth0   # 改成连机器人的网卡名
```

## 导出自训 PPO → ONNX

在已配置好 Isaac Lab 的环境中（例如 `source ../isaaclab_experiments/source_env.sh`）：

```bash
# 指定 checkpoint
bash scripts/export_ppo_onnx.sh /dev/shm/unitree_g1_ppo_gpu1/.../model_9999.pt

# 或自动取某次 run 目录下最新 model_*.pt
bash scripts/export_ppo_onnx.sh --run-dir /path/to/logdir
```

会写入：

- `robots/g1_29dof/config/policy/velocity/v1_ours/exported/policy.onnx`
- `exported/v1_ours/policy.onnx`（备份）

然后重启 `g1_ctrl` 即可（无需重编）。

## 与训练仓库同步

若上游改了 deploy 代码：

```bash
bash scripts/sync_from_unitree_rl_lab.sh
bash scripts/build_g1_ctrl.sh
```

会保留本地 `v1_ours` 策略槽。

## 官方来源

- 上游：https://github.com/unitreerobotics/unitree_rl_lab （`deploy/`）
- 本包装载的官方速度 ONNX：`robots/g1_29dof/config/policy/velocity/v0/`
- Mimic（舞蹈等）策略也在同目录 `config/policy/mimic/`，由 `config.yaml` 中 FSM 切换

更多细节见 `docs/DEPLOY.md`。
