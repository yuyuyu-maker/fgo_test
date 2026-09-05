# G1 Deploy 细节

## FSM 与按键（手柄）

来自 `robots/g1_29dof/config/config.yaml`：

| 当前状态 | 按键 | 下一状态 |
|----------|------|----------|
| Passive | L2 + Up | FixStand |
| FixStand | L2 + B | Passive |
| FixStand | R1 + X | Velocity |
| Velocity | L2 + B | Passive |
| Velocity | L2(2s) + Down | Mimic Dance |
| Velocity | L2(2s) + Left | Mimic Gangnam |

真机务必确认 **mode_machine = 5（29DoF）**，与 `main.cpp` 一致。

## 观测 / 动作对齐

训练 `velocity_env_cfg.py`、训练日志里的 `params/deploy.yaml`、官方 `v0/params/deploy.yaml` 在 **obs scale / action / history / step_dt / joint_ids_map** 上一致：

| 项 | 值 |
|----|-----|
| `base_ang_vel` scale | 0.2 |
| `projected_gravity` | 1.0 |
| `velocity_commands` | 1.0 |
| `joint_pos_rel` | 1.0 |
| `joint_vel_rel` | 0.05 |
| `last_action` | 1.0 |
| `history_length` | 5（每项） |
| action scale | 0.25 |
| `step_dt` | 0.02 |
| cmd ranges | vx[-0.5,1], vy[-0.3,0.3], yaw[-0.2,0.2] |

**唯一系统性差异：kp/kd（stiffness/damping）**

| 关节 | 训练（`UNITREE_G1_29DOF_CFG`） | 官方 v0 |
|------|-------------------------------|---------|
| `waist_roll` / `waist_pitch` stiffness | 40 | 200 |
| 手臂腕部 damping（SDK idx 15–28） | 1.0 | 10.0 |

- 跑官方 `v0` ONNX → 用 `velocity/v0/params/deploy.yaml`
- 跑自训 PPO/FPO → **必须用训练导出的** `params/deploy.yaml`（已放进 `v1_ours/`；`export_ppo_onnx.sh` 会自动拷）

参考副本：`docs/deploy_from_training_ppo.yaml`。
## 运行时库路径

```bash
export LD_LIBRARY_PATH="$(pwd)/thirdparty/onnxruntime-linux-x64-1.22.0/lib:${LD_LIBRARY_PATH}"
```

若 `g1_ctrl` 报找不到 `libonnxruntime.so.1.22.0`，多半是未设置该变量。

## 网络接口

`--network eth0` 传给 unitree ChannelFactory；仿真默认本地即可：

```bash
./g1_ctrl
# 或
./g1_ctrl --network lo
```

## 故障排查

| 现象 | 处理 |
|------|------|
| `Unmatched robot type` | 确认是 G1 29DoF，且无其它进程占用 lowcmd |
| `other process is using the lowcmd channel` | 关掉板载 / 其它控制器 |
| cmake 找不到 onnxruntime | `bash scripts/download_onnxruntime.sh` |
| 链接失败 unitree_sdk2 | `bash scripts/setup_unitree_sdk2.sh`，检查 `/usr/local/lib` |
| 策略站不稳 / 乱扭 | 核对 deploy.yaml 与训练一致；先用官方 `v0` 验证链路 |
