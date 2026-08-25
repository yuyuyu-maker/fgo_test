# RM75 单臂 Can / Square 仿真采数流水线

自建 **RM75-6FB** 单臂仿真，任务对齐 FPO RoboMimic 的 **Can**（抓放进容器）与 **Square**（窄槽插入），状态/动作/图像字段对齐真机 `realman_teleop_v2` 录制接口。

> 本仓库**不含**现成 RM75 Can/Square 公开数据集；下列数据均为脚本专家自采。

## 1. 读过并对齐的真机接口

| 文件 | 要点 |
|------|------|
| `realman_teleop_v2/.../utils.py` | 单臂 `ROBOT_STATE_NAMES` **14 维**；`joint_1…7` + `gripper` |
| `.../dataset_recorder_node.py` | `build_features()`：`observation.state` 与 `action` 同 shape/names；图像 `observation.images.{cam}_rgb`；**`action[t]=state[t+1]`**；`record_hz≈30`；`use_depth=false` |
| `.../config/teleop_params.yaml` | `home_joint_deg`、`cam0`/`cam1`、driver topics |

本模块 schema 固化在 `rm75_sim/schema.py`（与真机同名，单臂**不加** left/right 前缀）。

## 2. 仿真后端与理由

**选用 PyBullet（DIRECT）**

| 选项 | 为何不用 / 用 |
|------|----------------|
| Gazebo + `ros2_rm_robot` | 最同构，但本机容器无 Gazebo/ROS2 驱动栈，MVP 启动成本高 |
| Isaac Lab | 已有 FPO locomotion，但无 RM75 资产；重依赖，不适合本阶段自建 demo |
| **PyBullet** | 轻量、可 headless、易接脚本专家与 LeRobot 写出；**先跑通 Can 端到端** |

**Sim2Real gap（已知）**

- 夹爪为 URDF 上**自加的平行指**（官方 RM75-6FB URDF 无夹爪）；脚本专家用 **接近后 fixed constraint 软抓取**，保证 demo 成功率。
- 物理/摩擦/相机内参与真机不同；后续可用 Gazebo 同构采数，**success 定义保持不变**。

## 3. 目录结构

```
fgo_test/rm75_sim/
  assets/
    RM75-6FB/                 # 官方 URDF + meshes（RealMan rm_models）
    RM75-6FB_pybullet.urdf    # 相对路径 mesh + 合成夹爪 + ee_tip
    joint_map.json            # 仿真↔14D 映射（smoke 生成）
    SOURCE.txt
  rm75_sim/
    schema.py                 # 14D 名称（对齐真机）
    robot.py                  # 加载、home、IK、软抓取
    envs/can_env.py           # Can
    envs/square_env.py        # Square（窄槽）
    expert/can_script.py      # 模式 A：脚本专家
    dataset/features.py       # 对齐 build_features
    dataset/writer.py         # LeRobot v2.1 + action 后处理
    scripts/
      smoke_home.py
      collect_can.py
      collect_square.py
      check_dataset.py
  tests/test_schema.py
  data/                       # 输出数据集
  README.md
```

## 4. 14D 字段对照

| index | name | 单位 | 说明 |
|------:|------|------|------|
| 0–6 | `joint_*_rad` | rad | 与 URDF `joint_1…7` 顺序一致 |
| 7 | `gripper_open` | [0,1] | 1=开，0=关（真机同为归一化开度） |
| 8–10 | `eef_pos_*_m` | m | 世界系 `ee_tip` |
| 11–13 | `eef_rot_euler_*_rad` | rad | PyBullet XYZ 欧拉 |

图像：

- `observation.images.cam0_rgb`：腕部近似视角（240×320）
- `observation.images.cam1_rgb`：第三人称

Home（度，来自 `teleop_params.yaml`）：

`[-0.263, -0.505, -0.157, 43.953, 1.225, 121.751, -0.187]`

## 5. 成功定义

### Can

- 圆柱物体 XY 落在绿色目标区内（内半径 ×0.85），Z 在桌面附近
- 连续满足 `success_hold_steps`（默认 15）帧 → `is_success`

### Square

- 方块 XY 对齐窄槽中心（`xy_tol`）、偏航在容差内、Z 低于 `insert_z_max`
- 槽间隙约 2mm 级，**不是放大盘子**

## 6. 复现命令

使用 `fpo_manipulation` 环境（含 `lerobot` + `torch`；需 `pybullet`）：

```bash
MANIP=/nix/plsy/fgo_test/manipulation_experiments/thirdparty/miniconda3/envs/fpo_manipulation/bin/python
cd /nix/plsy/fgo_test/rm75_sim

# 单测 schema
$MANIP tests/test_schema.py

# M0：home / 关节表 / 夹爪
$MANIP rm75_sim/scripts/smoke_home.py

# M2：Can 脚本专家采 ≥5 条成功 episode（独立数据集）
$MANIP rm75_sim/scripts/collect_can.py --num-success 5 --max-attempts 20

# 检查字段
$MANIP rm75_sim/scripts/check_dataset.py data/local_rm75_sim_can_v0

# M3：Square 环境 smoke（可选 --collect N 写骨架包）
$MANIP rm75_sim/scripts/collect_square.py
```

数据集默认：

- Can：`data/local_rm75_sim_can_v0`（repo_id `local/rm75_sim_can_v0`）
- Square：`data/local_rm75_sim_square_v0`（**勿与 Can 混训**）

## 7. 采数模式说明

| 模式 | 本流水线 |
|------|----------|
| **A 脚本专家（优先）** | `CanScriptExpert`：IK 路径 + 软抓取约束，已用于 Can |
| B 人遥操作 | 未接 ROS2；真机仍用 `dataset_recorder_node`；仿真 dry_run 可后续接 |
| RL 微调 | **自动 rollout**，无需人边操作边录 |

`action` 写入约定与真机 VR 一致：`action[t] = state[t+1]`（末帧保持）。

## 8. 接到 FPO / RFPO 训练

`fgo_test/manipulation_experiments/pretrain_flow_bc.py` 使用 LeRobot v2.1，读 `observation.state` / `observation.images.*` / `action`。

对本数据集：

```bash
# 示例：把 --dataset 指到本包路径（具体 CLI 以 pretrain_flow_bc 为准）
# 注意相机 key 为 cam0_rgb / cam1_rgb，与 Panda robomimic 的 image key 不同，需在配置里改 image_observation_keys
```

在线 RL：`finetune_online_rl.py` 当前绑 Panda/DexMG；**RM75 需另接本 env 的 `reset/step/is_success`**（接口已留好，本阶段不强制改 RL 脚本）。

## 9. 里程碑状态

- [x] M0 资产 + home smoke + `joint_map.json`
- [x] M1 Can env + `is_success`
- [x] M2 脚本专家 → LeRobot ≥5 episode + `check_dataset`
- [x] M3 Square env reset/step/success（采数骨架可选）
- [x] M4 本文档

## 10. 已知限制

1. 官方 URDF 无夹爪 → 合成夹爪 + 软抓取  
2. IK 默认 **position-only**（带姿态时常不可达）  
3. 工作空间物体放在约 `x∈[0.28,0.36]`（相对基座）  
4. 图像分辨率 240×320，与真机 RealSense 可能不一致，meta 中已写 shape  
5. 未做深度对齐（与真机 `use_depth: false` 一致）
