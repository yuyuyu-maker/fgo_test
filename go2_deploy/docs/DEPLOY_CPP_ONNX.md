# C++ + ONNX Go2 policy deploy（FPO）

## 两种模式（网页会显示，不要混用）

| `control_mode` | 网页徽章 | 谁控腿 | 启动 |
|----------------|----------|--------|------|
| **sport** | SPORT · 内置步态 | 宇树 `SportClient.Move` | `python -m go2_deploy --mode sport --backend sdk2` |
| **fpo** | FPO · 模型 | **本仓库 Python ONNX** 50Hz（可加 `--fpo-hardware` 发 LowCmd） | `python -m go2_deploy --mode fpo --model baseline` |

FPO 默认 **mock 状态**（无 LowState，只跑推理）；真机加 `--fpo-hardware`（关 Sport，危险）。  
C++ 路径仍可参考 [unitree_rl_lab](https://github.com/unitreerobotics/unitree_rl_lab)；`--udp-cmd` 可把网页命令镜像出去。

网页命令流与策略均为 **50Hz**。

---

## 对齐参数

- `configs/deploy.yaml` → `policy.*`
- `configs/fpo_go2_aligned.yaml`

| 项 | 值 |
|----|-----|
| `policy_hz` / 网页上报 | **50** |
| `action_scale` | **0.25** |
| `kp` / `kd` | **25.0** / **0.5** |

## 导出 ONNX + 跑推理

```bash
bash scripts/export_fpo_onnx.sh --model baseline
python -m go2_deploy --mode fpo --model baseline
# 健康检查里看 policy_status.ticks / last_infer_ms
curl -s http://127.0.0.1:8080/api/health | python -m json.tool
```

后端代码：`go2_deploy/backends/fpo_onnx.py` + `go2_deploy/policy/`。

---

## 频率 / 按键

- **频率可改**：`policy.policy_hz`（超参；默认锁 50）。
- **不是**按一次键推理一次：环路固定 50Hz；按键只改命令；松开 → `(0,0,0)`。

---

## 推荐参考

| 仓库 | 说明 |
|------|------|
| **[unitree_rl_lab](https://github.com/unitreerobotics/unitree_rl_lab)** | 官方 Go2 C++/ONNX |
| [unitree_cpp_deploy](https://github.com/wty-yy/unitree_cpp_deploy) | 社区 Go2/Orin 说明更细 |
| [unitree_sdk2](https://github.com/unitreerobotics/unitree_sdk2) | C++ DDS |

```bash
bash scripts/clone_unitree_rl_lab.sh
```

## 流水线

1. `bash scripts/export_fpo_onnx.sh --model baseline`
2. 把 ONNX + `fpo_go2_aligned.yaml` 参数并进 `unitree_rl_lab/deploy/robots/go2`
3. 关 Sport → 跑 `go2_ctrl`；网页 `--mode fpo` 经 UDP 发 `(vx,vy,yaw)`
