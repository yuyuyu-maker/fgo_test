# C++ + ONNX Go2 policy deploy（FPO）

## 两种模式（网页会显示，不要混用）

| `control_mode` | 网页徽章 | 谁控腿 | 启动 |
|----------------|----------|--------|------|
| **sport** | SPORT · 内置步态 | 宇树 `SportClient.Move` | `python -m go2_deploy --mode sport --backend sdk2` |
| **fpo** | FPO · 模型 | **本仓库 Python ONNX** 50Hz（可加 `--fpo-hardware` 发 LowCmd） | `python -m go2_deploy --mode fpo --model baseline` |

FPO 默认 **mock 状态**（无 LowState，只跑推理）；真机加 `--fpo-hardware`（关 Sport，危险）。  
C++ 路径仍可参考 [unitree_rl_lab](https://github.com/unitreerobotics/unitree_rl_lab)；`--udp-cmd` 可把网页命令镜像出去。

网页命令流与策略均为 **50Hz**。

官方 PPO 的 Python 部署在独立目录 `/workspace/unitree_go2_ppo_deploy`，不要和本仓库 FPO 混用。
