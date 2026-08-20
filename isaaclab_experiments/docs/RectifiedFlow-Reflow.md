# Rectified Flow Reflow（简要索引）

详细实现说明见：**[RectifiedFlow-Reflow-实现说明.md](./RectifiedFlow-Reflow-实现说明.md)**

---

## 一句话

在 FPO 更新时加 **Reflow 辅助损失**：用当前 flow 生成 `(噪声, 动作)` 耦合对，在直线上训练速度场，使传输路径更直。

## 快速开启

```bash
python isaaclab_fpo/scripts/train.py \
  --task Isaac-Velocity-Flat-Unitree-Go2-v0 \
  --headless \
  agent.algorithm.reflow_enabled=True \
  agent.algorithm.reflow_loss_coef=1.0 \
  agent.algorithm.reflow_n_samples_per_obs=4 \
  agent.experiment_name=unitree_go2_flat_flow_reflow \
  --run_name reflow_v1
```

## 日志

- `reflow_loss`（TensorBoard / 终端）

## 代码入口

- `modules/actor_critic.py` → `get_reflow_loss()`
- `algorithms/fpo.py` → `FPO.update()`
