# Go2 Ours — per-step ONNX (Unitree-Go2-Velocity)

Source ckpt: `unitree_rl_lab/.../all_ideas_ppo_teacher/.../model_1499.pt`

| dir | baked flow steps |
|-----|------------------|
| `ours_s64/` | 64 |
| `ours_s32/` | 32 |
| `ours_s16/` | 16 |
| `ours_s8/` | 8 |
| `ours_s4/` | 4 |
| `ours_s1/` | 1 |

Layout (C++ `unitree_rl_lab` deploy):

```
ours_s64/
  exported/policy.onnx
  params/deploy.yaml     # 45-D obs, kp/kd from training
  policy.onnx            # flat copy for go2_deploy path resolver
```

**Obs = 45-D** (no `base_lin_vel`). Use official C++ `go2_ctrl` + this `deploy.yaml`, **not** the Python 48-D Isaac-flat FPO backend.

## Re-export

```bash
cd /workspace/fgo_test/unitree_rl_lab
bash scripts/export_go2_ours_onnx_steps.sh
```

## Square-walk sim (Isaac, for similarity logs)

```bash
# one step budget
bash scripts/run_go2_square_walk.sh --steps 64
# all budgets
STEPS="64 32 16 8 4 1" bash scripts/run_go2_square_walk.sh --all-steps
```

Schedule: warmup 1s → forward 0.3 m/s × 3s → left × 3s → back × 3s → right × 3s (repeat).  
Outputs: `rollout.npz`, `cmd_schedule.csv`, plots under the run's `square_walk/` folder.

## Hardware / MuJoCo deploy command injector

Same schedule over WebSocket (pair with C++ + UDP cmd or your stack):

```bash
cd /workspace/fgo_test/go2_deploy
python scripts/square_walk_cmd.py --speed 0.3 --segment-s 3 --cycles 3 --log /tmp/cmd_hw.csv
```
