#!/usr/bin/env bash
# Export FPO++ Go2 checkpoint to policy.onnx for official unitree_rl_lab C++ deploy.
# Does not launch Isaac Sim.
# Usage:
#   bash scripts/export_go2_fpo_onnx.sh
#   CHECKPOINT=/path/to/model_1499.pt bash scripts/export_go2_fpo_onnx.sh
set -eo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /workspace/fgo_test/isaaclab_experiments/source_env.sh

export PYTHONPATH="${ROOT}/source/isaaclab_fpo:${PYTHONPATH:-}"
export TORCHDYNAMO_DISABLE=1

DEFAULT_CKPT="${ROOT}/logs/fpo/unitree_go2_flat_flow/2026-09-02_20-16-21_fpo_baseline_unitree_mdp/model_1499.pt"
CHECKPOINT="${CHECKPOINT:-$DEFAULT_CKPT}"
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "checkpoint not found: $CHECKPOINT"
  exit 1
fi

python "${ROOT}/scripts/fpo/export_onnx.py" --checkpoint "$CHECKPOINT"

RUN_DIR="$(dirname "$CHECKPOINT")"
ONNX="${RUN_DIR}/exported/policy.onnx"
if [[ ! -f "$ONNX" ]]; then
  echo "policy.onnx not found at $ONNX"
  exit 1
fi

CFG="${ROOT}/deploy/robots/go2/config/config.yaml"
python - << PY
from pathlib import Path
p = Path("${CFG}")
text = p.read_text()
old = "    policy_dir: ../../../logs/rsl_rl/unitree_go2_velocity"
new = "    policy_dir: ../../../logs/fpo/unitree_go2_flat_flow/2026-09-02_20-16-21_fpo_baseline_unitree_mdp"
if old in text and new not in text:
    text = text.replace(old, new + "\n    # policy_dir: ../../../logs/rsl_rl/unitree_go2_velocity  # official PPO")
    p.write_text(text)
    print("updated", p)
else:
    print("config.yaml already points at FPO run or unexpected format")
PY

echo "exported: $ONNX"
echo "deploy.yaml: ${RUN_DIR}/params/deploy.yaml"
echo "official C++ policy_dir is this run (params/deploy.yaml + exported/policy.onnx)"
ls -lh "$ONNX"
