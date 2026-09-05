#!/usr/bin/env bash
# Export Go2 Ours ONNX for sampling steps 64/32/16/8/4/1.
# Usage:
#   bash scripts/export_go2_ours_onnx_steps.sh
#   CHECKPOINT=/path/model_1499.pt bash scripts/export_go2_ours_onnx_steps.sh
set -eo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Prefer isaaclab env if present
if [[ -f /workspace/fgo_test/isaaclab_experiments/source_env.sh ]]; then
  # shellcheck disable=SC1091
  source /workspace/fgo_test/isaaclab_experiments/source_env.sh
fi
export PYTHONPATH="${ROOT}/source/isaaclab_fpo:${PYTHONPATH:-}"
export TORCHDYNAMO_DISABLE=1

DEFAULT_CKPT="${ROOT}/logs/fpo/unitree_go2_all_ideas_ppo_teacher/2026-09-03_01-21-56_all_ideas_ppo_teacher_resume300/model_1499.pt"
CHECKPOINT="${CHECKPOINT:-$DEFAULT_CKPT}"
STEPS="${STEPS:-64 32 16 8 4 1}"

python "${ROOT}/scripts/fpo/export_go2_ours_steps.py" \
  --checkpoint "${CHECKPOINT}" \
  --steps ${STEPS}

echo
echo "ONNX dirs:"
echo "  ${CHECKPOINT%/*}/exported_steps/ours_s{64,32,16,8,4,1}/policy.onnx"
echo "  ../go2_deploy/exported/ours_s{64,32,16,8,4,1}/policy.onnx"
echo "Square-walk sim: bash scripts/run_go2_square_walk.sh --steps 64"
