#!/usr/bin/env bash
# Play official Unitree-Go2-Velocity FPO checkpoint with Viser web UI.
# Usage:
#   GPU=0 PORT=8080 bash scripts/play_go2_fpo_viser.sh
#   CHECKPOINT=.../model_1499.pt FPO_VARIANT=all_ideas_teacher_kd STEPS=64 bash scripts/play_go2_fpo_viser.sh
set -eo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /workspace/fgo_test/isaaclab_experiments/source_env.sh

export PYTHONPATH="${ROOT}/source/isaaclab_fpo:${PYTHONPATH:-}"
export OMNI_KIT_ACCEPT_EULA=YES
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd.json}"
export PYTHONUNBUFFERED=1
export NVIDIA_DRIVER_CAPABILITIES="${NVIDIA_DRIVER_CAPABILITIES:-all}"
unset CUDA_VISIBLE_DEVICES || true
export TORCHDYNAMO_DISABLE=1

GPU="${GPU:-0}"
PORT="${PORT:-8080}"
NUM_ENVS="${NUM_ENVS:-16}"
STEPS="${STEPS:-64}"
FPO_VARIANT="${FPO_VARIANT:-all_ideas_teacher_kd}"
CKPT="${CHECKPOINT:-${ROOT}/logs/fpo/unitree_go2_all_ideas_ppo_teacher/2026-09-03_01-21-56_all_ideas_ppo_teacher_resume300/model_1499.pt}"
ASSET_DIR="${ASSET_DIR:-${ROOT}/viser_assets/unitree_go2_velocity}"
# Square-walk schedule (same as run_go2_square_walk.sh): fwd/left/back/right
SQUARE="${SQUARE:-1}"
SQUARE_SPEED="${SQUARE_SPEED:-0.3}"
SQUARE_SEGMENT_S="${SQUARE_SEGMENT_S:-3.0}"
SQUARE_WARMUP_S="${SQUARE_WARMUP_S:-1.0}"

cd "$ROOT"
echo "[$(date '+%F %T')] Viser play variant=${FPO_VARIANT} steps=${STEPS} port=${PORT} square=${SQUARE}"
echo "  ckpt=${CKPT}"
echo "  open http://localhost:${PORT} (ssh -L ${PORT}:localhost:${PORT} ...)"

extra=()
if [[ "${SQUARE}" == "1" ]]; then
  extra+=(--square-walk --square-speed "${SQUARE_SPEED}" --square-segment-s "${SQUARE_SEGMENT_S}" --square-warmup-s "${SQUARE_WARMUP_S}")
  echo "  square: |v|=${SQUARE_SPEED} m/s × ${SQUARE_SEGMENT_S}s per side (fwd→left→back→right)"
fi

python -u "${ROOT}/scripts/fpo/play_with_viser.py" \
  --task Unitree-Go2-Velocity \
  --headless \
  --device "cuda:${GPU}" \
  --num_envs "${NUM_ENVS}" \
  --fpo_variant "${FPO_VARIANT}" \
  --checkpoint "${CKPT}" \
  --flow-sampling-steps "${STEPS}" \
  --viser \
  --viser-port "${PORT}" \
  --asset-dir "${ASSET_DIR}" \
  --real-time \
  "${extra[@]}"
