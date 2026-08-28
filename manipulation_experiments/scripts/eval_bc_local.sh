#!/usr/bin/env bash
# Eval BC base policies from local downloaded_checkpoints/ (200 ep).
#
# Usage:
#   GPU=1 bash scripts/eval_bc_local.sh Can
#   GPU=1 bash scripts/eval_bc_local.sh Square
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source source_env.sh

GPU="${GPU:-1}"
TASK="${1:-Can}"
NUM_ENVS="${NUM_ENVS:-30}"
NUM_EPISODES="${NUM_EPISODES:-200}"

case "$TASK" in
  Can)
    CKPT="${ROOT}/downloaded_checkpoints/95j3noe4_step_1000"
    STEP=step_1000
    ;;
  Square)
    CKPT="${ROOT}/downloaded_checkpoints/trc7rbt0_step_110000"
    STEP=step_110000
    ;;
  *)
    echo "Unknown task: $TASK"; exit 1
    ;;
esac

LOG_DIR="${ROOT}/logs/eval_bc_${TASK,,}_local"
mkdir -p "$LOG_DIR"

for mode in zero random; do
  zs=True
  [[ "$mode" == "random" ]] && zs=False
  log="${LOG_DIR}/${mode}.log"
  echo "[$(date '+%F %T')] BC eval ${TASK} ${mode} -> ${log}"
  CUDA_VISIBLE_DEVICES="${GPU}" MUJOCO_EGL_DEVICE_ID="${GPU}" \
    python eval_checkpoint.py \
      --local_checkpoint_path "${CKPT}" \
      --eval_env "${TASK}" \
      --eval_num_episodes "${NUM_EPISODES}" \
      --eval-num-envs "${NUM_ENVS}" \
      --load-ema True \
      --wandb_enable False \
      --zero-sampling "${zs}" \
      --save_video False \
      --checkpoint_step "${STEP}" \
      --experiment "eval_bc_${TASK}_${mode}" \
      > "${log}" 2>&1
  rg 'Success Rate:' "${log}" | tail -1
done
