#!/usr/bin/env bash
# Evaluate reflow-idea FPO++ runs (local checkpoints, paper-style 200 ep / 30 envs).
#
# Usage:
#   GPU=0 STAMP=2026-08-25_22-55-21 bash scripts/eval_reflow_ideas_final.sh
#   GPU=1 TASK=Square STAMP=2026-08-25_22-55-21 bash scripts/eval_reflow_ideas_final.sh
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source source_env.sh

export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export TMPDIR="${TMPDIR:-/workspace/tmp}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
mkdir -p "$TMPDIR"

GPU="${GPU:-0}"
TASK="${TASK:-Can}"
STAMP="${STAMP:-2026-08-25_22-55-21}"
CKPT_STEP="${CKPT_STEP:-step_499200}"
NUM_ENVS="${NUM_ENVS:-30}"
NUM_EPISODES="${NUM_EPISODES:-200}"
SAVE_VIDEO="${SAVE_VIDEO:-0}"

TASK_SLUG="$(echo "$TASK" | tr '[:upper:]' '[:lower:]')"
VARIANTS=(reflow reward_aware adaptive_compute fpo_operator theory)
LOG_DIR="${ROOT}/logs/eval_reflow_${TASK_SLUG}_${STAMP}"
mkdir -p "$LOG_DIR"

VIDEO_ARGS=(--save_video False)
if [[ "$SAVE_VIDEO" == "1" ]]; then
  VIDEO_ARGS=(--save_video True)
fi

for variant in "${VARIANTS[@]}"; do
  run_dir="$(ls -d "runs/flow_fpo_${TASK_SLUG}_${variant}_gpu"*"_${STAMP}" 2>/dev/null | head -1 || true)"
  if [[ -z "$run_dir" ]]; then
    echo "SKIP ${variant}: no run dir for stamp ${STAMP}"
    continue
  fi
  ckpt="${run_dir}/checkpoints/${CKPT_STEP}"
  log_file="${LOG_DIR}/${variant}.log"
  if [[ ! -d "$ckpt/policy" ]]; then
    echo "SKIP ${variant}: missing ${ckpt}"
    continue
  fi
  echo "[$(date '+%F %T')] GPU${GPU} eval ${TASK}/${variant} -> ${log_file}"
  CUDA_VISIBLE_DEVICES="${GPU}" MUJOCO_EGL_DEVICE_ID="${GPU}" \
    python eval_checkpoint.py \
      --local_checkpoint_path "${ckpt}" \
      --eval_env "${TASK}" \
      --eval_num_episodes "${NUM_EPISODES}" \
      --eval-num-envs "${NUM_ENVS}" \
      --load-ema True \
      --wandb_enable False \
      --checkpoint_step "${CKPT_STEP}" \
      --experiment "eval_reflow_${TASK_SLUG}_${variant}" \
      "${VIDEO_ARGS[@]}" \
      > "${log_file}" 2>&1
  rg 'Success Rate:|Evaluation completed' "${log_file}" | tail -2 || tail -5 "${log_file}"
done

echo "[$(date '+%F %T')] Done. Logs: ${LOG_DIR}"
