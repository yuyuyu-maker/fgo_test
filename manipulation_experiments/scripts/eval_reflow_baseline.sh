#!/usr/bin/env bash
# Post-training eval for baseline reflow runs (200 ep / 30 envs, matches eval_base_policies.sh).
#
# Usage:
#   STAMP=2026-08-26_22-15-00 bash scripts/eval_reflow_baseline.sh
#   GPU=0 TASK=Can STAMP=... bash scripts/eval_reflow_baseline.sh
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
STAMP="${STAMP:?Set STAMP from training run}"
TASK="${TASK:-}"  # empty = all tasks
CKPT_STEP="${CKPT_STEP:-latest}"
NUM_ENVS=30
NUM_EPISODES=200

TASKS=(
  "Can"
  "Square"
  "TwoArmBoxCleanup"
  "TwoArmLiftTray"
  "TwoArmThreading"
)
VARIANTS=(reflow reward_aware adaptive_compute fpo_operator theory)

resolve_ckpt() {
  local run_dir="$1"
  if [[ "$CKPT_STEP" != "latest" ]]; then
    echo "${run_dir}/checkpoints/${CKPT_STEP}"
    return
  fi
  ls -d "${run_dir}"/checkpoints/step_* 2>/dev/null | sort -V | tail -1
}

run_eval_task() {
  local task="$1"
  local task_slug
  task_slug="$(echo "$task" | tr '[:upper:]' '[:lower:]')"
  local log_dir="${ROOT}/logs/eval_reflow_baseline_${task_slug}_${STAMP}"
  mkdir -p "$log_dir"

  for variant in "${VARIANTS[@]}"; do
    run_dir="$(ls -d "${ROOT}/runs/flow_fpo_${task_slug}_${variant}_gpu"*"_${STAMP}" 2>/dev/null | head -1 || true)"
    if [[ -z "$run_dir" ]]; then
      echo "SKIP ${task}/${variant}: no run dir for stamp ${STAMP}"
      continue
    fi
    ckpt="$(resolve_ckpt "$run_dir")"
    if [[ -z "$ckpt" || ! -d "${ckpt}/policy" ]]; then
      echo "SKIP ${task}/${variant}: missing checkpoint under ${run_dir}"
      continue
    fi
    log_file="${log_dir}/${variant}.log"
    echo "[$(date '+%F %T')] eval ${task}/${variant} ckpt=${ckpt}"
    CUDA_VISIBLE_DEVICES="${GPU}" MUJOCO_EGL_DEVICE_ID="${GPU}" \
      python eval_checkpoint.py \
        --local_checkpoint_path "${ckpt}" \
        --eval_env "${task}" \
        --eval_num_episodes "${NUM_EPISODES}" \
        --eval-num-envs "${NUM_ENVS}" \
        --load-ema True \
        --wandb_enable False \
        --zero-sampling True \
        --save_video False \
        --experiment "eval_reflow_baseline_${task_slug}_${variant}" \
        > "${log_file}" 2>&1
    rg 'Success Rate:' "${log_file}" | tail -1 || tail -3 "${log_file}"
  done
}

if [[ -n "$TASK" ]]; then
  run_eval_task "$TASK"
else
  for t in "${TASKS[@]}"; do
    run_eval_task "$t"
  done
fi

echo "[$(date '+%F %T')] Eval done for stamp=${STAMP}"
