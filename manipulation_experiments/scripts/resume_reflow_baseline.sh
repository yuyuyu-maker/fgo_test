#!/usr/bin/env bash
# Resume baseline reflow pipeline after crash (no eval videos, auto-resume checkpoints).
#
# Usage:
#   nohup bash scripts/resume_reflow_baseline.sh > logs/resume_reflow_baseline.nohup.log 2>&1 &
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="${STAMP:-2026-08-26_22-18-01}"
CKPT_ROOT="${ROOT}/downloaded_checkpoints"
PIPE_LOG="${ROOT}/logs/resume_reflow_baseline_${STAMP}.log"

resolve_ckpt_dir() {
  local ckpt_dir="$1"
  if [[ -f "${ckpt_dir}/policy/model.safetensors" ]]; then
    echo "${ckpt_dir}"
    return
  fi
  local nested
  nested="$(find "${ckpt_dir}" -type f -path '*/policy/model.safetensors' 2>/dev/null | head -1 || true)"
  if [[ -n "$nested" ]]; then
    dirname "$(dirname "$nested")"
    return
  fi
  echo "${ckpt_dir}"
}

exec > >(tee -a "$PIPE_LOG") 2>&1
echo "[$(date '+%F %T')] Resume reflow baseline stamp=${STAMP}"

run_gpu0() {
  GPU=0 STAMP="$STAMP" SAVE_EVAL_VIDEO=0 \
    START_VARIANT=reflow \
    OUTPUT_DIR="${ROOT}/runs/flow_fpo_can_reflow_gpu0_${STAMP}" \
    RESUME_CKPT="${ROOT}/runs/flow_fpo_can_reflow_gpu0_${STAMP}/checkpoints/latest" \
    TASK=Can CKPT_DIR="${CKPT_ROOT}/95j3noe4_step_1000" \
    bash "${ROOT}/scripts/run_task_reflow_baseline.sh"
  GPU=0 STAMP="$STAMP" SAVE_EVAL_VIDEO=0 TASK=TwoArmBoxCleanup \
    CKPT_DIR="${CKPT_ROOT}/lainyisy_step_10000" \
    bash "${ROOT}/scripts/run_task_reflow_baseline.sh"
  echo "[$(date '+%F %T')] GPU0 pipeline done"
}

run_gpu1() {
  GPU=1 STAMP="$STAMP" SAVE_EVAL_VIDEO=0 \
    START_VARIANT=reflow \
    OUTPUT_DIR="${ROOT}/runs/flow_fpo_square_reflow_gpu1_${STAMP}" \
    RESUME_CKPT="${ROOT}/runs/flow_fpo_square_reflow_gpu1_${STAMP}/checkpoints/latest" \
    TASK=Square CKPT_DIR="${CKPT_ROOT}/trc7rbt0_step_110000" \
    bash "${ROOT}/scripts/run_task_reflow_baseline.sh"
  GPU=1 STAMP="$STAMP" SAVE_EVAL_VIDEO=0 TASK=TwoArmLiftTray \
    CKPT_DIR="${CKPT_ROOT}/ri0w9j39_step_20000" \
    bash "${ROOT}/scripts/run_task_reflow_baseline.sh"
  THREAD_CKPT="$(resolve_ckpt_dir "${CKPT_ROOT}/6vqrn614_step_10000")"
  while [[ ! -f "${THREAD_CKPT}/policy/model.safetensors" ]]; do
    echo "[$(date '+%F %T')] waiting for Threading BC ckpt..."
    THREAD_CKPT="$(resolve_ckpt_dir "${CKPT_ROOT}/6vqrn614_step_10000")"
    sleep 30
  done
  GPU=1 STAMP="$STAMP" SAVE_EVAL_VIDEO=0 TASK=TwoArmThreading \
    CKPT_DIR="${THREAD_CKPT}" \
    bash "${ROOT}/scripts/run_task_reflow_baseline.sh"
  echo "[$(date '+%F %T')] GPU1 pipeline done"
}

run_gpu0 &
PID0=$!
run_gpu1 &
PID1=$!

wait "$PID0" "$PID1"
echo "[$(date '+%F %T')] ALL TASKS DONE stamp=${STAMP}" | tee "${ROOT}/logs/resume_reflow_baseline_${STAMP}.done"
