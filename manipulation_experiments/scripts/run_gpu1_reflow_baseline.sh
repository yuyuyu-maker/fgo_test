#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="${STAMP:-2026-08-26_22-18-01}"
CKPT_ROOT="${ROOT}/downloaded_checkpoints"

GPU=1 STAMP="$STAMP" TASK=Square CKPT_DIR="${CKPT_ROOT}/trc7rbt0_step_110000" \
  bash "${ROOT}/scripts/run_task_reflow_baseline.sh"
GPU=1 STAMP="$STAMP" TASK=TwoArmLiftTray CKPT_DIR="${CKPT_ROOT}/ri0w9j39_step_20000" \
  bash "${ROOT}/scripts/run_task_reflow_baseline.sh"
THREAD_CKPT="${CKPT_ROOT}/6vqrn614_step_10000"
if [[ ! -f "${THREAD_CKPT}/policy/model.safetensors" && -f "${THREAD_CKPT}/6vqrn614_step_10000/policy/model.safetensors" ]]; then
  THREAD_CKPT="${THREAD_CKPT}/6vqrn614_step_10000"
fi
while [[ ! -f "${THREAD_CKPT}/policy/model.safetensors" ]]; do
  echo "[$(date '+%F %T')] waiting for Threading BC ckpt..."
  sleep 30
done
GPU=1 STAMP="$STAMP" TASK=TwoArmThreading CKPT_DIR="${THREAD_CKPT}" \
  bash "${ROOT}/scripts/run_task_reflow_baseline.sh"
echo "[$(date '+%F %T')] GPU1 pipeline done"
