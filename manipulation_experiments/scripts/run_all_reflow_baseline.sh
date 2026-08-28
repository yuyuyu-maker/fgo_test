#!/usr/bin/env bash
# Dual-GPU pipeline: 5 tasks x 5 reflow variants, baseline-matched FPO++ hyperparameters.
#
#   GPU0: Can -> TwoArmBoxCleanup
#   GPU1: Square -> TwoArmLiftTray -> TwoArmThreading (waits for BC ckpt)
#
# Usage:
#   nohup bash scripts/run_all_reflow_baseline.sh > logs/reflow_baseline_pipeline.nohup.log 2>&1 &
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source source_env.sh

export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export TMPDIR="${TMPDIR:-/workspace/tmp}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
mkdir -p "$TMPDIR" "$ROOT/logs"

STAMP="${STAMP:-$(date +%Y-%m-%d_%H-%M-%S)}"
CKPT_ROOT="${ROOT}/downloaded_checkpoints"
PIPE_LOG="${ROOT}/logs/reflow_baseline_pipeline_${STAMP}.log"

exec > >(tee -a "$PIPE_LOG") 2>&1
echo "[$(date '+%F %T')] Reflow baseline pipeline stamp=${STAMP}"

wait_for_ckpt() {
  local ckpt_dir="$1"
  local label="$2"
  echo "[$(date '+%F %T')] Waiting for ${label} checkpoint under ${ckpt_dir}"
  while true; do
    local resolved
    resolved="$(resolve_ckpt_dir "$ckpt_dir")"
    if [[ -f "${resolved}/policy/model.safetensors" ]]; then
      echo "[$(date '+%F %T')] Ready: ${label} -> ${resolved}"
      return 0
    fi
    sleep 30
  done
}

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

run_gpu0() {
  GPU=0 STAMP="$STAMP" TASK=Can \
    CKPT_DIR="${CKPT_ROOT}/95j3noe4_step_1000" \
    bash "$ROOT/scripts/run_task_reflow_baseline.sh"
  GPU=0 STAMP="$STAMP" TASK=TwoArmBoxCleanup \
    CKPT_DIR="${CKPT_ROOT}/lainyisy_step_10000" \
    bash "$ROOT/scripts/run_task_reflow_baseline.sh"
  echo "[$(date '+%F %T')] GPU0 pipeline done"
}

run_gpu1() {
  GPU=1 STAMP="$STAMP" TASK=Square \
    CKPT_DIR="${CKPT_ROOT}/trc7rbt0_step_110000" \
    bash "$ROOT/scripts/run_task_reflow_baseline.sh"
  GPU=1 STAMP="$STAMP" TASK=TwoArmLiftTray \
    CKPT_DIR="${CKPT_ROOT}/ri0w9j39_step_20000" \
    bash "$ROOT/scripts/run_task_reflow_baseline.sh"
  wait_for_ckpt "${CKPT_ROOT}/6vqrn614_step_10000" "TwoArmThreading"
  GPU=1 STAMP="$STAMP" TASK=TwoArmThreading \
    CKPT_DIR="$(resolve_ckpt_dir "${CKPT_ROOT}/6vqrn614_step_10000")" \
    bash "$ROOT/scripts/run_task_reflow_baseline.sh"
  echo "[$(date '+%F %T')] GPU1 pipeline done"
}

run_gpu0 &
PID0=$!
run_gpu1 &
PID1=$!

wait "$PID0" "$PID1"
echo "[$(date '+%F %T')] ALL TASKS DONE stamp=${STAMP}" | tee "$ROOT/logs/reflow_baseline_pipeline_${STAMP}.done"
